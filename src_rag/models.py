import numpy as np
import re
import tiktoken
import openai
import yaml
from dotenv import load_dotenv
import os
from FlagEmbedding import FlagModel

load_dotenv()  # charge le fichier .env

CONF = yaml.safe_load(open("config.yml"))

GROQ_KEY = os.getenv("GROQ_KEY") or CONF.get("groq_key", "")

CLIENT = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_KEY,
)

tokenizer = tiktoken.get_encoding("cl100k_base")


def get_model(config):
    if config:
        return RAG(**config.get("model", {}))
    else:
        return RAG()


class RAG:
    def __init__(self, chunk_size=256, top_k=5):
        self._chunk_size = chunk_size
        self._top_k = top_k          
        self._embedder = None
        self._loaded_files = set()
        self._texts = []
        self._chunks = []
        self._corpus_embedding = None
        self._client = CLIENT

    def load_files(self, filenames):
        texts = []
        for filename in filenames:
            if filename in self._loaded_files:
                continue

            with open(filename, "r", encoding="utf-8", errors="ignore") as f:
                texts.append(f.read())
                self._loaded_files.add(filename)

        self._texts += texts

        chunks_added = self._compute_chunks(texts)
        self._chunks += chunks_added

        new_embedding = self.embed_corpus(chunks_added)
        if self._corpus_embedding is not None:
            self._corpus_embedding = np.vstack(
                [self._corpus_embedding, new_embedding]
            )
        else:
            self._corpus_embedding = new_embedding

    def get_corpus_embedding(self):
        return self._corpus_embedding

    def get_chunks(self):
        return self._chunks

    # -------- Embedding --------
    def embed_questions(self, questions):
        embedder = self.get_embedder()
        if hasattr(embedder, "encode_queries"):
            emb = embedder.encode_queries(questions)
        else:
            emb = embedder.encode(questions)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return emb

    def embed_corpus(self, chunks):
        embedder = self.get_embedder()
        emb = embedder.encode(chunks)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return emb

    def get_embedder(self):
        if not self._embedder:
            self._embedder = FlagModel(
                "BAAI/bge-m3",  # ou bge-m3 si tu veux tester "BAAI/bge-m3" "BAAI/bge-large-en-v1.5"
                query_instruction_for_retrieval=(
                    "Represent this sentence for searching relevant passages:"
                ),
                use_fp16=True,
            )
        return self._embedder

    # -------- Chunking --------
    def small2big_chunking(self, md_text: str):
        sizes = [128, 256, 512]
        chunks = []
        for size in sizes:
            new_chunks = chunk_markdown(md_text, chunk_size=size)
            chunks.extend(new_chunks)
        return chunks

    def chunk_with_overlap(self, md_text: str, chunk_size: int = 256, overlap: int = 50):
        tokens = tokenizer.encode(md_text)
        chunks = []
        step = chunk_size - overlap

        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i:i + chunk_size]
            if not chunk_tokens:
                break

            chunk_text = tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

            if len(chunk_tokens) < chunk_size:
                break

        return chunks

    def _compute_chunks(self, texts):
        if self._chunk_size == "small2big":
            return sum((self.small2big_chunking(txt) for txt in texts), [])

        elif isinstance(self._chunk_size, dict) and "overlap" in self._chunk_size:
            size = self._chunk_size["size"]
            ov = self._chunk_size["overlap"]
            return sum((self.chunk_with_overlap(txt, size, ov) for txt in texts), [])

        else:
            return sum(
                (chunk_markdown(txt, chunk_size=self._chunk_size) for txt in texts),
                [],
            )

    # -------- RAG --------
    def reply(self, query):
        prompt = self._build_prompt(query)
        res = self._client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-oss-20b",
        )
        return res.choices[0].message.content

    def _build_prompt(self, query):
        context_str = "\n".join(self._get_context(query))
        return f"""Context information is below.

---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
If the answer is not in the context information, reply "I cannot answer that question".
Query: {query}
Answer:"""

    def _get_context(self, query):
        query_embedding = self.embed_questions([query])
        sim_scores = query_embedding @ self._corpus_embedding.T

        # 👇 on utilise le top_k de l’instance
        indexes = list(np.argsort(sim_scores[0]))[-self._top_k:][::-1]

        return [self._chunks[i] for i in indexes]


# -------- Utils --------
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


def parse_markdown_sections(md_text: str) -> list[dict[str, str]]:
    pattern = re.compile(r"^(#{1,6})\s*(.+)$")
    lines = md_text.splitlines()

    sections = []
    header_stack = []
    current_section = {"headers": [], "content": ""}

    for line in lines:
        match = pattern.match(line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()

            if current_section["content"]:
                sections.append(current_section)

            header_stack = header_stack[: level - 1]
            header_stack.append(title)

            current_section = {
                "headers": header_stack.copy(),
                "content": "",
            }
        else:
            current_section["content"] += line + "\n"

    if current_section["content"]:
        sections.append(current_section)

    return sections


def chunk_markdown(md_text: str, chunk_size: int = 128) -> list[str]:
    parsed_sections = parse_markdown_sections(md_text)
    chunks = []

    for section in parsed_sections:
        tokens = tokenizer.encode(section["content"])
        token_chunks = [
            tokens[i:i + chunk_size]
            for i in range(0, len(tokens), chunk_size)
            if tokens[i:i + chunk_size]
        ]

        for token_chunk in token_chunks:
            chunk_text = tokenizer.decode(token_chunk)
            chunks.append(chunk_text)

    return chunks
