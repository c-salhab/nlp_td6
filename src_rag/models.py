import numpy as np
import re
import tiktoken
import openai
import yaml

from FlagEmbedding import FlagModel

CONF = yaml.safe_load(open("config.yml"))

CLIENT = openai.OpenAI(
    base_url=CONF.get("xai_base_url", "https://api.x.ai/v1"),
    api_key=CONF.get("xai_api_key", CONF.get("groq_key", "")),
)

tokenizer = tiktoken.get_encoding("cl100k_base")

def get_model(config):
    if config:
        model_config = config.get("model", {})
        model_type = model_config.get("type", "standard")

        if model_type == "small2big":
            from src_rag.small2big import Small2BigRAG
            return Small2BigRAG(
                small_chunk_size=model_config.get("small_chunk_size", 128),
                large_chunk_size=model_config.get("large_chunk_size", 512),
                overlap=model_config.get("overlap", 0),
                top_k=model_config.get("top_k", 5),
                embedding_model=model_config.get("embedding_model", 'BAAI/bge-base-en-v1.5')
            )
        else:
            return RAG(**{k: v for k, v in model_config.items() if k != "type"})
    else:
        return RAG()


class RAG:
    def __init__(self, chunk_size=256, overlap=0, top_k=5, embedding_model='BAAI/bge-base-en-v1.5'):
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._top_k = top_k
        self._embedding_model = embedding_model
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

            with open(filename) as f:
                texts.append(f.read())
                self._loaded_files.add(filename)

        
        self._texts += texts

        chunks_added = self._compute_chunks(texts)
        self._chunks += chunks_added

        new_embedding = self.embed_corpus(chunks_added)
        if self._corpus_embedding is not None:
            self._corpus_embedding = np.vstack([self._corpus_embedding, new_embedding])
        else:
            self._corpus_embedding = new_embedding

    def get_corpus_embedding(self):
        return self._corpus_embedding

    def get_chunks(self):
        return self._chunks

    def embed_questions(self, questions):
        embedder = self.get_embedder()
        return embedder.encode(questions)

    def _compute_chunks(self, texts):
        return sum(
            (chunk_markdown(txt, chunk_size=self._chunk_size, overlap=self._overlap) for txt in texts),
            [],
        )

    def embed_corpus(self, chunks):
        embedder = self.get_embedder()
        return embedder.encode(chunks)

    def get_embedder(self):
        if not self._embedder:
            self._embedder = FlagModel(
                self._embedding_model,
                query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                use_fp16=True,
            )

        return self._embedder

    def reply(self, query):
        prompt = self._build_prompt(query)
        res = self._client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=CONF.get("xai_model", CONF.get("groq_model", "grok-4-latest")),
        )
        return res.choices[0].message.content
        

    def _build_prompt(self, query):
        context_str = "\n".join(self._get_context(query))

        return f"""Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
If the answer is not in the context information, reply \"I cannot answer that question\".
Query: {query}
Answer:"""

    def _get_context(self, query):
        query_embedding = self.embed_questions([query])
        sim_scores = query_embedding @ self._corpus_embedding.T
        indexes = list(np.argsort(sim_scores[0]))[-self._top_k:]
        return [self._chunks[i] for i in indexes]
    


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


def parse_markdown_sections(md_text: str) -> list[dict[str, str]]:
    """
    Parses markdown into a list of {'headers': [...], 'content': ...}
    Preserves full header hierarchy (e.g. ["Section", "Sub", "SubSub", ...])
    """
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

            # Save previous section
            if current_section["content"]:
                sections.append(current_section)

            # Adjust the header stack
            header_stack = header_stack[:level - 1]
            header_stack.append(title)

            current_section = {
                "headers": header_stack.copy(),
                "content": ""
            }
        else:
            current_section["content"] += line + "\n"

    if current_section["content"]:
        sections.append(current_section)

    return sections


def chunk_markdown(md_text: str, chunk_size: int = 128, overlap: int = 0) -> list[dict]:
    parsed_sections = parse_markdown_sections(md_text)
    chunks = []

    for section in parsed_sections:
        tokens = tokenizer.encode(section["content"])

        # Calculate step size based on overlap
        step_size = chunk_size - overlap if overlap > 0 else chunk_size

        # Create overlapping chunks
        token_chunks = []
        for i in range(0, len(tokens), step_size):
            chunk = tokens[i:i + chunk_size]
            if chunk:  # Only add non-empty chunks
                token_chunks.append(chunk)
            # Break if we've reached the end and added the last chunk
            if i + chunk_size >= len(tokens):
                break

        for token_chunk in token_chunks:
            chunk_text = tokenizer.decode(token_chunk)
            chunks.append(chunk_text)

    return chunks
