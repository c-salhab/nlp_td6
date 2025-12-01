import numpy as np
import re
import tiktoken
import openai
import yaml
from sentence_transformers import SentenceTransformer

CONF = yaml.safe_load(open("config.yml"))

CLIENT = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=CONF["groq_key"],
)

tokenizer = tiktoken.get_encoding("cl100k_base")

def get_model(config):
    if config:
        return RAG(**config.get("model", {}))
    else:
        return RAG()

class RAG:
    def __init__(
        self,
        chunk_size=512,
        chunk_overlap=50,
        embedding_model="all-MiniLM-L6-v2",
        llm_model="llama-3.1-8b-instant",
        top_k=5,
    ):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._embedding_model = embedding_model
        self._llm_model = llm_model
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

            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        texts.append(content)
                        self._loaded_files.add(filename)
            except Exception as e:
                print(f"Error reading {filename}: {e}")

        if not texts:
            return

        self._texts += texts
        chunks_added = self._compute_chunks(texts)
        
        if not chunks_added:
            return
            
        self._chunks += chunks_added
        print(f"Number of chunks added: {len(chunks_added)}")

        valid_chunks = [chunk for chunk in chunks_added if self._is_valid_chunk(chunk)]
        if valid_chunks:
            new_embedding = self.embed_corpus(valid_chunks)
            if new_embedding is not None:
                if self._corpus_embedding is not None:
                    self._corpus_embedding = np.vstack([self._corpus_embedding, new_embedding])
                else:
                    self._corpus_embedding = new_embedding
                print(f"Embeddings calculated for {len(valid_chunks)} chunks")

    def _is_valid_chunk(self, chunk):
        if not chunk or not chunk.strip():
            return False
        clean_chunk = chunk.strip()
        return len(clean_chunk) >= 20 and len(clean_chunk.split()) >= 3

    def get_corpus_embedding(self):
        return self._corpus_embedding

    def get_chunks(self):
        return self._chunks

    def embed_questions(self, questions):
        if isinstance(questions, str):
            questions = [questions]
        
        try:
            embedder = self.get_embedder()
            return embedder.encode(questions)
        except Exception as e:
            print(f"Error embedding questions: {e}")
            return np.random.randn(len(questions), 384)

    def _compute_chunks(self, texts):
        all_chunks = []
        for txt in texts:
            if not txt.strip():
                continue
            chunks = chunk_markdown(txt, self._chunk_size, self._chunk_overlap)
            valid_chunks = [chunk for chunk in chunks if self._is_valid_chunk(chunk)]
            all_chunks.extend(valid_chunks)
        return all_chunks

    def embed_corpus(self, chunks):
        if not chunks:
            return None
            
        try:
            embedder = self.get_embedder()
            valid_chunks = [chunk for chunk in chunks if self._is_valid_chunk(chunk)]
            
            if not valid_chunks:
                return None
                
            embeddings = embedder.encode(valid_chunks)
            return embeddings
                
        except Exception as e:
            print(f"Error embedding corpus: {e}")
            return None

    def get_embedder(self):
        if not self._embedder:
            self._embedder = SentenceTransformer(self._embedding_model)
        return self._embedder

    def reply(self, query):
        try:
            prompt = self._build_prompt(query)
            res = self._client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
            )
            return res.choices[0].message.content
        except Exception as e:
            print(f"Error generating reply: {e}")
            return "I cannot answer that question"

    def _build_prompt(self, query):
        context_chunks = self._get_context(query)
        context_str = "\n".join(context_chunks) if context_chunks else "No relevant context found."

        return f"""Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
If the answer is not in the context information, reply "I cannot answer that question".
Query: {query}
Answer:"""

    def _get_context(self, query):
        if (self._corpus_embedding is None or len(self._corpus_embedding) == 0 or len(self._chunks) == 0):
            return self._chunks[:3] if self._chunks else []
        
        try:
            query_embedding = self.embed_questions([query])
            sim_scores = query_embedding @ self._corpus_embedding.T
            top_k = min(self._top_k, len(self._chunks))
            indexes = list(np.argsort(sim_scores[0]))[-top_k:]
            return [self._chunks[i] for i in indexes]
        except Exception as e:
            print(f"Error in context retrieval: {e}")
            return self._chunks[:self._top_k] if self._chunks else []

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

            if current_section["content"].strip():
                sections.append(current_section)

            header_stack = header_stack[:level - 1]
            header_stack.append(title)

            current_section = {
                "headers": header_stack.copy(),
                "content": ""
            }
        else:
            current_section["content"] += line + "\n"

    if current_section["content"].strip():
        sections.append(current_section)

    return sections

def chunk_markdown(md_text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> list[str]:
    if not md_text.strip():
        return []
        
    parsed_sections = parse_markdown_sections(md_text)
    chunks = []

    for section in parsed_sections:
        content = section["content"].strip()
        if not content:
            continue
            
        tokens = tokenizer.encode(content)
        
        if len(tokens) <= chunk_size:
            chunk_text = tokenizer.decode(tokens)
            if section["headers"]:
                headers_str = " > ".join(section["headers"])
                chunk_text = f"Section: {headers_str}\n{chunk_text}"
            chunks.append(chunk_text)
            continue
            
        for i in range(0, len(tokens), chunk_size - chunk_overlap):
            end_idx = min(i + chunk_size, len(tokens))
            chunk_tokens = tokens[i:end_idx]
            
            if chunk_tokens:
                chunk_text = tokenizer.decode(chunk_tokens)
                if section["headers"]:
                    headers_str = " > ".join(section["headers"])
                    chunk_text = f"Section: {headers_str}\n{chunk_text}"
                chunks.append(chunk_text)
                
            if end_idx >= len(tokens):
                break

    return chunks