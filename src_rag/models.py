import numpy as np
import re
import tiktoken
import openai
import yaml
from rank_bm25 import BM25Okapi

from FlagEmbedding import FlagModel, FlagReranker

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
    def __init__(self, chunk_size=256, overlap=0, use_headers=False, embedding_model=None, top_k=5,
                 use_reranker=False, reranker_model=None, use_hybrid=False, hybrid_alpha=0.5,
                 use_small2big=False, window_size=1):
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._use_headers = use_headers
        self._embedding_model = embedding_model or "BAAI/bge-base-en-v1.5"
        self._top_k = top_k
        self._use_reranker = use_reranker
        self._reranker_model = reranker_model or "BAAI/bge-reranker-base"
        self._use_hybrid = use_hybrid
        self._hybrid_alpha = hybrid_alpha  # 0 = full BM25, 1 = full embedding
        self._use_small2big = use_small2big  # Small2Big: etendre avec chunks adjacents
        self._window_size = window_size  # Nombre de chunks avant/apres a inclure
        self._embedder = None
        self._reranker = None
        self._bm25 = None
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
        
        # Initialiser BM25 si hybrid search active
        if self._use_hybrid:
            tokenized_chunks = [chunk.lower().split() for chunk in self._chunks]
            self._bm25 = BM25Okapi(tokenized_chunks)

    def get_corpus_embedding(self):
        return self._corpus_embedding

    def get_chunks(self):
        return self._chunks

    def embed_questions(self, questions):
        embedder = self.get_embedder()
        return embedder.encode(questions)

    def _compute_chunks(self, texts):
        return sum(
            (chunk_markdown(
                txt, 
                chunk_size=self._chunk_size, 
                overlap=self._overlap,
                use_headers=self._use_headers
            ) for txt in texts),
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
    
    def get_reranker(self):
        if not self._reranker and self._use_reranker:
            self._reranker = FlagReranker(self._reranker_model, use_fp16=True)
        return self._reranker

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
If the answer is not in the context information, reply \"I cannot answer that question\".
Query: {query}
Answer:"""

    def _get_context(self, query):
        # Recuperer plus de candidats si on utilise le reranker
        initial_k = self._top_k * 3 if self._use_reranker else self._top_k
        
        if self._use_hybrid and self._bm25:
            # Hybrid search : combiner BM25 et embeddings
            query_embedding = self.embed_questions([query])
            embedding_scores = (query_embedding @ self._corpus_embedding.T)[0]
            
            # Normaliser les scores embedding entre 0 et 1
            embedding_scores = (embedding_scores - embedding_scores.min()) / (embedding_scores.max() - embedding_scores.min() + 1e-8)
            
            # BM25 scores
            tokenized_query = query.lower().split()
            bm25_scores = np.array(self._bm25.get_scores(tokenized_query))
            
            # Normaliser les scores BM25 entre 0 et 1
            if bm25_scores.max() > 0:
                bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
            
            # Combiner les scores
            combined_scores = self._hybrid_alpha * embedding_scores + (1 - self._hybrid_alpha) * bm25_scores
            indexes = list(np.argsort(combined_scores))[-initial_k:]
        else:
            # Recherche par embeddings uniquement
            query_embedding = self.embed_questions([query])
            sim_scores = query_embedding @ self._corpus_embedding.T
            indexes = list(np.argsort(sim_scores[0]))[-initial_k:]
        
        candidates = [self._chunks[i] for i in indexes]
        
        # Re-ranking si active
        if self._use_reranker:
            reranker = self.get_reranker()
            pairs = [[query, chunk] for chunk in candidates]
            rerank_scores = reranker.compute_score(pairs)
            
            # Trier par score de reranking
            sorted_indices = np.argsort(rerank_scores)[::-1]
            candidates = [candidates[i] for i in sorted_indices[:self._top_k]]
            indexes = [indexes[i] for i in sorted_indices[:self._top_k]]
        
        # Small2Big : etendre avec les chunks adjacents
        if self._use_small2big:
            expanded_candidates = []
            seen_ranges = set()
            for idx in indexes[:self._top_k]:
                start = max(0, idx - self._window_size)
                end = min(len(self._chunks), idx + self._window_size + 1)
                range_key = (start, end)
                if range_key not in seen_ranges:
                    seen_ranges.add(range_key)
                    # Concatener les chunks adjacents
                    expanded = "\n".join(self._chunks[start:end])
                    expanded_candidates.append(expanded)
            return expanded_candidates
        
        return candidates
    


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


def chunk_markdown(md_text: str, chunk_size: int = 128, overlap: int = 0, use_headers: bool = False) -> list[str]:
    """
    Découpe un texte markdown en chunks.
    
    Args:
        md_text: Le texte markdown à découper
        chunk_size: Taille maximale de chaque chunk en tokens
        overlap: Nombre de tokens de chevauchement entre chunks consécutifs
        use_headers: Si True, préfixe chaque chunk avec la hiérarchie des headers (Small2Big)
    
    Returns:
        Liste de strings représentant les chunks
    """
    parsed_sections = parse_markdown_sections(md_text)
    chunks = []
    
    step = max(1, chunk_size - overlap)  # Eviter step <= 0

    for section in parsed_sections:
        tokens = tokenizer.encode(section["content"])
        
        # Construire le préfixe de contexte si use_headers est activé
        header_prefix = ""
        if use_headers and section["headers"]:
            header_prefix = " > ".join(section["headers"]) + "\n"
        
        for i in range(0, len(tokens), step):
            token_chunk = tokens[i:i + chunk_size]
            if token_chunk:
                chunk_text = tokenizer.decode(token_chunk)
                if use_headers and header_prefix:
                    chunks.append(header_prefix + chunk_text)
                else:
                    chunks.append(chunk_text)

    return chunks
