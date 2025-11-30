import numpy as np
import re
import tiktoken
import openai
import yaml
import os 
from dotenv import load_dotenv


from FlagEmbedding import FlagModel

CONF = yaml.safe_load(open("config.yml"))

load_dotenv()
CLIENT = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),  
)

tokenizer = tiktoken.get_encoding("cl100k_base")

def get_model(config):
    if config:
        return RAG(**config.get("model", {}))
    else:
        return RAG()


class RAG:
    def __init__(self, chunk_size=256,overlap=0,small2big=False,embedding_model='BAAI/bge-base-en-v1.5', context_size=None):
        self._chunk_size = chunk_size
        self._embedding_model = embedding_model
        self._overlap = overlap
        self._small2big = small2big
        self._context_size = context_size or chunk_size * 2  
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

            with open(filename, encoding="utf-8") as f:  
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
        """return sum(
            (chunk_markdown(txt, chunk_size=self._chunk_size) for txt in texts),
            [],
        )"""
        
        return sum(
        (chunk_markdown(txt, chunk_size=self._chunk_size, overlap=self._overlap)  
         for txt in texts),
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

    """def _get_context(self, query):
        query_embedding = self.embed_questions([query])
        sim_scores = query_embedding @ self._corpus_embedding.T
        indexes = list(np.argsort(sim_scores[0]))[-5:]
        return [self._chunks[i] for i in indexes]"""
    
    def _get_context(self, query, top_k=5):
        query_embedding = self.embed_questions([query])
        sim_scores = query_embedding @ self._corpus_embedding.T
        indexes = list(np.argsort(sim_scores[0]))[-top_k:]
        
        if not self._small2big:
            # Mode normal
            return [self._chunks[i] for i in indexes]
        else:
            # Mode Small2Big :
            expanded_chunks = []
            for idx in indexes:
                # Récupérer le texte original et extraire un contexte plus large
                # Méthode 1 : combiner avec les chunks adjacents
                start_idx = max(0, idx - 1)
                end_idx = min(len(self._chunks), idx + 2)
                expanded_text = " ".join(self._chunks[start_idx:end_idx])
                expanded_chunks.append(expanded_text)
            
            return expanded_chunks
    


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
        
        # Chunking avec overlap
        if overlap > 0:
            step = chunk_size - overlap
            if step <= 0:
                step = chunk_size  # Fallback si overlap >= chunk_size
            token_chunks = [
                tokens[i:i + chunk_size] 
                for i in range(0, len(tokens), step) 
                if tokens[i:i + chunk_size]
            ]
        else:
            token_chunks = [
                tokens[i:i + chunk_size] 
                for i in range(0, len(tokens), chunk_size) 
                if tokens[i:i + chunk_size]
            ]

        for token_chunk in token_chunks:
            chunk_text = tokenizer.decode(token_chunk)
            chunks.append(chunk_text)

    return chunks