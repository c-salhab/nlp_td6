"""
Small2Big Chunking Strategy Implementation

This strategy creates small chunks for retrieval but returns larger parent chunks
for better context in the LLM response.
"""
import numpy as np
from src_rag.models import RAG, tokenizer, parse_markdown_sections, CONF, CLIENT


class Small2BigRAG(RAG):
    """
    RAG with Small2Big chunking strategy.

    - Small chunks are used for embedding and retrieval (more precise matching)
    - Large parent chunks are returned for context (better for LLM understanding)
    """

    def __init__(self, small_chunk_size=128, large_chunk_size=512, overlap=0):
        super().__init__(chunk_size=small_chunk_size, overlap=overlap)
        self._small_chunk_size = small_chunk_size
        self._large_chunk_size = large_chunk_size
        self._large_chunks = []  # Store the larger parent chunks
        self._small_to_large_map = {}  # Map small chunk index to large chunk index

    def _compute_chunks(self, texts):
        """Override to create both small and large chunks"""
        small_chunks = []
        large_chunks = []
        small_to_large_map = {}

        current_small_idx = len(self._chunks)
        current_large_idx = len(self._large_chunks)

        for text in texts:
            sections = parse_markdown_sections(text)

            for section in sections:
                tokens = tokenizer.encode(section["content"])

                # Create large chunks first
                step_size_large = self._large_chunk_size - self._overlap if self._overlap > 0 else self._large_chunk_size

                for i in range(0, len(tokens), step_size_large):
                    large_chunk_tokens = tokens[i:i + self._large_chunk_size]
                    if not large_chunk_tokens:
                        continue

                    large_chunk_text = tokenizer.decode(large_chunk_tokens)
                    large_chunks.append(large_chunk_text)

                    # Now create small chunks within this large chunk
                    step_size_small = self._small_chunk_size - self._overlap if self._overlap > 0 else self._small_chunk_size

                    for j in range(0, len(large_chunk_tokens), step_size_small):
                        small_chunk_tokens = large_chunk_tokens[j:j + self._small_chunk_size]
                        if not small_chunk_tokens:
                            continue

                        small_chunk_text = tokenizer.decode(small_chunk_tokens)
                        small_chunks.append(small_chunk_text)

                        # Map this small chunk to its parent large chunk
                        small_to_large_map[current_small_idx] = current_large_idx
                        current_small_idx += 1

                    current_large_idx += 1

                    # Break if we've processed all tokens
                    if i + self._large_chunk_size >= len(tokens):
                        break

        # Update the mappings
        self._small_to_large_map.update(small_to_large_map)
        self._large_chunks.extend(large_chunks)

        return small_chunks

    def _get_context(self, query):
        """Override to return large chunks instead of small chunks"""
        query_embedding = self.embed_questions([query])
        sim_scores = query_embedding @ self._corpus_embedding.T

        # Get top 5 small chunk indices
        small_chunk_indices = list(np.argsort(sim_scores[0]))[-5:]

        # Map to large chunks and deduplicate
        large_chunk_indices = set()
        for small_idx in small_chunk_indices:
            if small_idx in self._small_to_large_map:
                large_chunk_indices.add(self._small_to_large_map[small_idx])

        # Return the corresponding large chunks
        return [self._large_chunks[i] for i in sorted(large_chunk_indices)]

    def get_large_chunks(self):
        """Return the large parent chunks"""
        return self._large_chunks
