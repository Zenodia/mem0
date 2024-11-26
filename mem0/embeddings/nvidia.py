from typing import Optional

from langchain_nvidia_ai_endpoints import  NVIDIAEmbeddings, NVIDIARerank

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase
from dotenv import load_dotenv
load_dotenv()
import os

class NVEmbedding(EmbeddingBase):
    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        self.config.model = self.config.model or "nvidia/nv-embed-v1"
        print(self.config.model)
        self.model = NVIDIAEmbeddings( model=self.config.model , truncate="NONE", )

        self.config.embedding_dims = self.config.embedding_dims or 4096

    def embed(self, text):
        """
        Get the embedding for the given text using NVIDIA Embedding Class

        Args:
            text (str): The text to embed.

        Returns:
            list: The embedding vector.
        """
        return self.model.embed_query(text)
