from abc import ABC, abstractmethod
from typing import Type

import numpy as np

import torch
from transformers import AutoModel, AutoTokenizer
from .get_embeds import init_model_for_embeds, embed_queries, embed_qwen


class Embedder(ABC):
    """Something that produces embeddings."""

    @abstractmethod
    def embed(self, text: list[str], *, is_query: bool = False) -> list[np.ndarray]:
        pass


class NamedEmbedder(Embedder):
    """An embedder that can be constructed by name."""

    @abstractmethod
    def __init__(self):
        pass


class IBMEmbedder(NamedEmbedder):
    """ "ibm/MoLFormer-XL-both-10pct" """

    model: AutoModel
    model_str: str = "MoLFormer-XL-both-10pct"

    def __init__(self, dataset: str):
        self.dataset = dataset
        try:
            self.model = AutoModel.from_pretrained(
                "ibm/" + self.model_str, deterministic_eval=True, trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "ibm/" + self.model_str, trust_remote_code=True
            )
        except KeyError:
            raise NotImplementedError(self.model_str) from None

    def embed(
        self, text: list[str], feedback: str, is_query: bool = False
    ) -> list[np.ndarray]:
        inputs = self.tokenizer(text, padding=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return list(outputs.pooler_output.numpy())


class LlamaEmbedder(NamedEmbedder):
    """LLM2Vec based embeddings of Llama-3.1"""

    model_str: str = "llama3.1"

    def __init__(self, dataset: str):
        self.l2v = init_model_for_embeds(self.model_str)
        self.dataset = dataset

    def embed(
        self, text: list[str], feedback: str, is_query: bool = False
    ) -> list[np.ndarray]:
        embeds = embed_queries(self.l2v, text, self.dataset, feedback=feedback)
        return list(embeds)


class MistralEmbedder(NamedEmbedder):
    """LLM2Vec based embeddings of Mistral 7B"""

    model_str: str = "mistral7B"

    def __init__(self, dataset: str):
        self.l2v = init_model_for_embeds(self.model_str)
        self.dataset = dataset

    def embed(
        self, text: list[str], feedback: str, is_query: bool = False
    ) -> list[np.ndarray]:
        embeds = embed_queries(self.l2v, text, self.dataset, feedback=feedback)
        return list(embeds)


class QwenEmbedder(NamedEmbedder):
    """Embeddings of gte-qwen2-7B-Instruct"""

    model_str: str = "qwen2"

    def __init__(self, dataset: str):
        self.l2v = init_model_for_embeds(self.model_str)
        self.dataset = dataset

    def embed(
        self, text: list[str], feedback: str, is_query: bool = False
    ) -> list[np.ndarray]:
        embeds = embed_qwen(self.l2v, text, self.dataset, feedback=feedback)
        return list(embeds)


embedder_dict: dict[str, Type[NamedEmbedder]] = {
    "molformer": IBMEmbedder,
    "llama": LlamaEmbedder,
    "mistral": MistralEmbedder,
    "qwen": QwenEmbedder,
}


def embedder_from_name(name: str, dataset: str) -> NamedEmbedder:
    try:
        return embedder_dict[name](dataset)
    except KeyError as e:
        raise NotImplementedError(name) from e


class CachingEmbedder(Embedder):
    """Wrapper Class for Embedder"""

    e: Embedder

    def __init__(self, e: Embedder):
        self.e = e

    def embed(
        self, text: list[str], feedback=None, is_query: bool = False
    ) -> list[np.ndarray]:
        """Calls e.embed to embed the text."""
        embeds = self.e.embed(text, feedback, is_query=is_query)
        return embeds
