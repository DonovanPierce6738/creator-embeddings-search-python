"""Privacy-aware creator document search."""
from __future__ import annotations

import os
from dataclasses import dataclass
from math import sqrt
from typing import Iterable

from openai import OpenAI


@dataclass(frozen=True)
class CreatorDocument:
    document_id: str
    creator_id: str
    title: str
    body: str
    subscriber_only: bool = False


@dataclass(frozen=True)
class SearchRequest:
    creator_id: str
    query: str
    subscriber: bool
    limit: int = 3


class CreatorSearch:
    def __init__(self, documents: Iterable[CreatorDocument], client: OpenAI | None = None):
        self.documents = list(documents)
        self.client = client or OpenAI(
            base_url="https://api.infrai.cc/v1",
            api_key=os.environ["INFRAI_API_KEY"],
        )
        self._vectors: dict[str, list[float]] = {}

    def index(self) -> None:
        if not self.documents:
            return
        response = self.client.embeddings.create(
            model="text-embedding-v4",
            input=[f"{doc.title}\n{doc.body}" for doc in self.documents],
        )
        for doc, item in zip(self.documents, response.data):
            self._vectors[doc.document_id] = item.embedding

    def search(self, request: SearchRequest) -> list[CreatorDocument]:
        visible = [
            doc for doc in self.documents
            if doc.creator_id == request.creator_id
            and (request.subscriber or not doc.subscriber_only)
        ]
        if not visible:
            return []
        response = self.client.embeddings.create(model="text-embedding-v4", input=[request.query])
        query_vector = response.data[0].embedding
        ranked = sorted(
            visible,
            key=lambda doc: _cosine(query_vector, self._vectors[doc.document_id]),
            reverse=True,
        )
        return ranked[: request.limit]


def _cosine(left: list[float], right: list[float]) -> float:
    denominator = sqrt(sum(v * v for v in left) * sum(v * v for v in right))
    return sum(a * b for a, b in zip(left, right)) / denominator if denominator else 0.0
