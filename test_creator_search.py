from creator_search import CreatorDocument, CreatorSearch, SearchRequest


class FakeEmbeddings:
    def __init__(self):
        self.calls = []

    def create(self, *, model, input):
        self.calls.append((model, input))
        vectors = [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]]
        values = vectors[: len(input)]
        return type("Response", (), {"data": [type("Item", (), {"embedding": v}) for v in values]})()


class FakeClient:
    def __init__(self):
        self.embeddings = FakeEmbeddings()


def test_private_document_is_hidden_from_non_subscriber():
    docs = [
        CreatorDocument("private", "c1", "Subscriber vault", "members", True),
        CreatorDocument("public", "c1", "Public guide", "everyone"),
    ]
    search = CreatorSearch(docs, FakeClient())
    search.index()
    result = search.search(SearchRequest("c1", "guide", subscriber=False))
    assert [doc.document_id for doc in result] == ["public"]

