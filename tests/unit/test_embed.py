from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, ClassVar

import numpy as np
import pytest
from fastembed.sparse.sparse_embedding_base import SparseEmbedding
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from fakes import FakeEmbedder
from movie_rag.config import Settings
from movie_rag.ingest import embed as embed_module
from movie_rag.ingest.embed import Embedder, FastEmbedder, to_sparse_vector

MakeSettings = Callable[..., Settings]


def make_tokenizer() -> Tokenizer:
    """A tiny BERT-shaped tokenizer: wraps every text in [CLS] ... [SEP], like the real WordPiece model does."""
    vocab = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "hello": 3, "world": 4, "movie": 5, "plot": 6}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    return tokenizer


class FakeOnnxModel:
    def __init__(self) -> None:
        self.tokenizer = make_tokenizer()

    def tokenize(self, documents: list[str]) -> list[Any]:
        return self.tokenizer.encode_batch(documents)


class FakeTextEmbedding:
    instances: ClassVar[list[FakeTextEmbedding]] = []

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = FakeOnnxModel()
        self.seen: list[tuple[str, list[str] | str]] = []
        FakeTextEmbedding.instances.append(self)

    def embed(self, documents: list[str]) -> Iterable[np.ndarray[Any, Any]]:
        self.seen.append(("embed", documents))
        return (np.array([float(len(d)), 1.0]) for d in documents)

    def query_embed(self, query: str) -> Iterable[np.ndarray[Any, Any]]:
        self.seen.append(("query_embed", query))
        return iter([np.array([9.0, 9.0])])


class FakeSparseEmbedding:
    instances: ClassVar[list[FakeSparseEmbedding]] = []

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.seen: list[tuple[str, list[str] | str]] = []
        FakeSparseEmbedding.instances.append(self)

    def embed(self, documents: list[str]) -> Iterable[SparseEmbedding]:
        self.seen.append(("embed", documents))
        return (SparseEmbedding(indices=np.array([1, 2]), values=np.array([float(len(d)), 1.0])) for d in documents)

    def query_embed(self, query: str) -> Iterable[SparseEmbedding]:
        self.seen.append(("query_embed", query))
        return iter([SparseEmbedding(indices=np.array([7]), values=np.array([1.0]))])


@pytest.fixture
def embedder(monkeypatch: pytest.MonkeyPatch, make_settings: MakeSettings) -> FastEmbedder:
    FakeTextEmbedding.instances.clear()
    FakeSparseEmbedding.instances.clear()
    monkeypatch.setattr(embed_module, "TextEmbedding", FakeTextEmbedding)
    monkeypatch.setattr(embed_module, "SparseTextEmbedding", FakeSparseEmbedding)
    return FastEmbedder(make_settings().embeddings)


def test_models_are_loaded_lazily_with_the_configured_names(embedder: FastEmbedder) -> None:
    assert FakeTextEmbedding.instances == [] and FakeSparseEmbedding.instances == []
    embedder.embed_dense(["a"])
    assert [i.model_name for i in FakeTextEmbedding.instances] == ["BAAI/bge-small-en-v1.5"]
    assert FakeSparseEmbedding.instances == []
    embedder.embed_sparse(["a"])
    assert [i.model_name for i in FakeSparseEmbedding.instances] == ["Qdrant/bm25"]


def test_models_are_loaded_once(embedder: FastEmbedder) -> None:
    for _ in range(3):
        embedder.embed_dense(["a"])
        embedder.embed_sparse(["a"])
        embedder.count_tokens("hello")
    assert len(FakeTextEmbedding.instances) == 1 and len(FakeSparseEmbedding.instances) == 1


def test_count_tokens_excludes_the_special_tokens(embedder: FastEmbedder) -> None:
    assert embedder.count_tokens("hello world") == 2  # [CLS] hello world [SEP] -> 2
    assert embedder.count_tokens("hello world movie plot unknownword") == 5
    assert embedder.count_tokens("") == 0


def test_documents_use_embed_and_queries_use_query_embed(embedder: FastEmbedder) -> None:
    assert embedder.embed_dense(["ab", "abcd"]) == [[2.0, 1.0], [4.0, 1.0]]
    sparse = embedder.embed_sparse(["abc"])
    assert (sparse[0].indices, sparse[0].values) == ([1, 2], [3.0, 1.0])
    assert embedder.embed_dense_query("q") == [9.0, 9.0]
    query = embedder.embed_sparse_query("q")
    assert (query.indices, query.values) == ([7], [1.0])
    assert [kind for kind, _ in FakeTextEmbedding.instances[0].seen] == ["embed", "query_embed"]
    assert [kind for kind, _ in FakeSparseEmbedding.instances[0].seen] == ["embed", "query_embed"]


def test_to_sparse_vector_gives_plain_python_numbers() -> None:
    vector = to_sparse_vector(SparseEmbedding(indices=np.array([3, 5]), values=np.array([0.5, 1.5])))
    assert vector.indices == [3, 5] and vector.values == [0.5, 1.5]
    assert all(type(i) is int for i in vector.indices) and all(type(v) is float for v in vector.values)


def test_fast_embedder_and_the_fake_both_satisfy_the_embedder_protocol(embedder: FastEmbedder) -> None:
    for candidate in (embedder, FakeEmbedder()):
        typed: Embedder = candidate
        assert typed.count_tokens("one two three") >= 1


def test_fake_embedder_is_deterministic_and_normalised() -> None:
    fake = FakeEmbedder()
    first, second = fake.embed_dense(["a b c", "a b c"])
    assert first == second
    assert abs(sum(v * v for v in first) - 1.0) < 1e-9
    repeated = fake.embed_sparse(["x y x"])[0]
    assert sorted(repeated.values) == [1.0, 2.0] and len(repeated.indices) == 2
