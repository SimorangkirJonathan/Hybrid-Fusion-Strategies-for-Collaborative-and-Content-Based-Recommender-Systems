from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


def _topk_similarity_graph(vectors: sparse.spmatrix, top_k: int = 100) -> sparse.csr_matrix:
    """Build a sparse item-item cosine graph, retaining each item's top neighbors."""
    similarity = (vectors @ vectors.T).tocsr()
    similarity.setdiag(0)
    similarity.eliminate_zeros()
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []

    for row in range(similarity.shape[0]):
        start, end = similarity.indptr[row], similarity.indptr[row + 1]
        row_cols = similarity.indices[start:end]
        row_values = similarity.data[start:end]
        if len(row_values) > top_k:
            keep = np.argpartition(row_values, -top_k)[-top_k:]
            row_cols, row_values = row_cols[keep], row_values[keep]
        rows.extend([row] * len(row_cols))
        cols.extend(row_cols.tolist())
        values.extend(row_values.tolist())

    return sparse.csr_matrix((values, (rows, cols)), shape=similarity.shape, dtype=np.float32)


def _prepare_matrices(
    train: pd.DataFrame, item_ids: Iterable[str], positive_threshold: float = 4.0
) -> tuple[list[str], list[str], dict[str, int], dict[str, int], sparse.csr_matrix, dict[str, set[str]]]:
    user_ids = train["user_id"].astype(str).drop_duplicates().tolist()
    item_ids = list(dict.fromkeys(map(str, item_ids)))
    user_index = {value: index for index, value in enumerate(user_ids)}
    item_index = {value: index for index, value in enumerate(item_ids)}

    positive = train[train["rating"] >= positive_threshold]
    # Keep row/column filtering aligned if unexpected IDs occur.
    pairs = [
        (user_index[str(user)], item_index[str(item)])
        for user, item in zip(positive["user_id"], positive["item_id"])
        if str(user) in user_index and str(item) in item_index
    ]
    rows = [row for row, _ in pairs]
    cols = [col for _, col in pairs]
    values = np.ones(len(pairs), dtype=np.float32)
    interactions = sparse.csr_matrix((values, (rows, cols)), shape=(len(user_ids), len(item_ids)))
    interactions.data[:] = 1.0

    seen: dict[str, set[str]] = {}
    for user, item in zip(train["user_id"].astype(str), train["item_id"].astype(str)):
        seen.setdefault(user, set()).add(item)
    return user_ids, item_ids, user_index, item_index, interactions, seen


def fit_popularity(train: pd.DataFrame, positive_threshold: float = 4.0) -> pd.Series:
    """Return item popularity as count of positive training interactions."""
    positive = train.loc[train["rating"] >= positive_threshold, ["item_id"]].copy()
    positive["item_id"] = positive["item_id"].astype(str)
    return positive.groupby("item_id").size().sort_values(ascending=False)


def fit_item_cf(
    train: pd.DataFrame, item_ids: Iterable[str], top_k: int = 100, positive_threshold: float = 4.0
) -> dict:
    """Fit item-based collaborative filtering with positive co-occurrence cosine."""
    user_ids, item_ids, user_index, item_index, interactions, seen = _prepare_matrices(
        train, item_ids, positive_threshold
    )
    item_user = interactions.T.tocsr()
    norms = np.sqrt(np.asarray(item_user.multiply(item_user).sum(axis=1)).ravel())
    inverse = np.zeros_like(norms, dtype=np.float32)
    inverse[norms > 0] = 1.0 / norms[norms > 0]
    normalized = sparse.diags(inverse) @ item_user
    graph = _topk_similarity_graph(normalized, top_k=top_k)
    return {
        "user_ids": user_ids, "item_ids": item_ids, "user_index": user_index,
        "item_index": item_index, "interactions": interactions, "seen": seen,
        "similarity": graph,
    }


def fit_tfidf_cbf(
    train: pd.DataFrame,
    items: pd.DataFrame,
    item_ids: Iterable[str],
    text_column: str = "content_text_basic",
    top_k: int = 100,
    positive_threshold: float = 4.0,
) -> dict:
    """Fit TF-IDF item similarity and use training positives as user histories."""
    user_ids, item_ids, user_index, item_index, interactions, seen = _prepare_matrices(
        train, item_ids, positive_threshold
    )
    metadata = items.copy()
    metadata["item_id"] = metadata["item_id"].astype(str)
    text_by_id = metadata.set_index("item_id")[text_column].fillna("").astype(str)
    documents = [text_by_id.get(item_id, "") for item_id in item_ids]
    vectorizer = TfidfVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.8,
        max_features=100_000, sublinear_tf=True, dtype=np.float32,
    )
    item_vectors = vectorizer.fit_transform(documents).tocsr()
    graph = _topk_similarity_graph(item_vectors, top_k=top_k)
    return {
        "user_ids": user_ids, "item_ids": item_ids, "user_index": user_index,
        "item_index": item_index, "interactions": interactions, "seen": seen,
        "similarity": graph, "vectorizer": vectorizer,
    }


def recommend_from_model(
    model: dict,
    users: Iterable[str],
    candidates: Iterable[str],
    popularity: pd.Series,
    top_n: int = 200,
) -> dict[str, list[tuple[str, float]]]:
    """Rank sparse item-neighbor scores; popularity fills zero-score tail slots."""
    candidate_set = set(map(str, candidates))
    popularity_rank = [(str(item), float(score)) for item, score in popularity.items() if str(item) in candidate_set]
    result: dict[str, list[tuple[str, float]]] = {}
    for user in users:
        user = str(user)
        user_row = model["user_index"].get(user)
        scores = None
        if user_row is not None:
            scores = model["interactions"].getrow(user_row) @ model["similarity"]
            scores = scores.tocsr()
        seen = model["seen"].get(user, set())
        scored: dict[str, float] = {}
        if scores is not None and scores.nnz:
            row = scores.getrow(0)
            for index, value in zip(row.indices, row.data):
                item_id = model["item_ids"][index]
                if item_id in candidate_set and item_id not in seen:
                    scored[item_id] = float(value)
        ranked = sorted(scored.items(), key=lambda pair: pair[1], reverse=True)
        if len(ranked) < top_n:
            for item_id, _ in popularity_rank:
                if item_id not in seen and item_id not in scored:
                    ranked.append((item_id, 0.0))
                    if len(ranked) >= top_n:
                        break
        result[user] = ranked[:top_n]
    return result


def recommend_popularity(
    users: Iterable[str],
    candidates: Iterable[str],
    popularity: pd.Series,
    seen: dict[str, set[str]],
    top_n: int = 200,
) -> dict[str, list[tuple[str, float]]]:
    candidate_set = set(map(str, candidates))
    ranked = [(str(item), float(score)) for item, score in popularity.items() if str(item) in candidate_set]
    result = {}
    for user in users:
        user = str(user)
        user_seen = seen.get(user, set())
        user_ranked = []
        for item, score in ranked:
            if item not in user_seen:
                user_ranked.append((item, score))
                if len(user_ranked) == top_n:
                    break
        result[user] = user_ranked
    return result
