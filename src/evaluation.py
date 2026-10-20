from __future__ import annotations

from collections.abc import Iterable
import math

import pandas as pd


def build_relevant_items(
    validation: pd.DataFrame, users: Iterable[str], positive_threshold: float = 4.0
) -> dict[str, set[str]]:
    user_set = set(map(str, users))
    relevant = validation.loc[
        (validation["rating"] >= positive_threshold)
        & validation["user_id"].astype(str).isin(user_set)
    ]
    return relevant.groupby(relevant["user_id"].astype(str))["item_id"].apply(
        lambda values: set(values.astype(str))
    ).to_dict()


def evaluate_ranking(
    recommendations: dict[str, list[tuple[str, float]]],
    relevant: dict[str, set[str]],
    users: Iterable[str],
    k: int = 10,
) -> dict[str, float]:
    recalls: list[float] = []
    ndcgs: list[float] = []
    for user in map(str, users):
        targets = relevant.get(user, set())
        if not targets:
            continue
        ranked = [item for item, _ in recommendations.get(user, [])[:k]]
        hits = [1 if item in targets else 0 for item in ranked]
        recalls.append(sum(hits) / len(targets))
        dcg = sum(hit / math.log2(position + 2) for position, hit in enumerate(hits))
        ideal_hits = min(len(targets), k)
        idcg = sum(1 / math.log2(position + 2) for position in range(ideal_hits))
        ndcgs.append(dcg / idcg if idcg else 0.0)
    return {
        "users_evaluated": len(recalls),
        f"recall@{k}": sum(recalls) / len(recalls) if recalls else 0.0,
        f"ndcg@{k}": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
    }
