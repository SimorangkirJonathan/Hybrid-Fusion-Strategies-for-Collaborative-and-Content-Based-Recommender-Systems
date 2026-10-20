from __future__ import annotations

from collections.abc import Iterable


def weighted_score_fusion(
    cf_rankings: dict[str, list[tuple[str, float]]],
    cbf_rankings: dict[str, list[tuple[str, float]]],
    users: Iterable[str],
    cf_weight: float = 0.5,
    top_n: int = 200,
) -> dict[str, list[tuple[str, float]]]:
    """Fuse per-user top lists after min-max normalization of each component."""
    results = {}
    for user in map(str, users):
        cf = cf_rankings.get(user, [])
        cbf = cbf_rankings.get(user, [])
        cf_values = [score for _, score in cf]
        cbf_values = [score for _, score in cbf]
        # Handle each list separately so tied lists remain neutral within their own scale.
        cf_low, cf_high = (min(cf_values), max(cf_values)) if cf_values else (0.0, 0.0)
        cbf_low, cbf_high = (min(cbf_values), max(cbf_values)) if cbf_values else (0.0, 0.0)
        cf_norm = {
            item: (1.0 if cf_high == cf_low else (score - cf_low) / (cf_high - cf_low))
            for item, score in cf
        }
        cbf_norm = {
            item: (1.0 if cbf_high == cbf_low else (score - cbf_low) / (cbf_high - cbf_low))
            for item, score in cbf
        }
        item_ids = set(cf_norm) | set(cbf_norm)
        combined = {
            item: cf_weight * cf_norm.get(item, 0.0) + (1.0 - cf_weight) * cbf_norm.get(item, 0.0)
            for item in item_ids
        }
        results[user] = sorted(combined.items(), key=lambda pair: pair[1], reverse=True)[:top_n]
    return results


def reciprocal_rank_fusion(
    cf_rankings: dict[str, list[tuple[str, float]]],
    cbf_rankings: dict[str, list[tuple[str, float]]],
    users: Iterable[str],
    rrf_k: int = 60,
    top_n: int = 200,
) -> dict[str, list[tuple[str, float]]]:
    """Fuse rankings using standard reciprocal rank fusion."""
    results = {}
    for user in map(str, users):
        scores: dict[str, float] = {}
        for rankings in (cf_rankings, cbf_rankings):
            for rank, (item, _) in enumerate(rankings.get(user, []), start=1):
                scores[item] = scores.get(item, 0.0) + 1.0 / (rrf_k + rank)
        results[user] = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:top_n]
    return results
