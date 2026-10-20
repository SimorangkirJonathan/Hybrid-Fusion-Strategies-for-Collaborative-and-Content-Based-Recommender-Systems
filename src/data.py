from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class ProjectData:
    train: pd.DataFrame
    items: pd.DataFrame
    candidates: list[str]
    validation_warm: pd.DataFrame
    evaluable_users: list[str]


def _find_folder(root: Path, prefix: str) -> Path:
    matches = [path for path in root.iterdir() if path.is_dir() and path.name.startswith(prefix)]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one folder starting with {prefix!r} under {root}, found {len(matches)}."
        )
    return matches[0]


def load_project_data(root: str | Path = ".") -> ProjectData:
    """Load training data and the selected warm-start validation protocol."""
    root = Path(root).resolve()
    main_dir = _find_folder(root, "MAIN DATA-") / "MAIN DATA"
    supporting_dir = _find_folder(root, "SUPPORTING_DATA-") / "SUPPORTING_DATA"

    train = pd.read_csv(main_dir / "interactions_train.csv", dtype={"user_id": "string", "item_id": "string"})
    items = pd.read_csv(main_dir / "items.csv", dtype={"item_id": "string"})
    candidates = pd.read_csv(main_dir / "candidate_items_warm.csv", dtype={"item_id": "string"})["item_id"].dropna().astype(str).tolist()
    validation = pd.read_csv(
        supporting_dir / "interactions_validation_warm.csv",
        dtype={"user_id": "string", "item_id": "string"},
    )
    evaluable_users = pd.read_csv(
        supporting_dir / "evaluable_users_validation_warm.csv", dtype={"user_id": "string"}
    )["user_id"].dropna().astype(str).tolist()

    # Keep only the explicitly provided evaluation users and relevant targets.
    validation = validation[validation["user_id"].isin(evaluable_users)].copy()
    return ProjectData(
        train=train,
        items=items,
        candidates=candidates,
        validation_warm=validation,
        evaluable_users=evaluable_users,
    )
