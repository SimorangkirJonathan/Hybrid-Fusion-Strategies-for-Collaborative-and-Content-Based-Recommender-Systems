"""Alias kompatibilitas untuk nama modul lama.

Kode lama (notebook versi Task 6) mengimpor ``src.task6_initial_analysis``.
Modul ini hanya meneruskan ke ``src.task4_initial_analysis`` agar sel lama
tetap berjalan tanpa error. Hapus file ini setelah semua notebook memakai
nama baru.
"""

from .task4_initial_analysis import (  # noqa: F401
    METRICS,
    OUTPUT_DIR,
    audit_artifacts,
    build_insights,
    detect_anomalies,
    export_analysis,
    identify_extremes,
    load_artifacts,
    pairwise_wins,
    run,
    summarize_models,
    user_overlap,
)

__all__ = [
    "METRICS",
    "OUTPUT_DIR",
    "load_artifacts",
    "audit_artifacts",
    "summarize_models",
    "identify_extremes",
    "user_overlap",
    "pairwise_wins",
    "detect_anomalies",
    "build_insights",
    "export_analysis",
    "run",
]
