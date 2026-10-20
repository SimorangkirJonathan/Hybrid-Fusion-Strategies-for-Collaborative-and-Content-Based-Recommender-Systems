# 12S4054 Amazon Reviews 2023 Dataset

Category: Musical Instruments
Benchmark: official 5-core `timestamp_w_his` split.

## Core interaction files
- `interactions_train.csv`
- `interactions_validation.csv`
- `interactions_test.csv` (in FinalTest / Instructor package only)

Columns: `user_id,item_id,rating,timestamp`. The source `history` field is intentionally removed to keep the experimental protocol explicit and prevent accidental use of precomputed history strings.

## Metadata
The Development and Instructor packages contain the two official `raw_meta_Musical_Instruments` Parquet shards in `metadata_source/`. Run:

```bash
pip install -r requirements_metadata.txt
python prepare_items.py
```

This creates `items.csv`, `candidate_items_content.csv`, and `metadata_missing_item_ids.csv`. Core CBF text uses title, category, store/brand, features, description, and product details. Aggregate fields such as `average_rating` and `rating_number` are deliberately not used because they may encode post-cutoff information.

## Recommended evaluation
- Treat rating >= 4 as relevant for top-K ranking evaluation.
- Warm-start common protocol: train users + train items; use `interactions_*_warm.csv`.
- Primary metrics: Recall@10 and NDCG@10.
- Do not tune on the final test set.
- The FinalTest package should be released only after model/hyperparameter freeze.

## Exact split sizes
- Train: 427,957
- Validation: 41,284
- Test: 42,595
See `dataset_manifest.json` for exact users/items/timestamps/checksums.
