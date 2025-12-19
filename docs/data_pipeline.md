# Data Pipeline: DLATKDataGetter and Utilities

## Overview

The data pipeline utilities in LongitudeML are designed to extract, process, and prepare longitudinal data from relational databases (e.g., MySQL) for forecasting tasks. The main class is `DLATKDataGetter`, which interfaces with feature and outcome tables to produce datasets suitable for modeling.

## DLATKDataGetter: Method Reference

### `get_feature_tables(feature_tables=None, where='')`
**What it does:**
Retrieves features for each query/message from one or more feature tables, optionally filtered by a SQL `where` clause.

**Inputs:**
- `feature_tables` (list or str, optional): List of feature table names to use. If not provided, uses the instance’s `feature_tables`.
- `where` (str, optional): SQL WHERE clause to filter rows.

**Output:**
- Tuple:
  - `features_dict`: `{query_id: [emb1, emb2, ...], ...}` (features for each query/message)
  - `features_names`: `[[feat1, feat2, ...]_table1, [feat1, feat2, ...]_table2, ...]` (list of feature names per table)

---

### `get_long_features(where='', qry_seq_time_id_table=None)`
**What it does:**
Builds a longitudinal (sequence/time) representation of features, grouped by sequence and ordered by time.

**Inputs:**
- `where` (str, optional): SQL WHERE clause to filter messages.
- `qry_seq_time_id_table` (str, optional): Table to use for mapping query IDs to sequence and time IDs. Defaults to `msg_table`.

**Output:**
- Dict with keys:
  - `'seq_id'`: `[sid1, sid2, ...]`
  - `'time_ids'`: `[[t1, t2, ...]_sid1, [t1, t2, ...]_sid2, ...]`
  - `'embeddings'`: `[[[emb1, emb2, ...]_t1, ...]_sid1, ...]`
  - `'embeddings_names'`: `[[feat1, feat2, ...]_table1, ...]`

---

### `get_outcomes(outcome_fields=None, correl_field=None, where='')`
**What it does:**
Retrieves outcome values for each sequence or query, optionally filtered.

**Inputs:**
- `outcome_fields` (list or str, optional): Outcome field(s) to retrieve. Defaults to instance’s `outcome_fields`.
- `correl_field` (str, optional): Field to use for correlation (e.g., sequence or query ID).
- `where` (str, optional): SQL WHERE clause.

**Output:**
- Dict: `{seq_id: [outcome1, outcome2, ...], ...}`

---

### `get_long_outcomes(outcome_fields=None, where='')`
**What it does:**
Builds a longitudinal (sequence/time) representation of outcomes, grouped by sequence and ordered by time.

**Inputs:**
- `outcome_fields` (list or str, optional): Outcome field(s) to retrieve.
- `where` (str, optional): SQL WHERE clause.

**Output:**
- Dict with keys:
  - `'seq_id'`: `[sid1, sid2, ...]`
  - `'time_ids'`: `[[t1, t2, ...]_sid1, ...]`
  - `'outcomes'`: `[[[outcome1, ...]_t1, ...]_sid1, ...]`
  - `'outcomes_names'`: `[outcome_name1, ...]`

---

### `intersect_seqids(long_dict1, long_dict2)`
**What it does:**
Aligns two longitudinal dictionaries (e.g., features and outcomes) to only include common sequence IDs.

**Inputs:**
- `long_dict1`, `long_dict2`: Dicts with `'seq_id'` keys.

**Output:**
- Tuple: `(long_dict1_aligned, long_dict2_aligned)` (both only contain common sequence IDs)

---

### `combine_features_and_outcomes(outcomes_correl_field=None, features_where='', outcomes_where='')`
**What it does:**
Merges features and outcomes into a single data dictionary suitable for modeling.

**Inputs:**
- `outcomes_correl_field` (str, optional): Field to use for correlating features and outcomes.
- `features_where`, `outcomes_where` (str, optional): SQL WHERE clauses for filtering.

**Output:**
- Dict with keys:
  - `'seq_idx'`: `[seq_id1, ...]`
  - `'time_ids'`: `[[t1, t2, ...], ...]`
  - `'embeddings'`: `[[[emb1, ...], ...], ...]`
  - `'labels'`: `[[label1, ...], ...]`
  - `'query_ids'`: `[[qry_id1, ...], ...]`

---

### `train_test_split(dataset_dict, test_ratio=0.2, val_ratio=0.0, stratify=None)`
**What it does:**
Splits a dataset dictionary into train, validation, and test sets.

**Inputs:**
- `dataset_dict` (dict): Data dictionary (as output by `combine_features_and_outcomes`).
- `test_ratio` (float): Fraction for test set.
- `val_ratio` (float): Fraction for validation set.
- `stratify` (optional): Stratification labels.

**Output:**
- Dict: `{'train_data': ..., 'val_data': ..., 'test_data': ...}` (each is a data dictionary)

---

### `n_fold_split(dataset_dict, longtype_encoder, fold_column)`
**What it does:**
Splits the dataset into n folds for cross-validation, based on a fold column.

**Inputs:**
- `dataset_dict` (dict): Data dictionary.
- `longtype_encoder` (dict): Mapping of sequence/query IDs.
- `fold_column` (str): Name of the column in the outcomes table indicating fold assignment.

**Output:**
- Dict: Same as input, but with a `'folds'` key indicating fold assignment for each sequence.

---

## Example Usage
```python
from src.dlatk_datapipeline import DLATKDataGetter

# Initialize with table names and outcome fields
getter = DLATKDataGetter(
    msg_table='messages',
    feature_tables=['features_table1', 'features_table2'],
    outcome_table='outcomes',
    outcome_fields=['pcl_score']
)

# Get features and outcomes
features = getter.get_long_features()
outcomes = getter.get_long_outcomes()

# Align and merge
features, outcomes = getter.intersect_seqids(features, outcomes)
dataset = getter.combine_features_and_outcomes()
```

See also: `examples/ptsd_stop_forecasting/save_1_day_forecast_data_lang_selfreport_v6.2.py` for a full data preparation pipeline. 