# Dataset Input Contract

This contract describes how a future, locally supplied dataset enters the preparation pipeline. It does not approve or select a source. No dataset is included, downloaded, or used for training in Phase 3B.

## Source review required first

Before supplying a file, the project owner must confirm its license and source terms permit the intended local processing and model use. For student-created material, document the consent and labeling process, retention/deletion plan, and any required institutional or guardian approval. Do not include names, student IDs, contact details, health information, grades, or other identifiable or sensitive content. The code cannot verify consent, license, de-identification, or whether a label mapping is semantically defensible.

Keep original source files local under the ignored `data/raw/` directory. Do not commit source or prepared text. The preparation command writes split files containing text under ignored `data/processed/`; restrict access to those files. Audit and split reports exclude text and group identifiers. For group-aware splits, version 2 split JSONL also retains the supplied opaque `group_id` beside each record so the loader can verify cross-split disjointness. Treat those split files as restricted because the token may still be linkable data; it is not a model feature. Reports and checkpoints continue to omit group IDs.

## Supported files and columns

Accepted encodings and formats:

- UTF-8 CSV (`.csv`), with one header row. The `labels` and optional `known_labels` cells contain JSON arrays.
- UTF-8 JSON Lines (`.jsonl` or `.ndjson`), one JSON object per non-blank line. The `labels` and optional `known_labels` values are JSON arrays.

Required columns are exactly `text` and `labels`. The only allowed optional columns are `known_labels` and `group_id`. Extra columns are rejected so identifiers and unrelated source fields cannot silently flow into prepared data. Select only permitted columns before supplying a source file.

| Column | Contract |
| --- | --- |
| `text` | A non-empty string after trimming and whitespace collapse. Unicode is normalized to NFC. Case, punctuation, accents, emoji, and negation are preserved. Non-whitespace control characters are rejected. |
| `labels` | JSON array of positive emotion category names. A row can contain more than one label when the source genuinely supports multi-label annotations. An empty array means no positive labels among the categories explicitly known for that row; by itself it does not assert that every category is negative. |
| `known_labels` (optional) | JSON array of categories that were actually assessed for that example, including categories known to be negative. Positive `labels` must be a subset. If absent, only listed positives are known by default; absent labels remain unknown. |
| `group_id` (optional) | A stable, opaque, non-identifying grouping token for a participant, conversation, or other source unit that must stay in one split. If supplied, every row needs a value. Never use a name, student number, or reversible identifier. The validator cannot determine whether a token is truly anonymized or identifies the right grouping unit. |

The only target labels are `Bored`, `Confident`, `Confused`, `Curious`, and `Frustrated`. Matching is case-insensitive after trimming. No default source-label mappings are applied. Source labels outside the target vocabulary are rejected and counted. A reviewed mapping can be supplied separately as a JSON object from source labels to one of these five target labels; mapping to any other destination is rejected. The mapping must be justified from the source annotation definitions and documented by the project owner. The pipeline checks syntax and destination membership, not semantic equivalence. Never map a source class to an unsupported target merely to fill all five categories.

If the source has complete five-label judgments, either include all five names in `known_labels` for each row or use `--assume-all-labels-known` only after verifying completeness from source documentation. Do not use that flag for a single-label or partially annotated source: it would turn unknown categories into false negatives.

The column names are fixed. If a reviewed source has different field names, create a local, documented conversion that selects and renames only the approved text, labels, and (if available) non-identifying group field before calling this pipeline. Do not retain extra identifiers in the conversion output.

## Missing values and duplicates

- Blank, null, or whitespace-only text is rejected and counted. Invalid text types or disallowed control characters are separately counted.
- Missing or blank `labels` is rejected. `[]` is a valid explicit empty positive set only when at least one label is known for the row.
- If a `known_labels` column exists, each row must provide a valid array. Missing `known_labels` is not silently treated as complete annotation.
- If a `group_id` column exists, blank or invalid values are rejected rather than split as if no grouping information were present.
- Exact duplicate text after conservative normalization is deduplicated only when its positive labels, known-label mask, and group ID all match. Conflicting duplicates are all withheld from output and reported for review; the pipeline does not merge annotations or choose one arbitrarily.

Rows with unsupported labels, missing required values, invalid arrays, or schema errors are not split. The validation report includes input/accepted/invalid counts, missing-value counts, duplicate counts, unsupported-label occurrence and distinct-value counts, and per-target positive/known label counts. It contains row numbers and issue codes only; it does not echo unsupported label values, text, or group IDs, so a malformed label field cannot leak a phrase or identifier into the report.

## Reproducible splits and limits

The default split uses seed `42` and approximate ratios of 70% train, 15% validation, and 15% test. It greedily balances rows and observed label counts while keeping each grouping unit together. Actual per-split counts are written to `split_report.json`; uneven group sizes can make actual ratios differ from requested ratios. At least three independent grouping units are required.

When `group_id` is present, all rows with a matching supplied value stay in one split. The code does not verify that the values represent participants or source conversations. Without `group_id`, grouping uses a hash of normalized text to avoid exact-text leakage only. Participant-level leakage cannot be assessed or prevented without suitable non-identifying group identifiers. The same seed and same normalized source records produce the same assignments; this is a group-aware approximate stratification utility, not a statistical guarantee of balanced splits.

## Run the local pipeline

From the project root, after the source has passed the required review:

```powershell
.\.venv\Scripts\python.exe -m els_platform.training.dataset data/raw/reviewed_source.csv
```

The command writes a text-free audit to `data/processed/dataset_audit.json`. If validation passes and at least three split groups are available, it writes `train.jsonl`, `validation.jsonl`, `test.jsonl`, and `split_report.json` under `data/processed/splits/`. It exits without creating splits if validation fails. Use `--seed 42` to state the seed explicitly, `--output-dir` or `--audit-output` to select other local paths, and `--label-mapping-file` for a separately reviewed JSON mapping. Version 2 persisted splits retain `group_id` only when the split used supplied group IDs; the loader checks those IDs for disjointness across partitions. Legacy version 1 text-hash bundles remain readable, but legacy group-aware bundles are rejected because their files do not retain group IDs and participant separation cannot be verified after reload.

Use `--assume-all-labels-known` only when every category was explicitly judged, including negative judgments, for every row. The outputs are prepared data only; this phase does not train, evaluate, or claim any model results. No target dataset or BERT checkpoint has been selected by this contract.
