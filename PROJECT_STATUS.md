# Project Status

**Current phase:** Faculty demonstration readiness; five-label source rights and approvals remain unresolved.  
**Overall state:** The Streamlit application remains in fixed demonstration mode and can run without a trained checkpoint. The local raw GoEmotions part-one CSV and duplicate retry copy directly cover only Confused/`confusion` and Curious/`curiosity` among the five intended outputs; Reddit-text reuse rights remain unresolved, so neither file is approved training data. The prepared initial CSV is header-only. **No student data was collected, no model was trained, and no training is authorized.**

## Completed

- Preserved all five application labels and the demonstration-only Streamlit shell.
- Implemented and documented the dataset input contract, local validation/audit, fixed-seed split pipeline, and actual persisted formats: train.jsonl, validation.jsonl, test.jsonl, and split_report.json.
- Updated group-aware persisted splits to retain the supplied opaque `group_id` in restricted split JSONL, include it in content fingerprints, and verify cross-split group disjointness and per-split group counts on reload. IDs remain excluded from model features, reports, and checkpoint metadata; their participant/source meaning is not verified by code.
- Added version 2 split manifests for the group-ID format. Legacy version 1 text-hash bundles remain readable; legacy group-aware bundles without persisted IDs are rejected as unverifiable and must be regenerated from reviewed source data.
- Added a validated loader for prepared files. It checks strict schema and canonical label order, split/report counts, label distributions, manifest fingerprint, file-content fingerprints, exact normalized-text separation, group separation when group IDs were supplied, and match against an explicit reviewed authorization before constructing DatasetSplits.
- Kept parsed test-only/demo bundles as PreparedSplitBundle, which has no fit capability. TrainingAuthorization refuses test_fixture sources; the loader cannot turn them into approved DatasetSplits.
- Added reproducible training_manifest.json generation with opaque source/version and approval references, seed/ratios/counts, label distributions, ordered per-split SHA-256 fingerprints, and grouping/leakage limitations.
- Extended checkpoint metadata to record the dataset reference and split-manifest fingerprint. Checkpoint loading requires expected model type, configuration digest, dataset reference, and manifest fingerprint, while preserving label-order, stored-config digest, and weight-checksum checks.
- Documented a Python API training entry path consuming persisted validated splits. No training CLI was added. The preparation CLI does not create an approval-bearing manifest by itself.
- Added an offline BiLSTM CPU smoke command using only hard-coded synthetic test-only token IDs. It runs one forward/loss shape check if PyTorch exists; it performs no optimizer step or performance evaluation and does not download anything.
- Added regression tests for manifest stability, malformed/tampered manifests, file/report mismatch, approval mismatch and fixture refusal, checkpoint-manifest mismatch, and smoke skip behavior.
- Evaluation reports now include a per-label binary confusion matrix over known decisions. F1 is zero when there is no ground-truth positive support but false positives exist; it is undefined only when there are no actual or predicted positives. Checkpoint metadata validates the matrix schema and counts.
- Tightened manifest parsing so malformed version/source/grouping values fail as DatasetManifestError rather than leaking raw TypeError.
- Updated TRAINING_GUIDE.md with the persisted workflow, manifest/checkpoint meaning, API entry path, smoke command, and verified/unverified boundaries.
- During Phase 3G, no dependencies were changed, and no datasets, model weights, or external APIs were accessed in that phase; the current environment snapshot below is separately verified.
- Updated the demonstration interface and usage guidance to state the five intended categories and that genuine training awaits suitable, approved five-label data and human reviews.

## Verification and current environment

- Full suite: `.\.venv\Scripts\python.exe -m unittest discover -s tests -v` ran 70 tests in 14.497 seconds; all 70 passed with no skips.
- The suite executed the real offline BiLSTM CPU smoke path using synthetic test-only token IDs: one forward pass and finite loss check, with no optimizer step, dataset access, checkpoint, or performance metric. This is a framework smoke check, not model training.
- `app.py` imported successfully. A local Streamlit server started with `.\.venv\Scripts\python.exe -m streamlit run app.py --server.headless true --server.port 8765`; `/_stcore/health` returned `ok` and the app page returned HTTP 200. The server was stopped after the check. No trained checkpoint was required.
- Current environment: 64-bit Python 3.11.0; Streamlit 1.66.0; Plotly 6.9.0; PyTorch 2.11.0+cu130. `torch.cuda.is_available()` returned true and PyTorch detected the NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM; CUDA runtime in the wheel: 13.0). Transformers is not installed.
- UI and documentation remain explicit that the five scores are fixed illustrative fixtures and the analytics are demo-only. No approved-source end-to-end load-and-fit path, BERT load, real model evaluation, or real checkpoint has been verified.

## Pending

- Human approval, source/license or consent review, and availability of an appropriate five-label dataset with reviewed definitions and mapping. GoEmotions reuse rights are unresolved and its audited part-one labels do not cover Bored, Confident, or Frustrated.
- Confirm privacy/data-protection arrangements, storage/access, retention/deletion, and required institutional/faculty approval before collecting any supplementary study data.
- Confirm complete versus partial labels and known positive/negative coverage for all five categories in training, validation, and held-out test groups.
- Decide on any optional auxiliary corpus and document its source-specific licensing and limited label mappings.
- Resolve and review an exact pinned BERT checkpoint if BERT is to be used.
- Transformers and any other required optional packages are not installed; separately review dependency versions before a future model run. PyTorch is installed, but its presence is not training authorization.

## Known limitations

- Authorization and source references are caller assertions; code cannot verify that human approvals are genuine or that a caller has not mislabeled synthetic/demo content.
- The manifest uses SHA-256 consistency fingerprints, not signatures, proof of provenance, or anonymization. Guessable text may be vulnerable to dictionary attacks.
- Version 2 grouped split files retain opaque group tokens next to text under restricted local storage. Code can verify token disjointness but cannot prove the tokens identify participants or are anonymous. Legacy version 1 group-aware bundles are not loadable for training because their group disjointness is unverifiable.
- The loader and manifest writer have not been exercised against an approved real dataset. No real model fit, test evaluation, PyTorch serialization, or BERT loading has run.
- No real model fit, BERT load, held-out model evaluation, or real checkpoint artifact has been verified; mocked tests do not prove those paths work.
- No accuracy, performance, calibration, or real-user analytics claims exist. Gemini remains unused.

## Next recommended action

For the faculty demonstration, launch the app using the README command and explore Study challenge and Demonstration analytics; no checkpoint or Gemini key is needed. Before genuine training, first resolve dataset rights and human/faculty/privacy approvals, establish suitable reviewed labels for all five categories, complete the consent/storage/retention plan if collecting data, then prepare and audit the approved source, freeze and validate participant-separated splits, and only then run an authorized fit/evaluation. Keep the app in demonstration mode until a real model has been trained and evaluated.
