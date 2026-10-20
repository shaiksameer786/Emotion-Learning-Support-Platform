# Training Infrastructure Guide

## Scope and current state

This guide describes infrastructure for future five-label multi-label models. No model has been trained, no trained checkpoint exists, and no real performance results exist. A raw GoEmotions part-one CSV and a duplicate retry copy are present locally, but their Reddit-text reuse rights are unresolved and they are not approved training data; they directly cover only Confused/`confusion` and Curious/`curiosity` among the five targets. The Streamlit app remains in demonstration mode.

The training API is in els_platform.training.engine: fit_bilstm, fit_bert, and evaluate_model_on_test. Persisted prepared splits can now be validated and loaded through load_training_splits. There is no training CLI. PyTorch and Transformers imports remain lazy, so the app and non-framework tests work without the optional ML stack.

## Persisted split workflow

The preparation pipeline writes train.jsonl, validation.jsonl, test.jsonl, and split_report.json. After the source, license or consent, mapping, and data-protection reviews are complete and training is approved, write_training_manifest adds training_manifest.json. The training API loads and validates the bundle before it constructs a fit-capable DatasetSplits object.

The supported API sequence for a future separately approved run is:

1. Read the reviewed local source with load_dataset_file.
2. Create deterministic partitions with split_dataset using the reviewed seed and ratios.
3. Persist the partitions with write_split_outputs.
4. Construct a TrainingAuthorization only from actual documented human approvals, then call write_training_manifest with an opaque dataset/version reference.
5. In the training process, call load_training_splits with the same authorization. It checks the manifest, files, label order, schema, counts, fingerprints, approval reference, and exact normalized-text separation before constructing DatasetSplits.
6. Call fit_bilstm or fit_bert with the loaded splits and authorization. Each fit function repeats the training preflight checks.

There is no approved dataset in this project, so this sequence is documentation of the implemented API, not an instruction to assert approvals or begin training. The preparation CLI alone does not create an approval-bearing manifest. The fit APIs accept only reviewed_public or consent_approved_supplement source kinds. Test-only bundles can be parsed as PreparedSplitBundle for inspection, but cannot be loaded as fit-capable DatasetSplits.

The version 2 training manifest records the canonical label order, opaque source/version and approval references, split seed and requested ratios, row/group counts, label distributions, grouping limitations, and ordered per-split SHA-256 content fingerprints. When supplied group IDs were used, each restricted split JSONL also retains that opaque `group_id`; it is included in the content fingerprint, used to recheck group-disjointness after reload, and is never passed as a model feature or included in reports/checkpoint metadata. Keep these files private because they contain both text and grouping tokens. The code does not establish that a token represents a participant or that it is anonymous.

The loader rechecks file schema, label values/order, row and label counts, content hashes, manifest/report agreement, exact normalized-text separation, and—when the manifest reports supplied `group_id` values—requires those IDs in every split row and verifies that no ID crosses partitions and per-split group counts match. This proves disjointness of the supplied tokens, not their participant/source meaning; that meaning still requires human review. If no group IDs were supplied, reload can verify exact normalized-text separation only and must not claim participant separation. Version 1 text-hash bundles remain readable. Version 1 group-aware bundles are rejected because their saved files omit the IDs needed to verify the split; regenerate them in version 2 from the reviewed source.

## Data gate and target encoding

Training requires a TrainingAuthorization that affirmatively records approval for training, a reviewed source kind, license or consent review, label-mapping review, data-protection review, and opaque non-identifying dataset and approval codes. This is a caller assertion. It does not grant permission, verify a license, or replace faculty, ethics, privacy, or legal approval.

Target vectors follow the shared order Bored, Confident, Confused, Curious, Frustrated. Positive labels are multi-hot encoded. A separate known-label mask is used with elementwise binary cross-entropy so unknown labels are not treated as negative. Unknown source labels must be resolved by the preparation pipeline; the training layer accepts only canonical labels.

Before fitting, the gate requires non-empty train, validation, and test partitions; matching report row counts; at least one known positive and negative for each label in train and validation; no exact normalized text across partitions; and no supplied group ID reused across partitions. These are structural checks, not a universal data-sufficiency guarantee. Participant-level meaning of group separation depends on the source definition of group_id.

The authorization object cannot detect a caller falsely marking demonstration or synthetic data as reviewed. Keep UI fixtures and test-only examples separate and never authorize them. Automated tests use only explicitly synthetic fixtures and do not run a real fit loop.

## Models and configuration

TrainingConfig defaults to seed 42, batch size 8, five epochs, learning rate 1e-3, maximum sequence length 128, and CPU. TrainingConfig.for_bert uses a starting learning rate of 2e-5, batch size 4, and three epochs. BiLSTM dimensions and dropout are in BiLSTMModelConfig. These are resource-conscious starting values, not tuned or optimal settings.

The BiLSTM fits its hashed-token vocabulary on training text only, then uses right padding/truncation, a trainable embedding, a bidirectional LSTM, and five output logits. Vocabulary metadata stores bucket IDs, not observed words; hash collisions are possible.

BERT uses a five-logit head over an AutoModel encoder. No default checkpoint ID is selected. BertModelConfig requires a reviewed repository ID, a pinned 40-to-64-character commit hash, and explicit license, language-fit, and resource review flags. Loading is local-only; missing tokenizer or weights raise an error without a download attempt. The configured sequence length is checked against a local encoder's declared positional limit when available. Checkpoint identity and acquisition remain unresolved.

Both models use masked BCEWithLogitsLoss and AdamW. Sigmoid outputs are per-label scores and are not claimed to be calibrated probabilities.

## Evaluation sequence

1. Use one frozen, reviewed persisted split manifest for both model families.
2. Fit model parameters using the training partition and monitor validation loss. The fit loop does not read test examples.
3. Select each label threshold from validation predictions only. Threshold selection requires known positive and negative examples; missing class coverage leaves a threshold undefined and blocks finalization.
4. Finalize an evaluation plan bound to the configuration digest and validation-selected thresholds.
5. Call evaluate_model_on_test only after finalization. It checks the configuration, thresholds, authorization, in-memory split fingerprint, dataset reference, and persisted manifest fingerprint before scoring only the held-out test partition.

Reports contain aggregate macro and micro F1, per-label precision/recall/F1 and support, a per-label binary confusion matrix, subset accuracy on fully known rows, and Hamming loss over known decisions. Confusion matrices use actual labels as rows and predicted labels as columns, ordered negative then positive; unknown label decisions are excluded. If a label has no positive ground-truth support but has false-positive predictions, its F1 is zero and recall is undefined. F1 is undefined only when its denominator is zero (no actual or predicted positives). No per-example text or group IDs are written to reports.

The low-level evaluate_heldout_test and select_thresholds helpers accept arrays and a caller-provided split-name assertion; they cannot prove array provenance. Use the high-level evaluator for an actual held-out evaluation. Do not use test results to revise thresholds, training settings, or model selection.

## Checkpoints and privacy

build_checkpoint_metadata requires the shared label order, validation-selected thresholds, matching finalized configuration digest, approved data references, and the persisted split-manifest fingerprint. save_checkpoint writes a state_dict and JSON metadata with a SHA-256 weight checksum. load_checkpoint requires the expected model type, configuration digest, dataset reference, and split-manifest fingerprint; it validates all of these along with the metadata schema, label order, stored configuration digest, and weight checksum. It then loads with PyTorch weights_only=True and strict state-dict matching into a caller-built model.

These checkpoint helpers have not been exercised with real PyTorch artifacts. Tests mock serialization and loading to check application-level metadata, digest, and argument wiring only. BERT tokenizer assets are separate from the state-dict checkpoint and must be available locally at the reviewed pinned revision.

Metadata includes model/training settings, vocabulary or tokenizer identity, aggregate validation/test metrics when supplied, split counts/seed/grouping method, opaque source/approval references, and the manifest fingerprint. It omits raw text, group IDs, API keys, per-example outputs, absolute training paths, and the in-memory split fingerprint. Source artifacts and prepared text still require local access, retention, and deletion controls.

## Offline CPU smoke check

Run this from the project root:

    .\.venv\Scripts\python.exe -m els_platform.training.smoke_test --model bilstm

With PyTorch available, this performs one initialized small BiLSTM CPU forward pass and finite multi-label loss check on hard-coded synthetic test-only token IDs. It does not access files or datasets, download weights, take an optimizer step, save a checkpoint, or report performance. If PyTorch is missing, it prints SKIPPED and exits successfully. The current full suite executed this CPU-only smoke path; its result is recorded in `PROJECT_STATUS.md`.

## Dependencies and environment findings

The current `.venv` uses 64-bit Python 3.11.0 and pip 26.2.2, with Streamlit 1.66.0 and Plotly 6.9.0. PyTorch 2.11.0+cu130 is installed; `torch.cuda.is_available()` returned true for the NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM), using the wheel's CUDA 13.0 runtime. Transformers is not installed. This environment snapshot is informational; it does not establish that model training is approved or has been run. The base Streamlit application does not import the optional training stack.

requirements-models.txt keeps optional ranges torch>=2.4,<3.0 and transformers>=4.45,<5.0. The installed PyTorch version is within the declared range; Transformers remains absent. The detected CUDA-capable environment does not remove the need to use conservative batch/sequence settings for the 6 GB VRAM device. No training benchmark was run. Before any genuine training, separately approve the data and activity, then resolve and record the complete optional dependency set and exact versions.

## Verification and remaining limitations

The current full-suite and local application-launch results are recorded in `PROJECT_STATUS.md`. Synthetic test fixtures and mocks validate infrastructure boundaries only; they do not establish that a real approved-data model fit or held-out evaluation has run. No real checkpoint serialization, BERT load, or held-out model evaluation has been verified.

The successful production path from approved source through manifest creation and load is not end-to-end verified: no eligible real dataset or human approval exists, and tests deliberately do not mislabel synthetic fixtures as approved. The current CUDA-capable PyTorch installation does not establish that real model fitting or checkpoint serialization works; Transformers/BERT loading remains unavailable and untested. A manifest fingerprint detects ordinary inconsistency but is not a cryptographic signature; an actor able to change both files and manifest can recompute it. The loader cannot verify licenses, consent, source semantics, human approval truth, or participant-ID meaning.

Before real training, resolve and obtain human approval for an appropriately licensed or consent-approved five-label dataset; document mappings, label definitions, source/version, privacy approvals, retention and deletion; verify known positive and negative coverage in train, validation, and held-out test groups; review the exact split manifest; resolve/install the remaining optional dependencies if needed; and resolve a BERT checkpoint if BERT will be used. Keep the Streamlit app in demonstration mode until an approved model has been trained and evaluated.
