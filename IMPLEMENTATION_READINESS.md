# Implementation Readiness Review

**Project:** Emotion Detection & Learning Support Engine  
**Review date:** 2026-10-20  
**Review type:** Read-only inspection. No files were changed other than creating this report; no packages, data, or model weights were installed or downloaded, and no training was run.

## Executive finding

The repository contains a substantial but not yet real-data-validated training pipeline for five-label BiLSTM and local-checkpoint BERT experiments. It can prepare CSV/JSONL data, split in memory, write and reload approved split bundles, select thresholds from validation, evaluate held-out predictions, and save/load checkpoint metadata. However, the processed directory contains only its README; no persisted split files or training manifest are present. No model checkpoint or tokenizer artifact was found, and no trained model is connected to the app. The Streamlit app explicitly uses fixed demonstration scores.

The key readiness blocker is suitable approved data: the audited GoEmotions part-one file directly covers only Confused and Curious, is general Reddit text, and its comment-text reuse rights remain unresolved. The current persisted split representation also cannot independently establish participant disjointness after reload. Do not describe the project as trained, accurate, or ready for five-label model inference.

## 1. Existing components

| Area | Verified implementation | Current boundary |
|---|---|---|
| Data preparation | `els_platform/preprocessing.py::prepare_records` validates text, positive labels, optional `known_labels`, optional `group_id`, and duplicates. `els_platform/training/dataset.py::load_dataset_file` accepts CSV and JSONL/NDJSON; `main` prepares and writes splits. | No approved five-label training source is present. `data/raw` has GoEmotions part 1 and a retry copy; no prepared split bundle exists. |
| Train/validation/test splitting | `els_platform/training/splitting.py::split_dataset` uses seed 42 and approximate 70/15/15 ratios by default, greedily balancing row/label counts. Supplied `group_id` values are kept together; without them, normalized-text hashes prevent exact-text cross-split leakage only. | The split is in-memory until written. Participant grouping is not available when source records lack suitable group identifiers. |
| Persisted splits and loader | `els_platform/training/dataset.py::write_split_outputs` writes `train.jsonl`, `validation.jsonl`, `test.jsonl`, and `split_report.json`. `write_training_manifest` adds `training_manifest.json`. `read_prepared_split_files` validates schema, counts, label distributions, fingerprints, and exact normalized-text separation. `load_training_splits` requires a `TrainingAuthorization` before constructing fit-capable `DatasetSplits`. | The inspected `data/processed` tree has only `README.md`; there are no actual split files or manifests to reload. |
| BiLSTM | `els_platform/training/models.py::build_bilstm_model` creates a bidirectional LSTM with five logits. `els_platform/training/engine.py::fit_bilstm` fits `HashedVocabulary` via `HashedVocabulary.fit_training_split` in `els_platform/training/vocabulary.py`; vocabulary is fitted from train text. `_masked_loss` uses binary cross-entropy-with-logits and masks unknown labels. | Infrastructure exists; no real dataset fit or checkpoint is present. |
| BERT | `els_platform/training/config.py::BertModelConfig` has no default checkpoint and requires reviewed ID/revision/license/language/resources. `els_platform/training/models.py::load_bert_components` loads local files only and adds a five-logit head. `els_platform/training/engine.py::fit_bert` uses its tokenizer with truncation and the configured length. | No checkpoint ID is selected, no local weights/tokenizer were found, and no BERT fit was executed. Loading will not fetch missing files. |
| Authorization and checkpointing | `els_platform/training/data.py::TrainingAuthorization.validate` rejects unapproved/test-fixture source kinds and requires review assertions. `els_platform/training/checkpoints.py::save_checkpoint` and `load_checkpoint` validate model type, configuration, label order, checksums, dataset reference, and manifest fingerprint. | Approval fields are caller assertions; code cannot authenticate actual institutional decisions. No checkpoint files exist to inspect. |
| Evaluation | `els_platform/training/metrics.py::select_thresholds` chooses per-label F1 thresholds only when called for validation and requires known positive/negative examples. `finalize_evaluation_plan` locks validation thresholds/configuration. `els_platform/training/engine.py::evaluate_model_on_test` checks the fitted split fingerprint, manifest, configuration, thresholds, and authorization before scoring test data. | No real predictions were evaluated. The test gate binds the supplied run to the test split, but it does not enforce a one-time-only invocation. |

The directory inventory contained `data/raw/goemotions_1.csv`, `data/raw/goemotions_1_retry.csv`, and `data/raw/README.md`. `data/processed` contains only `README.md`; no `artifacts` directory or model/checkpoint/tokenizer/manifest files were found. This is a filesystem inventory result, not evidence that artifacts could not exist elsewhere on the machine.

## 2. Participant leakage after persisted-split reload

**Finding: participant separation cannot be independently reverified from the persisted bundle.**

- Before persistence, `els_platform/training/splitting.py::split_dataset` can place all examples sharing a supplied `group_id` in one split. If `group_id` is absent, it groups by normalized text instead and explicitly does not claim participant-level separation.
- `els_platform/training/dataset.py::write_split_outputs` writes only `text`, `labels`, and `known_labels` to each JSONL file; its docstring says group identifiers are intentionally excluded.
- `els_platform/training/dataset.py::_build_training_manifest` stores `groups_per_split`, `grouping_strategy`, and the explicit flags `group_id_values_stored_in_split_files: false` and `group_id_semantics_verified: false`. It does not store group membership or an assignment map.
- On reload, `els_platform/training/dataset.py::_read_split_examples` reconstructs examples from only those three fields. `_validate_loaded_text_separation` checks exact normalized text across partitions, not group IDs. The content fingerprints protect consistency against accidental file changes relative to that manifest; they do not prove that participant groups were disjoint or that source group IDs represented participants.

Therefore, if the split was correctly made with participant IDs before writing, the saved partitions reflect that initial assignment, but a reviewer cannot verify that fact from the reloaded bundle. Reproducible counts and a manifest fingerprint are not sufficient proof of participant separation. This is consistent with the limitation recorded in `PROJECT_STATUS.md` and in Section 4 of `ML_IMPLEMENTATION_PLAN.md`.

## 3. Evaluation implementation review

`els_platform/training/metrics.py::calculate_metrics` calculates per-label TP, FP, and FN over entries marked known, then computes precision, recall, and F1; it also returns macro/micro F1, subset accuracy over rows where all five labels are known, and Hamming loss over known decisions. This is a reasonable masked multi-label basis, and unknown labels are not treated as negatives.

Two qualifications:

1. **Confusion matrices are not implemented.** The function does not count or return TN, nor does it return a per-label `[ [TN, FP], [FN, TP] ]` matrix. The current evaluation output therefore does not satisfy the plan's confusion-matrix criterion.
2. **Zero-positive-support F1 edge case:** the code sets F1 to `None` whenever `support == 0`, even if the model produced false positives. If support is zero but FP is positive, the usual binary F1 formula has a nonzero denominator and F1 is 0; returning undefined and omitting that label from macro F1 can hide false-positive-only behavior. If there are neither actual nor predicted positives, F1 is genuinely undefined. The existing test `TrainingInfrastructureTests.test_metrics_and_undefined_label_metrics` checks the no-positive/no-prediction case but not the no-positive/false-positive case. This convention should be clarified and tested before reporting metrics.

Threshold selection is validation-only in `select_thresholds`; it leaves thresholds undefined when validation lacks a known positive or negative, and `finalize_evaluation_plan` refuses to finalize in that case. `evaluate_model_on_test` requires the same configuration, thresholds, exact in-memory split fingerprint, manifest reference, and authorization used for fitting. These are structural safeguards verified from code; they are not proof of an actual evaluation. No single-label confusion matrix should be used for the multi-label task; use one binary matrix per label and exclude unknown decisions.

## 4. Application behavior and artifacts

`app.py` calls `els_platform.presentation.render_app`. In `els_platform/presentation.py`, `_DEMO_CLASSIFIER` is instantiated as `DemonstrationClassifier`, and `_render_study_challenge` calls its `predict` method. `els_platform/prediction/demo.py::DemonstrationClassifier.predict` returns the same fixed score map for every nonempty input. The presentation labels these as fixed illustrative scores and says no BiLSTM or BERT model is loaded. `els_platform/analytics.py::make_demonstration_figure` supplies illustrative analytics.

Thus the app currently shows **demonstration/fixed-score behavior**, not real text-dependent inference. No trained-model predictor is wired into the Streamlit presentation, and no checkpoint artifact was found. The shared `EmotionClassifier` protocol and `PredictionResult` schema provide a future integration boundary, but they do not establish a trained model.

## 5. Acceptance criteria in `ML_IMPLEMENTATION_PLAN.md`

| Plan criterion | Status | Evidence and limits |
|---|---|---|
| Required ethics/faculty/privacy approvals and finalized collection process | **Unmet** | `DATA_COLLECTION_CHECKLIST.md` says pending and collection is not authorized; `DATA_COLLECTION_PROTOCOL.md` is a draft. No approval evidence is in the reviewed files. |
| Usable, approved source with all five target labels in an appropriate text domain (or formally approved scope change) | **Unmet** | `DATASET_AUDIT.md` reports only direct `confusion` and `curiosity` matches in GoEmotions part 1 and unresolved reuse rights. `DATASET_NEXT_STEPS.md` documents partial/domain-mismatched candidates, not an approved five-label student-text source. |
| Input contract represents positive, known-negative, unknown labels and a group token | **Partially met** | `DATASET_INPUT_CONTRACT.md` and `prepare_records` support `labels`, `known_labels`, and optional `group_id`. However, `data/templates/study_challenge_template.csv` is only `text,labels`, and participant grouping is not preserved in saved splits. No approved collection instrument exists. |
| Privacy-reviewed, auditable group-aware persisted splits | **Unmet** | `split_dataset` can group in memory. Persisted JSONL omits group IDs, and reload verifies exact text only. No prepared split files exist. |
| Text-free audit demonstrates approved data and per-label support under an agreed adequacy rule | **Unmet** | There is an audit of GoEmotions, not an approved five-label target dataset. No approved sufficiency criteria or target split distributions exist. |
| Approval gate blocks demo/test fixtures | **Structurally met; approval truth unverifiable** | `TrainingAuthorization` restricts source kinds and `load_training_splits` checks manifest assertions. The flags and references are supplied by the caller, so code cannot verify the underlying human approvals. |
| CPU smoke and training/evaluation/checkpoint path verified in the actual environment | **Unmet / unverified** | Infrastructure and a smoke command exist. `PROJECT_STATUS.md` records that the PyTorch framework smoke was skipped because PyTorch was unavailable; real fit, save/load, and BERT paths have not run. Mocked tests are not real execution evidence. |
| Evaluation protocol and acceptance criteria fixed before test evaluation | **Partially met** | Validation-only threshold selection and a held-out evaluation gate exist. No held-out data exists, stakeholder performance acceptance thresholds are intentionally unset, and the metrics need a confusion-matrix implementation plus clarification of zero-support F1. |
| App integration approved and predictions supported by all labels | **Unmet** | The app is safely in demo mode. No trained checkpoint, real inference adapter, evaluation result, or separate integration approval exists. |

Some approval facts are not merely absent from code; they cannot be established by software inspection. They require authentic human records. Likewise, metric or model behavior on real data cannot be verified without an approved dataset and actual framework execution.

## 6. Smallest safe implementation sequence

1. **Resolve human gates first.** Decide whether a consent-based study-text collection is permitted; obtain applicable supervisor/institutional/ethics and privacy decisions; finalize participants, consent, label definitions, withdrawal, storage, retention, deletion, and data-use terms. Do not recruit or accept text before authorization. If using public data, obtain source-specific rights approval; GoEmotions alone cannot support the five-class target.
2. **Close the data-contract and grouping gap.** Decide how positive, known-negative, and unknown answers are captured, and how a non-identifying participant grouping token is used. Add an approved, privacy-reviewed way to retain or attest group membership through persisted splits so the loader can verify group disjointness without exposing IDs as model features. Preserve fixtures as non-training sources.
3. **Prepare only authorized data.** Convert approved records to the strict contract; run privacy screening, schema/label validation, exact and near-duplicate review, class/known-label coverage audit, and group-level balance review. Define sufficiency criteria before looking at model results. Stop if any label lacks agreed positive/negative and independent-group support.
4. **Produce and validate persisted splits.** Deduplicate before assignment, split by participant group with a fixed seed, persist split/manifest evidence, and verify counts, text separation, group separation, and label coverage after reload. Fit vocabulary/transforms only on train. Do not use the raw GoEmotions file as five-class study data.
5. **Fix and test evaluation reporting before model claims.** Add per-label binary confusion matrices; settle the zero-support F1 convention and tests; confirm metrics honor known-label masks. Set evaluation objectives and stakeholder acceptance thresholds before opening the held-out test set.
6. **Verify the baseline environment and run the BiLSTM only after approval.** Separately approve/install optional dependencies, run the actual CPU smoke test, then train the BiLSTM on approved train data, select model/thresholds on validation, save reviewed metadata/checkpoint, and evaluate the locked configuration on test. Report only actual outputs and limitations.
7. **Consider BERT only if justified.** Assess independent sample/class coverage and available hardware first. Then review and pin an exact checkpoint, tokenizer, language fit, and license before separately approving its acquisition and comparing it on the same frozen splits/protocol.
8. **Integrate only after acceptance.** Build a model inference adapter for the shared schema, validate checkpoint metadata and every supported label, and conduct a separate privacy/UI review. Until then, retain the explicit demonstration mode and make no model-performance claims.

## Read-only boundary

This report adds no implementation and makes no model or approval claims. The recommendations above are a proposed safe sequence based on the inspected code and project documents; data collection, dataset rights, hardware sufficiency, and real model performance remain unresolved.
