# Implementation Plan

**Project:** AI-Driven Emotion Detection & Personalized Learning Support Platform  
**Phase:** 2B — research-informed implementation planning  
**Status:** Planning direction approved. The shell, dataset-preparation pipeline, and model-training infrastructure have been implemented in later phases; current milestone and verification details are tracked in [PROJECT_STATUS.md](PROJECT_STATUS.md) and [TRAINING_GUIDE.md](TRAINING_GUIDE.md). No dataset or model has been acquired or trained.

## 1. Architecture and responsibilities

The app will use a small Streamlit presentation layer over separately testable Python modules. Model inference will run locally when a model artifact is available. Each output is an estimate based on text, not a reliable reading of a learner's internal state and not a clinical assessment.

| Component | Responsibility |
| --- | --- |
| `app.py`, `els_platform/presentation.py` | Streamlit page flow, input validation, model selection/comparison, output display, consent and logging controls. |
| `els_platform/preprocessing.py` | Length checks and conservative text normalization shared by training and inference; preserve negation and meaningful punctuation. |
| `els_platform/prediction/schema.py`, `interface.py` | Stable prediction result type and a common interface for model adapters. |
| `els_platform/prediction/bilstm.py`, `bert.py` | Load local artifacts and return the shared result schema; no training code in the UI. |
| `els_platform/training/` | Dataset loading, reproducible splitting, BiLSTM/BERT training, validation threshold selection, and evaluation. |
| `els_platform/enhancement.py` | Versioned, explainable keyword/phrase rules; preserve model scores and record rule signals separately. |
| `els_platform/guidance.py` | Optional Gemini guidance, key lookup, timeouts/error handling, and a local fallback. |
| `els_platform/logging.py` | Opt-in append/clear operations for privacy-minimized CSV interaction records. |
| `els_platform/analytics.py` | Read local aggregate records and prepare Plotly-compatible summaries without sending data elsewhere. |
| `data/` | Documented locations for approved source data and derived training data; not a place for unreviewed student text. |
| `tests/` | Unit and integration checks for schemas, rules, privacy, model adapters, and UI behavior. |

## 2. Proposed repository structure

Files marked **planned** are for a later implementation phase; only the phase-one files and package markers are created now.

```text
.
├── .env.example
├── .gitignore
├── IMPLEMENTATION_PLAN.md
├── PROJECT_STATUS.md
├── README.md
├── requirements.txt
├── requirements-models.txt       # optional training/inference stack
├── app.py                        # planned Streamlit entry point
├── els_platform/
│   ├── __init__.py               # created package marker
│   ├── preprocessing.py          # planned
│   ├── presentation.py           # planned
│   ├── prediction/
│   │   ├── __init__.py           # created package marker
│   │   ├── schema.py             # planned shared output schema
│   │   ├── interface.py          # planned
│   │   ├── bilstm.py             # planned inference adapter
│   │   └── bert.py               # planned inference adapter
│   ├── training/
│   │   ├── dataset.py            # local input, audit, and split preparation
│   │   ├── config.py             # shared and model-specific settings
│   │   ├── data.py               # authorization, leakage, and target checks
│   │   ├── vocabulary.py         # train-only BiLSTM vocabulary
│   │   ├── models.py             # lazy BiLSTM and local-only BERT builders
│   │   ├── engine.py             # fit loops and gated test evaluation
│   │   ├── metrics.py            # multi-label metrics and thresholds
│   │   └── checkpoints.py        # safe metadata and local state-dict persistence
│   ├── enhancement.py            # planned
│   ├── guidance.py               # planned
│   ├── logging.py                # planned
│   └── analytics.py              # planned
├── data/
│   ├── raw/README.md
│   └── processed/README.md
├── artifacts/models/             # local model weights; ignored by Git
├── logs/                         # local generated logs; ignored by Git
└── tests/                        # planned; no tests created in Phase 1
```

No `.streamlit/config.toml` is included yet. There is no settled theme or non-default Streamlit behavior that needs configuration. Deployment secrets, if used, belong in the ignored `.streamlit/secrets.toml` or the deployment platform's secret store. The training package and its tests now exist; see `TRAINING_GUIDE.md` for implementation boundaries and environment findings.

## 3. Dataset acquisition, licensing, and labeling

The five application labels remain **Bored**, **Confident**, **Confused**, **Curious**, and **Frustrated**. There is no approved or acquired target dataset. Do not treat any reviewed public corpus as ground truth for all five categories. See [DATASET_AND_MODEL_RESEARCH.md](DATASET_AND_MODEL_RESEARCH.md) for the reviewed sources and limitations.

### Primary target data: unresolved

The preferred route for a future five-label evaluation set is a documented, consent-based collection of short study-challenge descriptions, but the source, approval process, participant population, and collection instrument are not yet selected. Do not solicit, collect, or retain identifiable or sensitive student data. Before any collection, document consent, purpose, retention/deletion, access, and labeling procedures; obtain any required institutional/guardian approval; and collect only non-identifiable, non-sensitive educational text. Do not ask for names, student IDs, grades, health information, or other sensitive details. Document how to discard or redact any such details a participant accidentally includes before a text enters a retained dataset. If safe and approved collection is not feasible, keep the classifier exploratory and make no validation or performance claims.

For each retained example, ask the participant which of the five labels apply to their own study challenge at that moment, allowing multiple selections. Keep **“none of these”** separate from **“not sure / prefer not to label”** and from skipped or missing responses. A label is negative only when it is explicitly answered as not applying; unknown labels are masked from training and metrics. Preserve label provenance and do not conflate a participant’s self-report with annotator judgments. If an independent annotation subset is approved, use separate annotators, adjudicate disagreements, and report agreement. These are reports about a text/context, not objective measurements of a person’s internal state.

### Optional public auxiliary data and mapping

GoEmotions may be considered as an **optional auxiliary dataset**, subject to separate rights review. Its official [README](https://github.com/google-research/google-research/blob/master/goemotions/README.md) describes English Reddit comments, not study-challenge reports, with 27 emotion labels plus Neutral and multi-label annotations. Its dataset-specific text reuse terms must be resolved; the Google Research code repository’s Apache license must not be assumed to license third-party Reddit text. Do not download or use it before that review.

The only source-to-application mappings proposed for a clearly marked auxiliary experiment are:

| GoEmotions source label | Application label | Treatment |
| --- | --- | --- |
| `confusion` | **Confused** | Defensible label-name/concept mapping; retain the original source label and annotator provenance. |
| `curiosity` | **Curious** | Defensible label-name/concept mapping; retain the original source label and annotator provenance. |
| Any other label, including `annoyance`, `optimism`, `pride`, or `Neutral` | None | Leave unmapped. Do not use it to infer Bored, Confident, Frustrated, or “none of these.” |

Do not turn a single GoEmotions label into a five-label target vector by filling all other categories with zero. In an auxiliary experiment, keep mapped labels and unknown targets explicit, retain source-specific evaluation separately, and do not present auxiliary scores as performance on student text. The eventual choice to use GoEmotions for a separate auxiliary head, masked partial-label experiment, or no model training at all remains unresolved pending license and method review.

Other reviewed candidates are not approved training sources: [EduAffect](https://link.springer.com/article/10.1007/s44217-026-02098-1) is student-focused text but uses one dominant GoEmotions label per entry and covers only confusion/curiosity among the target concepts; [EmpatheticDialogues](https://huggingface.co/datasets/facebook/empathetic_dialogues) has a single situation label and an exact Confident class, but is not learning-domain text and is CC BY-NC 4.0; [DAiSEE](https://people.iith.ac.in/vineethnb/resources/daisee/) is restricted, video-based data; and the reviewed MEED repository has no visible dataset license. Do not combine these sources or treat their different annotation schemes as interchangeable. Re-review licensing and provenance before any future use.

### Demonstration examples

If sample text is needed to exercise a UI, fixture, or schema, label every item prominently as a **demonstration example**. Do not describe demo examples as observations, collected learner reports, annotated ground truth, or validated training/evaluation data. Keep them out of dataset manifests, training, validation, and test partitions.

Finalize category definitions before annotation. Distinguish confusion from frustration, curiosity from confidence, and boredom from low engagement or a merely short response. Keep the five application labels stable even if optional auxiliary sources use other label taxonomies.

## 4. Five-category strategy and model integration

Use a **multi-label** target rather than a five-way softmax for approved target examples: more than one category can be reported. Train both model families with five output scores (sigmoid heads and binary cross-entropy over known labels for the initial design). Mask unknown labels. Never assume a single-label public corpus or classifier naturally provides mixed-emotion ground truth.

The shared prediction schema should include a schema version, model name/version, timestamp in UTC, a score for each of the five categories, validation-selected thresholds, selected label(s), and any rule signals. Keep `raw_model_scores` distinct from rule-adjusted scores and rule-produced signals. Call values confidence/probability only when calibration supports that description; otherwise label them model scores. Two or more labels form a mixed output only when each independently clears its validation-selected threshold. If no label clears its threshold, return an explicit “no clear category” result instead of forcing the top score into a label. Unit tests must cover zero, one, and multiple selected labels and stable ordering/schema behavior.

**BiLSTM:** use a training-only tokenizer/vocabulary and a compact trainable embedding plus bidirectional LSTM and dropout. Save vocabulary/tokenization metadata with weights. This avoids assuming a licensed external embedding is available and is practical for a small CPU experiment, though data quantity and generalization will be limited.

**BERT:** fine-tune a suitable pretrained encoder for the same five multi-label targets using the same approved target examples and split. Research identified two candidates, but no checkpoint is selected: `distilbert/distilbert-base-uncased` (lighter first experiment; English uncased; Apache-2.0, [model card/config](https://huggingface.co/distilbert/distilbert-base-uncased)) and `google-bert/bert-base-uncased` (standard comparator; English uncased; Apache-2.0, [model card](https://huggingface.co/google-bert/bert-base-uncased)). These are pretrained language encoders, not target-emotion classifiers; use the matching tokenizer and add a five-logit head. Confirm the exact revision, license, storage size, language fit, and machine resource needs before selection. Download weights only after approval for that step; never commit them.

Both adapters must emit the same schema and be compared on identical held-out examples from the same approved target dataset. Keep inference local by default. Rule-based keyword enhancement is a separate, versioned layer: account for negation/context, explain matches, evaluate it independently, and never describe keyword matches as a trained model or as a person's true emotion. If GoEmotions is later used as auxiliary data, report its source-specific task and metrics separately from the five-label target evaluation.

## 5. Evaluation and reproducibility

Use fixed seed `42` and keep a reproducible split manifest. Split by participant (and by source/conversation where relevant) to prevent near-duplicate or same-person leakage; aim for 70% train, 15% validation, and 15% test with group-aware label balance. The exact split must be frozen before model comparisons and reused for BiLSTM and BERT. If the dataset is too small for a stable test set, document that limitation and use group-based cross-validation for development; do not tune against the final test set.

Report macro F1 and **per-label precision, recall, F1, and support**, micro F1, per-label/macro PR-AUC where sample counts support it, exact-match/subset accuracy, multilabel confusion summaries, and error review. Report calibration diagnostics only when data volume and method support them. Select a threshold for each label on validation data only using a documented, preselected objective; freeze thresholds before test evaluation. Use the same split, eligible examples, and known-label mask for both models. Compare any rule-enhanced variant on that same test partition and report raw scores separately. Do not publish or imply performance claims until the evaluation has actually run; do not use ordinary accuracy alone for this imbalanced, multilabel task.

## 6. Gemini guidance and fallback

Read `GEMINI_API_KEY` from the process environment (with optional local `.env` loading) or Streamlit secrets. Never put the key in source, logs, UI output, or a committed secrets file. Gemini should be opt-in at the moment guidance is requested, with a clear notice that the submitted description is sent to Google's service. Do not solicit or send identifiable or sensitive student data. Send only the non-sensitive text and minimal context needed for the requested tip; do not send logs or identifiers. Handle missing keys, timeouts, quota/network errors, and malformed responses with a short local, non-generative fallback. Keep guidance educational, supportive, and non-diagnostic; do not claim it is based on certainty about the learner's state.

## 7. CSV logging, privacy, and analytics

Logging should be off by default and explicitly enabled by the user/operator. Store only a UTC timestamp, model/version, selected labels, category scores, rule version/signals, and guidance success/source as appropriate. Do **not** store the free-text description, sensitive content, API key, student name, or identifiers in the interaction CSV. Do not use interaction logs as training data. Keep logs local and gitignored. Provide controls to disable logging and clear the local log; document the file location, retention responsibility, and that clearing a file cannot retract data already sent to Gemini. Analytics should use only available aggregate/minimized records and handle an empty or missing CSV. Plotly views may show counts and confidence trends, with minimum-count suppression considered before any shared or cohort-level view.

## 8. Testing strategy

Use Python's standard `unittest` initially to avoid a required test framework dependency. Add checks for preprocessing/negation, category vocabulary and schema validation, threshold selection and mixed-label cases, model adapter consistency, rule conflicts, missing-key/Gemini failure fallback (mock network clients), CSV columns and absence of raw text/secrets, log disable/clear behavior, empty analytics, and split leakage/reproducibility. Add small offline smoke checks for the Streamlit flow after an app exists. Tests should use temporary directories and clearly marked demonstration fixtures, not real student data or live Gemini calls. Demonstration fixtures must never be presented as observations or included in train/validation/test datasets. No tests are included or run in Phase 1.

## 9. Setup and deployment requirements

Target Python 3.11 for development. The UI/guidance dependencies are in `requirements.txt`; optional PyTorch and Transformers dependencies are isolated in `requirements-models.txt`. The implemented required metrics use the standard library, so scikit-learn is not currently required. For CPU development, a small BiLSTM is the first practical training path; BERT can run with a small batch/sequence length but may be slow. A CUDA-capable GPU is recommended for practical BERT fine-tuning, but is not required to develop the UI or run small experiments. Install PyTorch using the official selector for the target OS/CUDA setup when needed.

Before deployment, pin and resolve dependencies in the chosen target environment; confirm Python and model-framework compatibility; configure secrets outside source control; provide reviewed model artifacts through secure storage; and check memory, startup time, request latency, ephemeral-disk behavior, log retention, and concurrent CSV writes for the selected host. Keep the app usable with Gemini disabled and with a missing model artifact (clear availability message). No deployment target has been selected.

## 10. Technical risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Few or weakly labeled study-specific examples | Start with a narrow opt-in study set, track provenance, use group splits, and make no performance claims until evaluation supports them. |
| Public corpus labels/domain do not match the five categories | Keep GoEmotions optional; map only `confusion` to Confused and `curiosity` to Curious, leave other target labels unknown, and keep auxiliary results separate. Use only approved consent-based examples for target evaluation. |
| Text prediction is mistaken for a fact about a person | Use estimate-oriented UI copy, calibrated/qualified scores, an abstention path, and no diagnostic language. |
| Mixed labels are forced by a single-label setup | Use a multi-label objective, validation-tuned thresholds, explicit abstention, and tests for multiple outputs. |
| Keyword rules over-trigger on negation or context | Keep rules transparent and separate, test conflicts/negation, and compare their effect on held-out data. |
| BERT training is too slow or artifacts exceed host limits | Keep the UI CPU-compatible, use small batches/sequence lengths, recommend GPU for fine-tuning, and assess compact encoders and deployment resources. |
| Text or credentials are retained or exposed | Logging off by default, exclude raw text and secrets, gitignore local data/artifacts, minimize Gemini payloads, and provide log clearing. |
| Gemini is unavailable or sends text to an external provider | Explicit opt-in/disclosure, minimal payload, timeout/error handling, and deterministic local fallback. |

## 11. Current phase boundary and unresolved decisions

The planning direction is approved and the shell, preparation pipeline, and training infrastructure are implemented. This plan does **not** authorize downloading data or weights, installing optional model dependencies, collecting examples, training models, calling Gemini, or deploying. Model training and data acquisition still require their own explicit human approval.

The following choices remain open and must be recorded before training:

| Decision | Current direction | Still unresolved |
| --- | --- | --- |
| Primary target dataset | Only approved, consent-based, non-identifiable and non-sensitive study-challenge examples can provide five-label target supervision. No dataset is selected or collected. | Whether an appropriate collection can be approved; participant population; responsible collector; consent/ethics process; retention/deletion; and whether the project will remain exploratory if collection is not feasible. |
| GoEmotions use and mapping | Optional auxiliary only. `confusion` → Confused and `curiosity` → Curious are the only proposed mappings. Other labels and Neutral remain unmapped; never infer Bored, Confident, or Frustrated from them. | Resolve dataset-specific use rights; decide whether to omit it or use a separately reported auxiliary head / masked partial-label experiment; decide how to use source rater negatives without treating them as self-report ground truth. |
| BERT checkpoint | Candidates are `distilbert/distilbert-base-uncased` and `google-bert/bert-base-uncased`; both are English uncased Apache-2.0 checkpoints, not target emotion classifiers. | Select the first checkpoint and whether to include the second as comparator; confirm language suitability, fixed revision, license at acquisition, storage/memory fit, and hardware. |

The fixed comparison protocol remains seed `42`, a participant/source-grouped reproducible split, validation-only threshold selection, and per-label metrics for both BiLSTM and BERT on the same eligible held-out examples. If approved examples are unavailable, do not fabricate labels or model results.
