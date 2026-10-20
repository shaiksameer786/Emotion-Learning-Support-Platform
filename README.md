# AI-Driven Emotion Detection & Personalized Learning Support Platform

A Streamlit application shell, local dataset-preparation pipeline, and approval-gated training infrastructure for a planned learning-support platform. The current application demonstrates the interface and shared five-label prediction schema; it does **not** contain a trained emotion classifier or an approved training dataset.

## Current shell

- Study-challenge text input with empty-input validation.
- Five target labels: Bored, Confident, Confused, Curious, and Frustrated.
- A shared prediction schema with score validation, per-label thresholds, multiple selected labels, and an explicit no-clear-category result.
- Clearly labelled, fixed demonstration scores. They are the same for every non-empty input and do not analyze its contents; they are not validated model probabilities.
- General local study tips. No Gemini call or other external AI request is made.
- A Plotly analytics page containing illustrative fixture counts only.
- Interaction logging is off by default. If the metadata-only logging helper is enabled in code, raw text and input references are not written.
- A local CSV/JSONL validation, audit, and reproducible split pipeline is available. It does not fetch or select a dataset and does not train a model.
- Future BiLSTM/BERT training and evaluation infrastructure is present behind optional dependencies and explicit data-approval checks. It has not been run with a real dataset or model weights.

The implementation is separated into `els_platform/prediction` (shared schema, classifier interface, and demo fixture), `guidance.py`, `analytics.py`, `interaction_log.py`, and `presentation.py`; `app.py` is the Streamlit entry point. Automated checks are in `tests/`.

There is no trained BiLSTM or BERT model, pretrained checkpoint, or performance result. A raw GoEmotions part-one dataset is present locally but has unresolved reuse rights and does not provide all five intended labels. Future emotion outputs are intended as text-based estimates, not reliable readings of internal emotional states or diagnoses.

## Prerequisites

- Python 3.11.
- The existing workspace virtual environment at `.venv` (already created for this project).
- Base packages listed in `requirements.txt` (currently pinned to Streamlit 1.66.0 and Plotly 6.9.0; Streamlit 1.66 provides the chart/table description APIs used by the shell).

The optional model-training stack is not needed to run this shell. PyTorch is installed in the current `.venv`; Transformers is not installed. Training infrastructure details and current environment findings are in [TRAINING_GUIDE.md](TRAINING_GUIDE.md). Gemini integration is not implemented and no API key is needed.

## Install base dependencies

From the project root in PowerShell, use the existing environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For a fresh checkout without `.venv`, create it first with Python 3.11:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the application

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit. Choose **Study challenge**, enter any non-empty text, and select **Analyse demonstration** to explore the fixed illustrative score table and local study tips. Choose **Demonstration analytics** to view fixture-only chart data. No trained checkpoint is required. The interface identifies the five intended outputs and states that genuine training is pending suitable, approved data and human review.

## Run tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Prepare a reviewed dataset later

Read [DATASET_INPUT_CONTRACT.md](DATASET_INPUT_CONTRACT.md) before supplying any source data. A local GoEmotions part-one CSV and a duplicate retry copy exist under `data/raw/`; the source provides direct concept-level labels only for Confused (`confusion`) and Curious (`curiosity`) among the five target categories. Its Reddit-text reuse rights remain unresolved, and it is not approved for training. The prepared initial CSV is header-only. No suitable, approved five-label dataset or trained checkpoint is available. Any training input must be locally available, appropriately licensed and reviewed, non-identifying, and restricted to approved text/label fields. The contract documents CSV/JSONL formats, partial-label handling, explicit label mappings, duplicate behavior, audit privacy, and split limits.

After review, run the preparation-only command from the project root:

```powershell
.\.venv\Scripts\python.exe -m els_platform.training.dataset data/raw/reviewed_source.csv
```

It writes a text-free audit report to `data/processed/dataset_audit.json` and, if validation succeeds, local prepared splits under `data/processed/splits/`. The prepared split files contain the supplied text and are Git-ignored; restrict access to them. The default split seed is `42`. For a source with complete judgments of all five labels, and only after verifying that fact, add `--assume-all-labels-known`. No model-training dependencies are required for preparation, and this command does not train or evaluate models.

## Privacy and limitations

The shell does not send text to Gemini or another external AI service, does not retain the submitted text in a file, and does not log raw text. Please do not enter identifying or sensitive information. The prediction fixture is static and illustrative; local tips are general suggestions rather than emotion-personalized guidance. The analytics chart is not based on learner activity.

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the approved planning direction and [DATASET_AND_MODEL_RESEARCH.md](DATASET_AND_MODEL_RESEARCH.md) for dataset/model research and unresolved decisions. See [PROJECT_STATUS.md](PROJECT_STATUS.md) for the current milestone status.
