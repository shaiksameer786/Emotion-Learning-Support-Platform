# Dataset and Model Research

**Project:** AI-Driven Emotion Detection & Personalized Learning Support Platform  
**Research date:** 2026-10-20  
**Scope:** Source review and recommendations only. No dataset or model weights were downloaded, no packages were installed, and no training or inference was run.

## Executive findings

- The Phase 1 plan had not selected a dataset or a pretrained checkpoint. It named GoEmotions as a candidate and proposed a consent-based, multi-label study-challenge dataset as the target source.
- No reviewed text dataset directly provides trustworthy, multi-label study-challenge examples for all five target categories.
- **EduAffect** is the closest text-domain candidate found: it contains student-focused academic and campus discourse, but assigns one dominant emotion from the GoEmotions taxonomy to each entry. The taxonomy directly covers only *confusion* and *curiosity* among the five requested categories.
- Public corpora can inform exploratory work, but they cannot be combined or relabeled into five-category multi-label ground truth without unsupported assumptions. The recommended target training and evaluation data is a small, consent-based collection with explicit participant self-report for each target label.
- Candidate English checkpoints are `distilbert/distilbert-base-uncased` as the lighter first BERT-family experiment and `google-bert/bert-base-uncased` as a standard BERT-base comparator. Both are pretrained language models, not emotion classifiers.

## 1. What the Phase 1 documents specify

The reviewed `IMPLEMENTATION_PLAN.md`, `README.md`, and `PROJECT_STATUS.md` describe a Streamlit app with separate preprocessing, model interfaces/adapters, training/evaluation, keyword rules, optional Gemini guidance, CSV logging, analytics, and tests. The shared model output is planned to carry scores for five categories and allow multiple selected labels via independently thresholded sigmoid scores.

The plan has no chosen dataset, no selected checkpoint, and no model artifacts. Its planned split is participant/source-aware, seed `42`, approximately 70% train / 15% validation / 15% test. Those are plans, not executed results. The README targets Python 3.11. The user reported that the Python 3.11 virtual environment was created and verified in Phase 2A; project packages have not been installed or resolver-checked.

## 2. Public dataset candidates

“Direct” below means that the source dataset has a category with the same meaning as the target. It does not mean the dataset measures a learner’s true internal state. Nominal source labels such as *confusion* are normalized to the corresponding UI adjective *Confused* only as a label-name normalization. No unrelated emotion is substituted for a target.

| Dataset | Source, language, and data | License / access | Labels and multi-label status | Direct coverage of the five targets | Fit and limitations |
|---|---|---|---|---|---|
| **EduAffect** | [Dataset DOI / Kaggle record](https://doi.org/10.34740/kaggle/dsv/13091236); [original dataset paper](https://link.springer.com/article/10.1007/s44217-026-02098-1). 2,423 short entries from university students in Ghana, drawn from public social posts and structured survey answers about academic pressures, campus life, and related experiences. English and English–Twi code-switching; some Twi content was translated during curation. | The paper’s data-availability section says the dataset is publicly available under **CC BY 4.0**. The paper itself separately uses **CC BY-NC-ND 4.0**; do not confuse the article license with the dataset license. Verify the Kaggle record and any underlying platform terms before acquisition or redistribution. | 28 GoEmotions-framework categories (27 emotions plus Neutral), but **one dominant label per entry**. No multi-label targets. | Bored: no. Confident: no. Confused: yes, via *confusion*. Curious: yes, via *curiosity*. Frustrated: no. | Strongest student-domain text candidate reviewed, but not specifically all study-challenge descriptions; it is from one Ghanaian institution and includes social/campus discourse. Its English–Twi language mix may not fit the intended user population. Single-dominant labels do not support genuine mixed-emotion training. The authors report imbalance and recommend a narrower taxonomy for downstream use. |
| **GoEmotions** | [Official Google Research dataset README](https://github.com/google-research/google-research/blob/master/goemotions/README.md); [original ACL paper](https://aclanthology.org/2020.acl-main.372/); [model card](https://github.com/google-research/google-research/blob/master/goemotions/goemotions_model_card.pdf). 58,009 English Reddit comments. | The dataset README reviewed here does not specify a separate license for the Reddit text. The broader [Google Research repository license](https://github.com/google-research/google-research/blob/master/LICENSE) is Apache 2.0, but that alone does **not** establish that third-party Reddit comments are Apache-licensed. Review dataset-specific reuse and Reddit content terms before use or redistribution. Public availability is not itself a license. | 27 labels plus Neutral: admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, Neutral. Original annotations have one binary indicator per category per rater; multiple emotions can be marked. The agreement-filtered files retain comma-separated emotion IDs and can include multiple labels. | Bored: no. Confident: no. Confused: yes, via *confusion*. Curious: yes, via *curiosity*. Frustrated: no. | Useful text-only multi-label research corpus, but Reddit conversations are not student study-challenge descriptions. It includes potentially problematic content and documented representation/annotation biases. Do not map *annoyance* to Frustrated or *optimism/pride/approval* to Confident. |
| **EmpatheticDialogues** | [Official archived repository](https://github.com/facebookresearch/EmpatheticDialogues); [original ACL paper](https://aclanthology.org/P19-1534/); [dataset card](https://huggingface.co/datasets/facebook/empathetic_dialogues). About 25,000 English crowdsourced personal situations and dialogues. | **CC BY-NC 4.0** per the official repository license and dataset card: attribution is required and non-commercial terms apply. The repository is archived read-only. | 32 situation-level emotion labels: Surprised, Excited, Angry, Proud, Sad, Annoyed, Grateful, Lonely, Afraid, Terrified, Guilty, Impressed, Disgusted, Hopeful, Confident, Furious, Anxious, Anticipating, Joyful, Nostalgic, Disappointed, Prepared, Jealous, Content, Devastated, Embarrassed, Caring, Sentimental, Trusting, Ashamed, Apprehensive, Faithful. A situation is grounded in **one selected emotion label**; it is not multi-label ground truth, even though conversations may mention related feelings. | Bored: no. Confident: yes. Confused: no. Curious: no. Frustrated: no. | *Confident* is an exact source category. Data concerns empathetic open-domain conversation rather than learning challenges. Do not map *annoyed* to Frustrated or positive labels to Confident. The dataset card leaves several provenance, annotation, and personal-information fields as “More Information Needed.” |
| **DAiSEE** | [Official DAiSEE page](https://people.iith.ac.in/vineethnb/resources/daisee/); [original paper](https://arxiv.org/abs/1609.01885). 9,068 video snippets from 112 users in e-learning/computer-use settings. Labels describe visual/behavioral affect, not text. | Access requires a request form and agreement to the site’s terms. The site says **research use only**, prohibits providing images to second parties, and describes a download of about 15 GB. No standard open license was identified on the official page; the terms, not an assumed license, govern access. | Boredom, confusion, engagement, and frustration, each rated at four ordinal levels (very low, low, high, very high); the source describes it as multi-label video classification. | Bored: yes (*boredom*). Confident: no. Confused: yes (*confusion*). Curious: no. Frustrated: yes. | Relevant to e-learning but not text. It cannot train the planned text classifier without a separate aligned text modality, which the official description does not offer. Do not infer text emotion labels from faces or video. |
| **MEED (Multimodal Epistemic Emotion Dataset)** | [Public repository README](https://github.com/Nemo9303/MEED-dataset). Repository describes utterance transcripts and pre-extracted multimodal features from collaborative problem-solving dialogues; 2,995 utterances across 210 dialogues. | No dataset license was visible in the reviewed repository (which lists a README and feature file). Publicly visible files do not establish reuse rights. Obtain a clear license and provenance/consent documentation before use. | Repository lists Neutral, Delight, Curiosity, Confusion, and Frustration; each utterance is assigned one category, so this is single-label classification. Text includes transcripts, plus 1024-dimensional BERT-Large vectors, acoustic and visual features. | Bored: no. Confident: no. Confused: yes. Curious: yes. Frustrated: yes. | Potentially relevant to student problem-solving, but the repository alone does not establish license, collection protocol, or broad suitability. Small class counts (the README reports 109 Frustration utterances) and transcript/dialogue context differ from free-text learner descriptions. Treat as a lead to investigate, not an approved data source. |

### Interpretation of coverage

No reviewed public **text** corpus has all five target categories as documented labels in a study-challenge setting. Labels like *sadness*, *anger*, *joy*, *fear*, *annoyance*, *optimism*, and *pride* are not equivalent to Bored, Confident, Confused, Curious, or Frustrated and are not mapped to them here. DAiSEE contributes some target concepts but only through video/behavioral annotations. MEED provides three relevant concepts in transcripts, but its reuse rights and provenance are unresolved.

## 3. Recommended data strategy

Use a **consent-based, study-specific supplementary dataset as the primary target dataset**. Treat EduAffect as the best public text-domain comparison/auxiliary candidate only after checking the Kaggle record, source terms, and the intended use. It may help assess educational-domain language, but its labels cannot be presented as five-class multi-label ground truth. GoEmotions can be explored separately for its *confusion* and *curiosity* labels after rights review; EmpatheticDialogues can inform an auxiliary *Confident* category only if the non-commercial license fits the project. Keep sources and tasks separate in reporting. Do not merge these corpora and claim a common ground-truth label scheme.

For target collection, present a brief, neutral definition for each of the five categories and ask a participant which labels fit their own study challenge **right now**, allowing more than one. Include distinct responses for “none of these” and “not sure / prefer not to label.” Record each label as positive, explicit negative, or unknown; an unanswered checkbox must not automatically become a negative. Keep participant self-report provenance separate from any independent annotator labels. Use appropriate consent and institutional review, and additional guardian/ethics safeguards if minors are involved. Remove direct identifiers, set a retention/deletion policy, and do not collect more personal data than needed.

Self-report remains a participant’s report at that moment, not an objective measurement of an internal state. If human annotators label a subset, treat those judgments as a separate annotation source, use independent annotation and adjudication, and report agreement. If collecting suitable data is not approved or feasible, limit the project to an exploratory demo and do not claim a validated five-category classifier.

## 4. Multi-label strategy and model training

The reviewed public corpora do not supply a common, five-target multi-label dataset:

- GoEmotions is genuinely multi-label at its source, but its target taxonomy covers only two of the five categories directly and is Reddit-domain text.
- EduAffect and EmpatheticDialogues use one dominant/situation label per entry. Converting those labels to five binary columns would create artificial negatives or erase co-occurring feelings. They do not become multi-label data through a one-hot conversion.
- DAiSEE has multiple affect dimensions, but they are video-based ordinal annotations.
- MEED assigns one class per transcript utterance and has unresolved licensing.

For collected target data, train a five-output model with one logit per category, sigmoid outputs, and binary cross-entropy over **known** labels only. Mask unknown labels in the loss. An explicit “none” response means all five are negative; “not sure” means the applicable targets are unknown, not all negative. A mixed-emotion result is the set of labels that independently pass their saved validation thresholds. If no label passes, return a no-clear-category outcome. Scores should not be called calibrated probabilities unless calibration is evaluated.

**BiLSTM baseline:** fit a vocabulary/tokenizer using only the training split, use a compact trainable embedding and bidirectional LSTM with dropout, and save its vocabulary and preprocessing configuration with the weights. No external embedding checkpoint is selected.

**BERT-family candidates:** fine-tune the pretrained encoder with a new five-logit multi-label head on the same approved target examples. Load the tokenizer from the same model ID and pin the model revision in the eventual training manifest. The checkpoints are language encoders pretrained with language-model objectives; neither is already a classifier for these target categories.

### Candidate checkpoints

| Exact model ID | Authoritative source | Language / license | Intended role and resource notes |
|---|---|---|---|
| `distilbert/distilbert-base-uncased` | [Hugging Face model card and files](https://huggingface.co/distilbert/distilbert-base-uncased); [config](https://huggingface.co/distilbert/distilbert-base-uncased/blob/main/config.json); [DistilBERT paper](https://arxiv.org/abs/1910.01108) | English, uncased; model repository declares **Apache-2.0**. | Recommended first transformer experiment. The published config specifies 6 layers, hidden dimension 768, vocabulary size 30,522, and 512 maximum positions. Smaller architecture than BERT-base, but no local runtime or memory benchmark was performed. Use the tokenizer from this exact checkpoint. |
| `google-bert/bert-base-uncased` | [Hugging Face model card and files](https://huggingface.co/google-bert/bert-base-uncased); [original BERT paper](https://arxiv.org/abs/1810.04805) | English, uncased; model repository declares **Apache-2.0**. | Standard BERT-base comparator. The model card identifies 110 million parameters and MLM/next-sentence pretraining; it is not a task-specific emotion model. Use the matching tokenizer. Expect a greater memory/compute burden than DistilBERT, subject to actual batch/sequence settings and hardware. |

These licenses apply to the checkpoint repositories as listed; they do not grant rights to training data used in the project. Pin a specific revision and re-check its card/license at acquisition time. Do not download either checkpoint in this research phase.

## 5. Evaluation and fair comparison

Use the same approved examples, label definitions, known-label masks, text preprocessing policy, and participant/source-grouped split for BiLSTM and BERT. Retain the Phase 1 proposal of seed `42` and roughly 70/15/15 train/validation/test, adjusting only if the collected cohort is too small or group distribution makes that split unusable. Save a split manifest; keep a participant or dialogue wholly in one partition. Check target-label counts and group coverage before fitting. If the dataset is too small for a defensible held-out test, use grouped cross-validation during exploration and label conclusions preliminary.

Report at minimum:

- Macro F1 across target labels and **per-label precision, recall, F1, and support**.
- Micro F1 and per-label/macro precision-recall AUC where sample counts permit.
- Exact-match/subset accuracy, clearly defined as all predicted labels matching the known label set for an example; this is stringent and should not stand alone.
- Error review by label and an account of missing/unknown targets. Calibration diagnostics (for example, Brier score) only if the sample size and method support them.

Select a separate threshold for each label using validation data only, with a documented objective chosen before tuning (for example, best validation F1 or a predeclared minimum precision/recall). Freeze thresholds before looking at test results. Report the thresholds and validation method. Never tune thresholds or select models on the final test set. For fair comparisons, rule-enhanced results must use the same held-out examples, with raw model outputs shown separately from rule signals.

## 6. CPU/GPU and operating constraints

- A CPU-only machine can run the planned Streamlit presentation once its base app dependencies are installed. The UI should handle missing model artifacts with an availability message; no app code exists yet, so the application is not currently runnable.
- A small BiLSTM experiment is the practical CPU-first training path. Its feasibility and usefulness still depend on the amount and quality of approved labeled text.
- CPU inference for BERT-family models may be possible, but latency and memory depend on checkpoint, batch size, sequence length, framework build, and machine. BERT fine-tuning on CPU is possible in principle but likely to be slow for an interactive project schedule.
- A CUDA-capable GPU is recommended for practical BERT fine-tuning. No exact VRAM minimum or elapsed-time estimate is stated because the host hardware, dataset size, batch size, sequence length, and precision are not known and no benchmark was run. DistilBERT is the lower-resource candidate, not a guarantee of acceptable speed on this computer.
- The current machine’s RAM/GPU was not inspected. Do a hardware check only when implementation/training is approved, then select batch length and device accordingly.

## 7. Dependencies: base application versus optional model stack

The current `requirements.txt` lists `streamlit`, `pandas`, `plotly`, `python-dotenv`, and `google-genai`. The current `requirements-models.txt` lists `torch`, `transformers`, and `scikit-learn`. No duplicate package names appear across the two files.

Recommended grouping:

| Dependency | Suggested role |
|---|---|
| `streamlit`, `pandas`, `plotly` | Base UI, local CSV/aggregate handling, and interactive analytics. |
| `python-dotenv` | Optional convenience only if the app will load `.env` locally. Environment variables and Streamlit secrets can be used without it. |
| `google-genai` | Optional Gemini guidance integration; keep the app able to start and provide local fallback without it. |
| `torch` | Optional model stack for BiLSTM training/inference and PyTorch-backed BERT. Platform/CUDA wheel selection must match the deployment machine. |
| `transformers` | Optional tokenizer and BERT-family loading/fine-tuning. |
| `scikit-learn` | Optional training/evaluation utility for splitting and metrics; not required for Streamlit presentation or model forward passes. |

The dependency ranges have not been installed or resolved in this environment, so practical compatibility is **not verified**. Keep model and Gemini packages optional in the eventual app design. No additional `datasets` dependency is needed if the approved files are read with pandas or standard tooling.

## 8. Verified facts, assumptions, and unresolved decisions

### Verified from the reviewed sources and project files

- Phase 1 selected no dataset or pretrained checkpoint; it proposed multi-label outputs and a consent-based study dataset.
- EduAffect is described as 2,423 entries of student-focused text, based on the GoEmotions taxonomy, one dominant label per entry; its paper reports dataset availability at Kaggle under CC BY 4.0.
- GoEmotions has 58,009 English Reddit comments and 27 emotions plus Neutral. Its official README describes per-rater binary label columns and a filtered multi-label format.
- EmpatheticDialogues has 32 single situation labels, includes Confident, and its repository/dataset card identifies CC BY-NC 4.0.
- DAiSEE is video-based, has 9,068 snippets from 112 users, four ordinal affect labels, request-based access, and stated research-only restrictions.
- The two model repositories declare Apache-2.0 and identify their intended English checkpoints; the BERT-base model card reports 110M parameters.
- The project requirements separate the UI stack from the optional model stack; dependencies have not been installed or resolver-checked.

### Assumptions and recommendations

- The initial app will use English text. This aligns with the two proposed checkpoints, but project/user language has not been decided. EduAffect’s English–Twi code-switching is not automatically covered by an English-only checkpoint.
- Participant self-report is the recommended target-label source because it directly asks about the intended categories. It remains subjective and situational, not objective emotional ground truth.
- EduAffect is a promising domain comparator but must not be used as full five-label or mixed-emotion ground truth.

### Unresolved before implementation

1. Approve the primary data source: consented study-challenge collection, public EduAffect as an auxiliary resource, or another reviewed option.
2. Confirm research/ethics approval, participant age range, consent language, retention/deletion period, and whether de-identified student text can be retained for training.
3. Confirm the dataset terms for EduAffect/Kaggle and any third-party source restrictions before acquisition. Establish whether GoEmotions/EmpatheticDialogues licenses fit intended use; do not use MEED without written license/provenance clarification.
4. Finalize operational label definitions and whether an example can receive multiple self-reported labels. Decide handling for none, unsure, skipped, and ambiguous responses.
5. Confirm target language and intended deployment context, Gemini disclosure/consent, and whether the system is strictly educational/non-commercial.
6. Once implementation is approved, pin Python package versions and checkpoint revisions, resolve dependencies on the target environment, and inspect the actual CPU/GPU before training.

## Sources

- [GoEmotions official dataset README](https://github.com/google-research/google-research/blob/master/goemotions/README.md) · [GoEmotions paper](https://aclanthology.org/2020.acl-main.372/) · [GoEmotions model card](https://github.com/google-research/google-research/blob/master/goemotions/goemotions_model_card.pdf)
- [EduAffect paper and data availability statement](https://link.springer.com/article/10.1007/s44217-026-02098-1) · [EduAffect dataset DOI](https://doi.org/10.34740/kaggle/dsv/13091236)
- [EmpatheticDialogues repository](https://github.com/facebookresearch/EmpatheticDialogues) · [dataset card](https://huggingface.co/datasets/facebook/empathetic_dialogues) · [original paper](https://aclanthology.org/P19-1534/)
- [DAiSEE official page and terms](https://people.iith.ac.in/vineethnb/resources/daisee/) · [original paper](https://arxiv.org/abs/1609.01885)
- [MEED public repository](https://github.com/Nemo9303/MEED-dataset)
- [DistilBERT checkpoint](https://huggingface.co/distilbert/distilbert-base-uncased) · [configuration](https://huggingface.co/distilbert/distilbert-base-uncased/blob/main/config.json) · [paper](https://arxiv.org/abs/1910.01108)
- [BERT-base checkpoint](https://huggingface.co/google-bert/bert-base-uncased) · [paper](https://arxiv.org/abs/1810.04805)
