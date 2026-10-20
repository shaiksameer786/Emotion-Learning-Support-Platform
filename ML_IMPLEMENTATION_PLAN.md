# Proposed ML Implementation Plan

**Project:** Emotion Detection & Learning Support Engine  
**Document status:** Proposal only. This document does not approve data collection, dataset use, dependency installation, training, or deployment.

## 1. Evidence and current state

The source reports describe a GoEmotions part-one CSV with 70,000 annotation rows. Its taxonomy has direct labels for **Confused** (`confusion`) and **Curious** (`curiosity`) among the five project targets. It does not directly label **Bored**, **Confident**, or **Frustrated**. The raw material is English Reddit text with per-rater annotations, rather than study-challenge text. The reports do not establish that the Reddit text itself may be reused for this project. See [DATASET_AUDIT.md](DATASET_AUDIT.md).

The remaining GoEmotions parts are expected to use the same emotion taxonomy and therefore are not expected to fill those three gaps. The research report lists other possible sources, but their modality, domain, label meaning, access terms, or reuse rights make them unsuitable as an automatic substitute for five-label student study text. See [DATASET_NEXT_STEPS.md](DATASET_NEXT_STEPS.md). Those reports are the evidence summarized here; no new source verification is claimed by this proposal.

The project already has a five-label prediction schema and demonstration classifier. Its training infrastructure includes group-aware splitting when an opaque `group_id` is supplied. Project status says persisted split files do not retain group IDs, so participant separation cannot be rechecked from those files after reload. The real-data training path has not been validated on an approved dataset. The app must remain in demonstration mode until the gates in this plan are met.

## 2. Proposed five-class labeling protocol

### Unit and label source

Use one response about one study challenge at a defined, participant-understood time point. The primary label source should be the participant's own multi-select report about that response. Treat labels as subjective, context-dependent reports, not objective measurements of an internal state. Do not infer labels from keywords, writing style, or general positive/negative sentiment.

The current collection protocol's proposed definitions are a starting draft for human review:

| Target label | Draft operational meaning |
|---|---|
| **Bored** | The study activity feels uninteresting or unstimulating at that moment. Shortness or low engagement alone is insufficient. |
| **Confident** | The participant feels capable of understanding or handling this study task. General positivity alone is insufficient. |
| **Confused** | The participant is unsure what a concept, instruction, or next step means. It may co-occur with other labels. |
| **Curious** | The participant wants to explore, ask about, or learn more about something in the task. It may co-occur with confusion. |
| **Frustrated** | The participant feels blocked by an obstacle while trying to make progress. Difficulty alone is insufficient. |

These definitions and the final prompt/time framing require faculty or other appropriate review before collection. The form should offer five independent selections, not a forced single choice, and distinguish:

- Selected positive labels.
- **None of these apply**, meaning all five were explicitly assessed and negative.
- **Not sure / prefer not to label** and **skip/stop**, which mean unknown or absent judgments, not negative labels.

Use the existing input contract's `labels` for positives and `known_labels` for categories actually assessed. An explicit “none” has no positive labels and all five labels known. Partial answers must preserve unknown categories. Exclude records that cannot be represented without turning unknowns into negatives. Multiple positive labels are valid only when the participant actually selected them.

Independent annotators may be used in an approved quality study, but their judgments must be stored and analyzed separately from participant self-reports. Do not silently adjudicate one source into the other. If used, approve an annotation guide, reviewer training, double-review subset, agreement measure, and disagreement resolution process first.

## 3. Consent-based collection and privacy

**Assumption for planning:** a limited, voluntary collection of non-identifying study-challenge text may be considered. **Approval status:** not established; the existing collection protocol and checklist say collection is not authorized.

Before recruitment or acceptance of any text, obtain all approvals applicable to the institution and population. At minimum, the responsible faculty/project supervisor and institutional ethics/research authority (or its written determination that review is not required), plus the privacy/data-protection contact, must approve the final protocol. If minors could participate, obtain a separately reviewed child-participant protocol, guardian permission, and assent; otherwise restrict participation to adults. Final consent wording, label instrument, withdrawal cutoff, retention schedule, and storage arrangements require human approval.

The approved consent process should be voluntary, allow skipping/stopping without penalty, explain that text and self-reports may be used for local model training and evaluation, and state who can access records, how long they are retained, how withdrawal works, and its limits after model training. Warn participants not to enter names, contact details, IDs, medical information, grades, or identifying details about themselves or others. Screen submissions; quarantine or delete unsafe text rather than copying it into reports or logs.

Keep consent/withdrawal administration separate from text and labels. Use institution-approved encrypted storage and least-privilege access. A random withdrawal code or opaque participant grouping token may still be linkable data and needs privacy review. Define secure deletion for intake, prepared files, backups, and affected checkpoints before collection. Keep participant data out of source control, demo fixtures, analytics, issue trackers, Gemini, and other external services under this proposal. Audit reports should contain aggregate counts only, never raw text or participant/group identifiers.

The existing draft proposes a retention schedule for review, but it is not an approved policy. The responsible authority must set the actual retention and deletion dates and decide how withdrawal after training is handled. No collection or retention duration is authorized by this plan.

## 4. Dataset readiness and quality controls

Use only a source whose license/terms or consent explicitly cover the intended local processing, training, evaluation, storage, and any proposed sharing of derived artifacts. Record provenance and approval references. Do not map unrelated GoEmotions labels such as joy, sadness, anger, fear, or neutral to a target category. If GoEmotions is later approved for auxiliary research, keep it separate from the primary student-text dataset and do not use it to claim study-domain or five-label coverage.

Before any fit, produce a text-free audit covering schema, record counts, missing/unknown judgments, per-label positive and known-negative counts, label co-occurrence, language, source/version, and validation exclusions. Apply these checks:

1. Validate the existing contract (`text`, `labels`, optional `known_labels`, and optional `group_id`); reject unsupported labels and undocumented mappings.
2. Reject blank text, malformed label arrays, and rows with no usable known target labels. Keep raw text out of error reports.
3. Screen for accidental personal or sensitive information under the approved review process. Automated screening alone is not proof of de-identification.
4. Detect exact duplicates after conservative normalization. Keep one only when text, positive labels, known-label mask, and grouping agree; quarantine conflicting duplicates for approved review. Review near-duplicates and repeated prompts/templates because they can leak across splits. Do not merge labels automatically.
5. Report class balance by both responses and independent participant groups, plus known-positive/known-negative support and multi-label co-occurrence. Do not solve imbalance by copying records across partitions. Any class weighting or training-only resampling must be documented and applied only inside the training partition.
6. Do not use demo fixtures, unit-test fixtures, or unapproved public text as student training observations.

Do not invent a universal minimum number of examples. Before training, the supervisor/research reviewer and model owner must define a data-sufficiency rule from the intended population, label prevalence, number of independent participants, uncertainty desired, and whether every label can be evaluated. The existing trainer requires known positives and negatives for each label in train and validation; if that or an agreed support criterion fails, defer training or seek an approved data-design revision rather than weaken the checks.

### Participant-separated splits and leakage prevention

Require a stable, opaque grouping token for participant responses when repeated responses are possible. Keep it out of model features and public reports. Split by participant/group so every response from one participant belongs wholly to train, validation, or test. The current splitter's 70/15/15 ratios and seed 42 are existing defaults, not a guarantee of label support or an approved study design; retain them only if a reviewed split can meet the coverage requirements, and record the actual counts and seed.

The split must also group identical/near-duplicate source items where appropriate, deduplicate before assignment, and verify exact-text and group disjointness. Fit vocabulary, text transforms, class weights, and all model parameters from training data only. Validation data is for model selection and threshold selection. Freeze the selected model/configuration/thresholds before the held-out test is opened.

**Infrastructure gap to resolve before a real-data run:** the current project status says persisted split files omit group IDs and reload cannot reconstruct group membership. For genuine participant-separated evaluation, either maintain a restricted, approved sidecar with opaque group tokens excluded from model inputs, or strengthen the split-manifest process to verify group-disjoint assignment at split creation and preserve privacy-safe evidence sufficient for audit. The chosen approach must be reviewed for linkability and deletion. A manifest fingerprint alone does not prove participant separation or data provenance.

## 5. Model sequence

### First: BiLSTM baseline

Use the existing five-output BiLSTM infrastructure as the initial neural baseline, after authorized data passes readiness checks. Fit its vocabulary on training text only; freeze vocabulary before transforming validation/test. Use padding/truncation and five-output logits in the canonical order: Bored, Confident, Confused, Curious, Frustrated. Train as a multi-label task with sigmoid/BCE-with-logits behavior and mask unknown labels so unassessed categories do not become false negatives. Select epoch and per-label decision thresholds on validation only. A fixed seed supports reproducibility but does not guarantee identical results across hardware/library versions.

Before the data run, perform the already documented offline CPU smoke check and verify that installed optional dependencies and checkpoint serialization work in the actual environment. This smoke check is structural only; it is not training evidence or a performance result. CPU-only development is suitable for data checks and likely for small experiments, but recurrent model training may be slow. Hardware and dataset size must be measured before promising a practical run time.

### Optional second: BERT-family comparison

Do not select or fetch a checkpoint from this proposal. BERT is optional and should proceed only if the approved dataset has adequate independent-participant and per-label support, the language/domain fit is defensible, and available memory and compute can support fine-tuning. Before checkpoint acquisition, approve the exact model ID, pinned revision, tokenizer, license/terms, language fit, input-length policy, and resource requirements. Those details remain unresolved; no pretrained weights are asserted to exist locally.

If justified, fine-tune a five-output BERT-family head on the same approved training partition and known-label mask as the BiLSTM. Use the checkpoint's matching tokenizer, truncate according to a predeclared maximum length, and keep validation/test partitions unchanged. Compare models on the same records, label order, split manifest, validation threshold-selection procedure, and final metrics. Document the compute/tuning budget so one model does not receive hidden extra test feedback. If dataset support or hardware is inadequate, record the reason and defer BERT.

## 6. Evaluation and comparison

Use multi-label evaluation and publish per-label support alongside:

- Precision, recall, and F1 for each of the five labels.
- Macro F1 and micro F1, with the averaging convention documented.
- A per-label 2-by-2 confusion matrix (TP/FP/FN/TN). A single ordinary multiclass confusion matrix is not sufficient for multi-label predictions. A label co-occurrence summary may be supplemental.
- Optionally, subset accuracy and Hamming loss to show exact-set and per-label error behavior; do not let one aggregate metric hide a weak class.

Select each label's threshold on validation data only using a predeclared rule (for example, validation F1 subject to an approved operating objective). Record the rule, tie behavior, selected thresholds, known-label masks, and undefined cases. If a partition has no known positives or negatives for a label, mark the corresponding metric undefined and do not imply successful evaluation. Thresholds and model choice are frozen before one held-out test evaluation. Do not use test results to retune, select checkpoints, or rewrite mappings. Report the split design, group counts, per-label support, limitations, and uncertainty; do not invent numeric pass targets. Stakeholders must approve any performance acceptance thresholds before seeing test results.

## 7. Eventual application integration

Keep the current Streamlit demonstration behavior until an approved, reproducible model passes data, evaluation, and integration review. A future adapter should implement the existing classifier interface and return the shared prediction schema with canonical label order, model name/version, thresholds, and `model_inference` mode. It should load only a checkpoint whose metadata matches the approved dataset reference, split-manifest fingerprint, model configuration, and label order.

Do not show an unsupported category as a model prediction. A model is not eligible for integration if any of the five categories lacks reviewed target labels and adequate evaluation support. If the project owner chooses a reduced label set, that requires a deliberate, versioned schema and UI change; do not fill missing scores with keyword matches, arbitrary constants, or labels from unrelated datasets. Preserve a no-clear-category outcome. Display scores as model estimates, not calibrated probabilities unless calibration is separately assessed; keep rule/keyword enhancement visibly separate from model inference. Do not claim that predictions reveal a learner's true internal state, diagnose a condition, or demonstrate learning benefit.

Text should remain local under the approved privacy design. Logging stays disabled by default; if later enabled, obtain consent and record only the minimum approved non-text fields. Gemini is outside this training proposal; no learner text should be sent to it without a separate privacy, terms, and consent review.

## 8. Risks and mitigations

| Risk | Planned response |
|---|---|
| No source directly covers all five labels in the target text domain | Treat approved self-report collection as the likely primary path; do not force mappings. Keep auxiliary corpora separate and document domain/label limits. |
| Collection may be disallowed, delayed, or unsuitable for the population | Obtain written institutional/privacy decisions first; keep the app in demo mode until then. |
| Self-reports are subjective and time-dependent | Use a reviewed prompt and definitions; preserve “unknown,” report label-source limits, and avoid claims about true internal states. |
| Sparse or imbalanced classes and few independent participants | Audit group-level support, define adequacy before training, defer unsupported labels, and report uncertainty rather than overstate metrics. |
| Duplicate language or multiple rows from one participant leak across splits | Deduplicate/review near-duplicates and group all participant records before any fit; audit disjointness. |
| Persisted splits currently cannot revalidate participant identity separation | Resolve the restricted sidecar/manifest evidence design before real evaluation. |
| Accidental identifying or sensitive text is retained | Minimize collection, screen and quarantine, restrict/encrypt storage, define deletion and withdrawal steps, and avoid raw-text logs/services. |
| BERT checkpoint rights, fit, or hardware are unsuitable | Select only after exact checkpoint review; pin revision; defer if language, licensing, or resources do not fit. |
| Test-set feedback inflates reported performance | Keep test held out; finalize model and validation thresholds first; run the test evaluation once for the approved comparison. |

## 9. Decisions requiring human approval

Before collection or training, the project owner/supervisor and applicable institutional reviewers must resolve:

1. Whether collection is permitted, participant eligibility/age, recruitment, consent wording, withdrawal deadline, privacy contact, storage, access, retention, and deletion—including backup and post-training withdrawal handling.
2. Final five label definitions, response time framing, treatment of none/unknown/skipped responses, and whether independent annotation is in scope.
3. Primary dataset/source and allowed use. In particular, whether any public auxiliary corpus may be used after source-specific rights review; current reports do not establish GoEmotions text reuse rights.
4. How opaque participant grouping evidence will be preserved through split creation and persisted training records without exposing identities.
5. Data sufficiency and minimum per-label/per-group coverage rules, plus predeclared evaluation objectives and stakeholder-approved acceptance thresholds. No numeric values are set here.
6. Whether BERT is justified; if yes, its exact checkpoint, pinned revision, tokenizer, language/license review, and available CPU/GPU/memory budget.
7. Whether an eventual app may use the model, what limited logging is permitted, and whether any future Gemini use is separately approved.

## 10. Acceptance criteria before implementation/training

Proceed to a real-data implementation only when all applicable items are satisfied and documented:

- Required faculty/institutional, ethics/research, and privacy/data-protection decisions are recorded; the collection protocol, consent text, data instrument, storage, retention, and withdrawal process are approved before recruitment.
- A legally and ethically usable source exists for all five target labels in an appropriate text domain, or the project owner formally approves a versioned scope change. No unsupported category is synthesized by mapping.
- The input contract explicitly represents positive, known-negative, and unknown label states and contains an approved non-identifying participant grouping mechanism when needed.
- A privacy-reviewed group-aware split strategy is implemented and auditable across persisted files; train/validation/test separation and duplicate safeguards pass automated checks.
- Text-free audit reports show usable known positive and negative support for each label in the intended partitions under the approved adequacy rule. If any category fails, training/integration is deferred.
- The model/data approval gate references real human review evidence; demo and test-only fixtures remain excluded.
- The baseline's offline smoke, training/evaluation interfaces, metrics, threshold selection, checkpoint metadata, and load checks run in the chosen environment. Failures are reported; no mocked result is presented as real model execution.
- The evaluation protocol and performance acceptance criteria are fixed before final test evaluation. The test partition is not used for model selection.
- App integration is separately reviewed; output labels are supported, scores are described correctly, privacy safeguards remain effective, and the UI remains in demonstration mode until acceptance is complete.

**Next step:** obtain the applicable human decisions in Section 9, beginning with whether a consent-based five-label study-text collection is permitted and who will approve its protocol. Until that decision is documented, do not collect data or train a model.
