# Study-Challenge Data Collection Protocol

**Project:** AI-Driven Emotion Detection & Personalized Learning Support Platform  
**Protocol status:** Draft for faculty/institutional, ethics, and data-protection review. **Collection is not authorized. Do not recruit participants or collect submissions under this draft.**  
**Data status:** No participant data has been collected. The app remains in demonstration mode.

## 1. Purpose and intended use

The proposed study would gather short descriptions of study-related challenges together with participants’ own, contemporaneous reports about whether one or more of five categories apply: **Bored**, **Confident**, **Confused**, **Curious**, and **Frustrated**. The intended research use is to prepare and, only after separate approval, train and evaluate local text-classification baselines for an educational-support prototype.

The categories are labels reported by a participant about a particular text and moment. They are not objective measurements of an internal state, psychological assessments, diagnoses, or grounds for decisions about a learner. Data must not be used to grade, discipline, rank, monitor, or make eligibility decisions about participants. No secondary use, external sharing, or Gemini submission is included in this proposal; any such use would require separate review and consent.

## 2. Participants and voluntary informed consent

The participant population, recruitment method, institution, and responsible data steward must be approved before any approach to participants. The recommended initial scope is adults only. Do not include minors unless the institution approves a separate child-participant protocol, guardian permission, age-appropriate assent, and additional safeguards.

Participation must be voluntary, with no effect on grades, services, employment, or relationship with a teacher or institution. Participants may decline, skip any item, or stop without penalty. Consent must be obtained before a description is accepted. The final notice must be in language participants understand and reviewed by the relevant faculty/ethics authority and privacy contact.

The approved consent notice must explain, in plain language:

- Who is responsible for the study and how participants can ask questions or request withdrawal.
- The purpose, what information is requested, and that the text and self-reported labels may be used to create training and evaluation files for this project.
- That the participant must not enter identifying or sensitive information, and that submissions will be screened for accidental disclosures before retained use.
- That labels are self-reports and model outputs are experimental estimates, not facts, diagnoses, or assessments of the participant.
- Which people can access the data, where it is stored, how long it is retained, and how it will be deleted.
- How withdrawal works, the deadline or model-training cutoff, and any practical limits after model training. No model should be trained until this limitation is explicit and approved.
- Whether any de-identified aggregate results may be reported. No raw descriptions will be published or shared under this protocol.

Consent records, if required, must be stored separately from descriptions and labels. Do not place names, contact details, signatures, or institutional identifiers in the model-data files. If an institution requires a consent register, its access, retention, and separation from responses must be approved first.

## 3. Proposed description and self-report procedure

After consent, request one concise description of a current or recent study-related challenge. The form should ask for task context needed to understand the challenge, not personal history. Display the privacy instruction before the text field: **Do not include names, contact details, student IDs, course/section identifiers, medical or mental-health information, grades, or details that could identify you or another person.** Provide an option to skip the text field or leave the study.

After the description, present five independent checkboxes with an option to choose more than one. Use the following draft definitions for review; they are not final until faculty/ethics approval:

| Label | Proposed participant-facing meaning |
|---|---|
| **Bored** | The study activity feels uninteresting or unstimulating at this moment. A short response, tiredness, or low engagement alone does not establish this label. |
| **Confident** | The participant feels capable of understanding or handling this study task. A generally positive mood alone does not establish confidence about the task. |
| **Confused** | The participant is unsure what a concept, instruction, or next step means. Confusion can occur with or without frustration. |
| **Curious** | The participant wants to explore, ask about, or learn more about something in the task. Curiosity can occur with or without confusion. |
| **Frustrated** | The participant feels blocked or frustrated by an obstacle while trying to make progress. A task being difficult alone does not establish this label. |

Use distinct responses for:

- **None of these apply** — only when the participant explicitly indicates all five do not apply.
- **Not sure / prefer not to label** — applicable labels are unknown, not negative.
- **Skip / stop** — no label judgment is inferred.

Do not force a single choice. Do not infer missing labels from the description, keywords, or another selected label. Preserve an explicit positive, explicit negative, or unknown state for each category. If all five categories were answered, the prepared contract file may mark all five as known. If answers are partial, use the contract’s `known_labels` field to preserve only assessed categories. A fully skipped or uncertain record must not be converted to five negatives; exclude it from model-ready data unless a reviewed, valid known-label representation exists.

The collection wording, definitions, and whether a participant answers “right now” or for a bounded recent period must be fixed and approved before collection. Do not change the instrument mid-collection without documenting and reviewing the change.

## 4. Privacy screening and withdrawal

Do not ask for personal identifiers or sensitive details. Before retaining a record, use a privacy screening step that checks for names, handles, contact details, student/course IDs, health information, grades, and identifying details about third parties. Automated detection, if later added, is only a screen and does not guarantee de-identification. Quarantine a questionable submission; redact only if an approved reviewer can do so safely and without changing the meaning of the response. Otherwise delete it. Do not copy questionable text into audit reports, issue trackers, or logs.

Provide a withdrawal route that does not require a participant to reveal their identity to the project team. The proposed method is a random withdrawal code shown once to the participant and stored in a restricted, encrypted lookup index separate from text and labels. Keep no name-to-code key. Treat the code/index as potentially linkable data and obtain privacy approval for it. Explain that a lost code may make it impossible to locate a response. If the institution requires another withdrawal mechanism, use only the approved method.

Before the approved withdrawal cutoff, a valid request should remove the matching response from the intake file, review copy, prepared dataset, split files, and any model artifacts trained with it; if a model has already been trained, pause use and retrain without that record when required and feasible. Record only that a deletion was completed and its date, not the removed text. The consent notice must state the exact cutoff and accurately describe limits once an anonymized aggregate or model has been produced. No cutoff is set by this draft; approval is required before collection.

## 5. Storage, access, retention, and deletion

Use an institution-approved, encrypted storage location with access limited to the approved data steward and named reviewers who need it. Do not store participant text in the source repository, personal cloud drives, email, chat, notebooks, screenshots, analytics, or application logs. Do not transmit the text to Gemini or other external services. Backups, synchronization, processors, and cross-border storage must be reviewed before use.

Keep these materials separated:

1. **Consent/withdrawal administration:** consent evidence or withdrawal-code index, if required, in a separately restricted location.
2. **Intake records:** participant text and label selections in an approved restricted staging location. Screen for accidental identifiers before transfer.
3. **Model-ready data:** only approved, privacy-screened text, labels, `known_labels` when needed, and an opaque non-identifying `group_id` if grouping is necessary. Store locally in ignored data directories with restricted access; never commit it.
4. **Demonstration fixtures:** existing explicitly labelled demo examples remain separate from all participant and training data. They are not observations and must never enter a dataset or evaluation split.
5. **Reports and logs:** counts and quality summaries only; no raw text, identifiers, withdrawal codes, or row-level details that could identify someone.

Before collection begins, the approving body must set and publish the retention period. Proposed default for review: delete intake copies and withdrawal linkage as soon as withdrawal eligibility ends and the approved model-ready data are checked; delete model-ready text and derived artifacts no later than 12 months after the final approved evaluation, unless a documented institutional requirement mandates a different period. Delete sooner when consent is withdrawn or approval is revoked, subject to the approved model retraining/deletion procedure. Secure deletion must include working copies and backups according to the storage provider’s documented deletion process. This schedule is a proposal, not an approved active retention policy.

## 6. Label consistency and data coverage review

Before collecting the full set, obtain approval for the label definitions and test the form’s clarity through an approved review process. Do not use test responses as training examples unless separately consented and approved. Review label counts, missing/unknown rates, “none” responses, co-occurrence, language, duplicates, and privacy-screening outcomes using aggregate reports without text.

Before model training, require:

- A documented label protocol and a traceable source/provenance record for every prepared file.
- Known positives and explicit negatives for each target label, with adequate independent participant/group support in train, validation, and test. Predefine numeric minimums after an approved pilot/design review; this protocol does not invent sample-size thresholds.
- Actual co-occurring labels if the project intends to evaluate mixed-emotion predictions. A single-label dataset cannot establish this capability.
- A group-aware split using a non-identifying participant or source token where repeated records exist. Confirm all records from one participant remain in one split. Do not claim participant-level separation if suitable tokens are unavailable.
- A review of class imbalance, near-duplicate/leakage risks, and per-label support. Freeze the split before model comparisons.
- If independent human annotation is added, separate it from self-report, define the annotator instructions, double-label an approved subset, document agreement and adjudication, and retain the distinction in all reports.

If a category lacks reliable coverage, do not fabricate or infer it from another label. Keep the five-label interface in demonstration mode and report that the category is not evaluated.

## 7. Contract template and data separation

[`data/templates/study_challenge_template.csv`](data/templates/study_challenge_template.csv) is a blank, header-only template containing the input contract’s two required columns: `text` and `labels`. It contains no sample records. The contract accepts UTF-8 CSV and requires `labels` to be a JSON array of target names.

This minimal template is not a complete consent form or collection instrument. The protocol’s tri-state answers and participant grouping may require the optional `known_labels` and `group_id` fields in a reviewed conversion/export. Use `known_labels` to preserve explicit negatives and unknown categories; use `group_id` only if the approved design requires repeat records per participant and its privacy handling is approved. Do not assume that absent labels mean negative. Follow `DATASET_INPUT_CONTRACT.md` and its no-default-mapping rule when preparing any data.

## 8. Interpretation and reporting limitations

Self-reports are subjective, context- and time-dependent, and may be affected by wording, interpretation, memory, culture, language, and willingness to disclose. A participant may report more than one category or none; different people may use the same words differently. A self-report is the study label for that response, not proof of a person’s internal state. Model scores are estimates from text and must not be presented as diagnoses or validated emotional facts.

Reports must name the source and label method (self-report or separate annotator), describe consent and privacy limits, language/population coverage, missingness and class coverage, split method, and per-label uncertainty. Do not generalize beyond the sampled population or imply clinical or educational outcome benefit without evidence. The current app continues to show demonstration output until a real model is separately trained, evaluated, and approved for integration.

## 9. Approval gate

Use [`DATA_COLLECTION_CHECKLIST.md`](DATA_COLLECTION_CHECKLIST.md) to record the required human approvals. Collection begins only after every applicable item is signed off and the consent notice, storage arrangement, withdrawal deadline, retention schedule, and data instrument are finalized. This document and the blank CSV template do not authorize collection.
