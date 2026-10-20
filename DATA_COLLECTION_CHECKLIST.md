# Data Collection Approval Checklist

**Status:** Pending. Collection is not authorized. Do not recruit participants or accept submissions until all applicable approvals are documented.

Record the approver, decision, date, conditions, and review/reference number for each applicable item. “Not applicable” must be justified by the responsible authority, not assumed by the project developer.

## Study and institutional review

- [ ] Faculty/project supervisor has approved the purpose, scope, participant population, recruitment plan, label definitions, and student burden.
- [ ] The appropriate institutional ethics/research review body has approved the protocol or documented that review is not required; retain its written determination and any conditions.
- [ ] If participants may be minors, the institution has approved age-appropriate safeguards, guardian permission, and participant assent. Otherwise, the study is restricted to adults.
- [ ] Participation is voluntary and clearly unrelated to grades, services, employment, or institutional standing; recruitment avoids pressure from teachers or supervisors.
- [ ] No personal or sensitive data are requested. The form warns against names, contact details, student IDs, course identifiers, medical/mental-health details, grades, or information identifying another person.

## Consent and participant rights

- [ ] Final consent wording has been reviewed for plain language and translated where needed.
- [ ] Consent explains purpose, data fields, self-report labels, training/evaluation use, access, storage, retention, deletion, and whether any aggregate reporting is planned.
- [ ] Consent explains voluntary participation, skipping, stopping, withdrawal route, exact withdrawal cutoff, and the limits of deletion after anonymization/model training.
- [ ] Participant contact/questions route is staffed and approved; no contact details are added to the model-data file.
- [ ] A separate consent record, if required, is approved for storage and deletion and is not joined to participant text in model files.

## Instrument and labeling

- [ ] Study-challenge prompt and five label definitions have been reviewed and approved.
- [ ] The form allows multiple labels and distinguishes “none,” “not sure/prefer not,” and skipped/stopped responses.
- [ ] The export preserves positive, explicit negative, and unknown status; rules for `known_labels` and excluded incomplete rows are documented.
- [ ] Language(s), eligibility, one/multiple responses per participant, and participant/source grouping strategy are approved.
- [ ] Any pilot or cognitive testing is approved and its records are either excluded from training or separately consented and documented.
- [ ] A label-quality review plan is approved, including aggregate coverage checks and separate handling of any independent annotator judgments.

## Data protection and storage

- [ ] Institutional privacy/data-protection contact has reviewed applicable laws, policies, student-data rules, consent, and whether opaque withdrawal/group tokens count as personal data.
- [ ] Data steward and authorized reviewers are named; least-privilege access is configured.
- [ ] Approved encrypted storage location is specified. Backups, synchronization, external processors, cross-border transfers, and incident response are reviewed.
- [ ] Intake, consent/withdrawal administration, model-ready files, demonstration fixtures, logs, and aggregate reports are separated as described in `DATA_COLLECTION_PROTOCOL.md`.
- [ ] Withdrawal-code design and deletion workflow are tested with empty/test-only fixtures, without participant data.
- [ ] Retention duration, deletion trigger, backup deletion process, model artifact handling after withdrawal, and deletion responsibility are documented.
- [ ] No participant text will be sent to Gemini or another external service under this protocol.

## Data rights, quality, and readiness

- [ ] The source and collection permissions cover the intended local processing, model training, evaluation, storage, and any proposed sharing of derived artifacts.
- [ ] The data input conversion follows `DATASET_INPUT_CONTRACT.md`: only approved columns, valid target labels, explicit mappings, and no raw text or identifiers in audit reports.
- [ ] A plan sets label-coverage/support minimums after an approved pilot/design review; no model is trained if a label lacks adequate known positives/negatives or independent-group support.
- [ ] Group-aware split, duplicate/leakage review, and source provenance manifest are approved.
- [ ] The team has confirmed the app stays in demonstration mode until training/evaluation and a separate integration approval.

## Final authorization

- [ ] All applicable approvals above are complete and conditions are incorporated into the final protocol and consent form.
- [ ] The responsible faculty/institutional authority has explicitly authorized the start date and recruitment/collection process.

**Authorized start date:** ____________________  
**Responsible approver:** ____________________  
**Approval/reference ID:** ____________________
