# Dataset Investigation: Next Steps

**Project:** Sameer's Emotion Detection & Learning Support Engine  
**Review date:** 2026-10-20  
**Scope:** Official GoEmotions documentation, local `data/raw` inventory, and documented partial-label dataset candidates. No datasets were downloaded and no existing project files were edited.

## Local GoEmotions files

The official [GoEmotions README](https://github.com/google-research/google-research/blob/master/goemotions/README.md) gives these raw-file URLs:

| File | Official URL | Local status at review |
|---|---|---|
| `goemotions_2.csv` | <https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv> | Not present |
| `goemotions_3.csv` | <https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv> | Not present |

The same README describes the raw release as three CSV files containing all rater annotations and comment metadata. Each row is one rater's annotation of one comment. The documented fields include `text`, `id`, `author`, `subreddit`, `link_id`, `parent_id`, `created_utc`, `rater_id`, `example_very_unclear`, and binary columns for each emotion category. This is the expected format for files 2 and 3; those files have not been downloaded or locally verified.

The read-only directory inventory contained `goemotions_1.csv` (14,174,600 bytes), `goemotions_1_retry.csv` (14,174,600 bytes), and `README.md` (238 bytes). It contained neither file 2 nor file 3. No files were changed.

## GoEmotions taxonomy and target coverage

The official README defines 27 emotion categories plus Neutral. Its categories are `admiration`, `amusement`, `anger`, `annoyance`, `approval`, `caring`, `confusion`, `curiosity`, `desire`, `disappointment`, `disapproval`, `disgust`, `embarrassment`, `excitement`, `fear`, `gratitude`, `grief`, `joy`, `love`, `nervousness`, `optimism`, `pride`, `realization`, `relief`, `remorse`, `sadness`, and `surprise`, plus `Neutral`.

The taxonomy contains no direct `boredom`/`bored`, `confident`/`confidence`, or `frustration`/`frustrated` category. The official README's three-file download instruction points to partitions of this same corpus and taxonomy, so files 2 and 3 are **not expected to add those missing categories**. The current audit found direct concept-level matches only for project **Confused** (`confusion`) and **Curious** (`curiosity`). Do not reinterpret `neutral`, `annoyance`, `anger`, `disappointment`, `optimism`, or `pride` as the missing targets. [Official README](https://github.com/google-research/google-research/blob/master/goemotions/README.md) · [Original paper](https://arxiv.org/abs/2005.00547)

## Documented candidates for missing labels

| Dataset | Potential direct labels | Format, access, and terms | Relevance and limitation |
|---|---|---|---|
| **DAiSEE** | Boredom and frustration (also confusion and engagement), each rated at four ordinal levels: very low, low, high, very high | The official dataset page describes 9,068 video snippets from 112 users. Access is by submitting the download form and accepting its terms. The page says use is for research only and the images must not be provided to third parties; no standard open-data license is identified on the page. [Official DAiSEE page](https://people.iith.ac.in/vineethnb/resources/daisee/) · [Original paper](https://arxiv.org/abs/1609.01885) | Learning-context labels for Bored and Frustrated are present, but the source is video/visual behavior, not text or learner self-report. It cannot directly train the current text-only classifier. Any use would require a modality change and review of the request terms. |
| **EmpatheticDialogues** | Exact category `Confident` appears in the original paper's data examples | The original paper describes about 25,000 crowdsourced, English, open-domain conversations grounded in one emotion-labeled situation. The official repository provides a download command for `empatheticdialogues.tar.gz`. Its repository license and dataset card identify CC BY-NC 4.0; noncommercial and attribution conditions apply. [Official repository and download instructions](https://github.com/facebookresearch/EmpatheticDialogues) · [Repository license](https://github.com/facebookresearch/EmpatheticDialogues/blob/main/LICENSE) · [Dataset card](https://huggingface.co/datasets/facebook/empathetic_dialogues) · [Original paper and `Confident` example](https://arxiv.org/html/1811.00207) | This is a direct text-label lead for Confident, but the label attaches to an open-domain life situation/dialogue, not a study challenge. It is single-situation supervision, not multi-label student self-report. Treat it only as a possible separately identified auxiliary source after confirming rights and intended noncommercial use. |
| **IEMOCAP** | Frustration is one of the categorical attributes | USC describes an English, acted, multimodal dyadic corpus with dialog transcriptions and at least three human annotators per utterance. Access requires a signed, nonexclusive, nontransferable first-party license. The release form limits use to internal research, prohibits commercial use and onward sharing, and asks users to consult USC about planned evaluation/reporting. [USC dataset information](https://sail.usc.edu/iemocap/iemocap_info.htm) · [USC release form](https://sail.usc.edu/iemocap/Data_Release_Form_IEMOCAP.pdf) | Transcripts could supply text paired with a frustration annotation, but the speech is acted and not about studying. The restrictive license and different domain make it an unsuitable primary training/evaluation source for this product without specific approval. |
| **Academic Boredom Survey: Junior High School Students in Indonesian Islamic Boarding Schools (N=642)** | Academic boredom is the directly studied construct | The article says item-level survey data are available from the [OSF record](https://doi.org/10.17605/OSF.IO/GUBCD). It describes 642 junior-high students and reports ethics approval and written guardian consent. The reviewed sources did not establish a separate license or reuse terms for the raw OSF data; the article's CC BY license applies to the article and is not assumed to license its data. [Original article and data/ethics statements](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2026.1812545/full) | Relevant for academic-boredom construct and survey/label-definition research, but it is structured survey response data, not free-text descriptions labeled with Bored. It does not directly train the current text model. Raw-data terms remain unverified, so do not acquire it until those terms are confirmed. |

These sources provide only partial, modality- or domain-mismatched leads. The reviewed sources do not establish a public dataset of **study-challenge text** with direct, suitable coverage of all five project labels.

## Recommended five-label data strategy

1. Preserve the product labels **Bored, Confident, Confused, Curious, and Frustrated**. Do not convert unrelated public labels into these targets.
2. After institutional/faculty and data-protection approvals, use the existing consent-based protocol to collect non-identifying study-challenge text with explicit multi-select self-reports for all five labels. Keep collection unauthorized until those approvals and consent/storage arrangements are complete.
3. If source rights and methods are approved, evaluate GoEmotions only as a separate auxiliary source for `confusion` and `curiosity`; do not merge it into the target-label evaluation set. EmpatheticDialogues may be considered separately for `Confident`, subject to CC BY-NC 4.0 and domain limitations. DAiSEE or IEMOCAP should be considered only if the project explicitly adopts and reviews a visual/audio modality; neither provides study-challenge text labels.
4. If approved five-label text is not available, keep the application in demonstration mode or formally revise the target labels. Do not claim a validated five-label classifier from partial public labels.

## Unresolved before acquisition

- Confirm intended use is compatible with EmpatheticDialogues' noncommercial license and complete its attribution obligations if selected.
- Obtain and review the DAiSEE terms before any request; confirm whether its image-only restrictions are compatible with any proposed use.
- Obtain USC's signed IEMOCAP license and consult USC before any proposed evaluation/reporting if that source is considered.
- Verify the OSF raw-data license and reuse terms separately from the article's CC BY license before considering the academic-boredom survey.
- Obtain human approval for the project's own consent-based collection before recruiting participants or retaining any student text.

No GoEmotions file 2 or 3 was acquired, no other dataset was downloaded, and no training or model evaluation was performed for this report.
