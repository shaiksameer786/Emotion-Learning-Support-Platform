# Dataset Source and Rights Follow-up

**Project:** Emotion Detection & Learning Support Engine  
**Review date:** 2026-10-21  
**Scope:** Targeted review of the 2026 ICSE SEET paper's data availability trail and the existing GoEmotions release. No data were downloaded and no existing report or source file was changed.

## 2026 ICSE SEET paper and deposit

The accessible author-hosted paper is [“Learning Programming in Informal Spaces: Using Emotion as a Lens to Understand Novice Struggles on r/learnprogramming”](https://imranraad07.github.io/posts/documents/papers/ICSE_SEET_2026.pdf). In its data-availability note, it directs readers to [Zenodo record 17145615](https://zenodo.org/records/17145615) for the study datasets, scripts, and annotation instructions. The paper is listed as SEET@ICSE 2026, pages 13–24, DOI [10.1145/3786580.3786946](https://doi.org/10.1145/3786580.3786946).

The accessible paper text did not expose a separate supplementary-material download or another data-repository URL; the Zenodo record is the exact deposit identified by the paper. The paper describes a multi-label Learning-Centered Emotions scheme including Boredom, Confusion, Curiosity, and Frustration. It does not list Confident as one of those eight emotion labels. This verifies what the paper says about its taxonomy, **not** the contents or license of the deposit. The paper's reported record and label counts are not treated as evidence that the records are retrievable.

### Deposit access and terms status

- Direct attempts to open the Zenodo record and its record API returned “URL … is not accessible via this tool.” Exact attempted URLs: `https://zenodo.org/records/17145615` and `https://zenodo.org/api/records/17145615`.
- Search for the exact record identifier and DOI did not return a repository file listing or license statement. The ACM DOI page also returned HTTP 403 in the available research tool.
- Consequently, the deposit's current publication state, downloadable filenames, schema, data files, checksum, and dataset-specific license/reuse conditions could not be verified from the repository itself.
- The paper's article license/copyright statement, if any, would not by itself establish a license for its separately deposited Reddit-derived records. No repository terms are inferred from the paper.
- **No files were downloaded.** This is an access/verification blocker in this review, not proof that Zenodo is unavailable to all users. Because reuse terms are unverified, the source is not cleared for project use and source-specific pursuit stops here.

## GoEmotions: files and reuse status

The project currently has `data/raw/goemotions_1.csv` and `data/raw/goemotions_1_retry.csv` (both 14,174,600 bytes), plus `data/raw/README.md`. The two CSVs are duplicate copies and must not be counted as separate data. The existing `DATASET_AUDIT.md` records that this part has 70,000 annotation rows; it reports direct source-label matches only for `confusion` and `curiosity`. Neither source contains a direct label for Bored, Confident, or Frustrated. No GoEmotions examples were copied into a prepared training CSV during this review; `data/prepared/emotion_dataset_initial.csv` remains header-only.

### Terms evidence and limitation

- The [official GoEmotions README](https://github.com/google-research/google-research/blob/master/goemotions/README.md) identifies the data as Reddit comments and describes the raw files as containing comment text and metadata. It does not state a separate license for the underlying comment text.
- The [Google Research Datasets GoEmotions card](https://huggingface.co/datasets/google-research-datasets/go_emotions) displays `apache-2.0` and says the GitHub repository housing the dataset has an Apache 2.0 license. The [Google Research repository LICENSE](https://github.com/google-research/google-research/blob/master/LICENSE) is Apache-2.0. This is evidence of the publisher/repository license statement; it does not resolve third-party rights in Reddit users' comments.
- Reddit's current [Data API Terms](https://redditinc.com/policies/data-api-terms), last revised July 20, 2026, state that User Content remains owned by users, limit the granted license, and prohibit use of User Content to train a machine-learning or AI model without express permission from applicable rightsholders. The terms' application to this previously published, third-party archival dataset requires human/legal review; this report does not decide that question.
- The GoEmotions card also notes that raw data contain original Reddit usernames. The local CSV therefore requires restricted handling and must not be treated as de-identified merely because it is a research dataset.

**Decision:** Do not treat GoEmotions as approved model-training data based solely on the Apache-2.0 repository/card metadata. The project has no documented rightsholder permission or legal review resolving the underlying Reddit-content terms. Keep it out of the training pipeline pending that review. Its direct target coverage remains partial (Confused/`confusion`, Curious/`curiosity` only) regardless of licensing.

## Verified findings versus unknowns

### Verified from accessible sources or local files

- The paper points to Zenodo record `17145615`; its described LCE taxonomy includes Boredom, Confusion, Curiosity, and Frustration and omits Confident.
- The GoEmotions publisher card displays Apache-2.0; the Google Research repository has an Apache-2.0 LICENSE; the GoEmotions README describes Reddit-derived comment text and metadata.
- Current Reddit Data API Terms contain an express-permission condition for model training with User Content. The GoEmotions audit and local inventory identify the two duplicate part-1 CSV files and a README.
- No source files were downloaded, altered, or copied; no participant data, synthetic data, training, or model evaluation was involved.

### Not verified

- Whether Zenodo record 17145615 is currently public/downloadable, what exact files it contains, the actual label values in those files, checksums, and its dataset-specific license or reuse restrictions.
- Whether a rightsholder permission or other agreement applicable to the GoEmotions Reddit comments authorizes this project's model-training use under current terms.
- Whether either public Reddit-derived source is ethically or legally appropriate for this student-support project beyond what the cited publication and terms state.

## Next concrete action

Have the project owner/institutional reviewer resolve the written reuse terms for GoEmotions and the Zenodo deposit before any files are acquired for training. For the 2026 deposit, the reviewer must inspect record 17145615 directly and capture its file manifest and license/terms; if that cannot be accessed or the terms do not grant the needed reuse, close this source and use the already documented consent-based collection route after the required approvals. No dataset is currently cleared for training, and no source establishes all five target labels in a suitable, approved text dataset.
