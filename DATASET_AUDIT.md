# GoEmotions Dataset Audit — Part 1

**Project:** AI-Driven Emotion Detection & Personalized Learning Support Platform  
**Audit date:** 2026-10-20  
**Scope:** Read-only inspection of `data/raw/goemotions_1.csv` only. No other GoEmotions files were downloaded or audited.

## Source and provenance

- Official dataset repository and documentation: [Google Research GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions)
- Exact official download URL for this part: [goemotions_1.csv](https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv)
- Original paper: [GoEmotions: A Dataset of Fine-Grained Emotions](https://arxiv.org/abs/2005.00547)
- Repository license: [Google Research repository LICENSE](https://github.com/google-research/google-research/blob/master/LICENSE) displays Apache-2.0. The GoEmotions README does not specify a separate data license for the underlying Reddit comments. The repository license is not treated here as confirmation of rights to reuse or redistribute third-party comment text; rights review remains necessary.

The official README describes the raw files as one row per rater annotation per comment, with binary emotion columns and comment/rater metadata. It also notes that GoEmotions contains English Reddit comments rather than study-challenge reports.

## Local file verification

| Check | Result |
|---|---:|
| Local path | `data/raw/goemotions_1.csv` |
| Local size | 14,174,600 bytes — verified exact match |
| Local MD5 | `8803568F912B257630AA623AFD633CD0` |
| Server Content-Length | 14,174,600 bytes, as supplied for this audit |
| Server checksum | Not retrieved or compared; no server-side checksum verification is claimed |

## Method

The file was streamed with Python 3.11's built-in `csv` module using strict UTF-8 decoding. Counts below are from this file only. Missing values mean blank or absent CSV cells. Duplicate text is exact, unnormalized text; the duplicate-row count is the number of occurrences after the first occurrence of each distinct nonempty text. Emotion counts are per annotation row; distinct comment-ID counts indicate IDs with at least one positive annotation in this file. Label cardinality counts positive emotion columns within each annotation row. No comment text, Reddit username, or ID value was emitted into command output or this report.

For encoding inspection, 25 deterministic pseudorandom nonempty text samples were checked in memory for the Unicode replacement character and common mojibake sequences. The full text column was also scanned for those markers. Sample contents were not printed.

## File shape and missing values

- **Data rows, excluding header:** 70,000
- **Columns:** 37
- **Missing values:** 0 in every column, including `text`, metadata, `example_very_unclear`, and all 28 emotion columns.
- **Rows with extra CSV fields:** 0

Columns, in source order:

`text`, `id`, `author`, `subreddit`, `link_id`, `parent_id`, `created_utc`, `rater_id`, `example_very_unclear`, `admiration`, `amusement`, `anger`, `annoyance`, `approval`, `caring`, `confusion`, `curiosity`, `desire`, `disappointment`, `disapproval`, `disgust`, `embarrassment`, `excitement`, `fear`, `gratitude`, `grief`, `joy`, `love`, `nervousness`, `optimism`, `pride`, `realization`, `relief`, `remorse`, `sadness`, `surprise`, `neutral`.

## Duplicate text and annotation flags

- **Distinct nonempty exact text values:** 43,182
- **Distinct text values occurring more than once:** 19,794
- **Duplicate text rows beyond each value's first occurrence:** 26,818
- **Rows with `example_very_unclear=True`:** 1,129
- **Other nonempty values in `example_very_unclear`:** none

Text repetition is expected in a per-rater file because multiple annotation rows can refer to the same comment; it does not by itself establish duplicate source examples or leakage.

## Emotion label validation and counts

All **1,960,000** cells across the 28 emotion columns contained only `0` or `1`; there were no missing or invalid emotion values. The table reports positive annotation rows and, separately, distinct comment IDs having at least one positive annotation in this file.

| Source emotion | Positive annotation rows | Distinct comment IDs with a positive annotation |
|---|---:|---:|
| admiration | 5,647 | 4,680 |
| amusement | 3,081 | 2,533 |
| anger | 2,589 | 2,270 |
| annoyance | 4,443 | 4,042 |
| approval | 5,928 | 5,372 |
| caring | 1,988 | 1,805 |
| confusion | 2,471 | 2,199 |
| curiosity | 3,267 | 2,828 |
| desire | 1,248 | 1,127 |
| disappointment | 2,771 | 2,553 |
| disapproval | 3,774 | 3,417 |
| disgust | 1,704 | 1,567 |
| embarrassment | 817 | 756 |
| excitement | 1,900 | 1,730 |
| fear | 1,048 | 925 |
| gratitude | 3,863 | 2,932 |
| grief | 227 | 216 |
| joy | 2,607 | 2,315 |
| love | 2,745 | 2,192 |
| nervousness | 598 | 563 |
| optimism | 2,887 | 2,541 |
| pride | 452 | 422 |
| realization | 2,867 | 2,692 |
| relief | 452 | 433 |
| remorse | 849 | 749 |
| sadness | 2,193 | 1,928 |
| surprise | 1,806 | 1,600 |
| neutral | 18,423 | 15,210 |

Label cardinality per annotation row:

| Positive labels in row | Rows |
|---:|---:|
| Zero | 1,129 |
| One | 56,984 |
| Multiple | 11,887 |

The multi-label counts show that individual rater annotations can contain multiple positive labels. They are annotation-level labels, not self-reports by the Reddit authors.

## Text encoding check

- Strict UTF-8 decoding of the full CSV passed.
- The 25 sampled texts contained **0** replacement-character flags and **0** common-mojibake-marker flags.
- Across all nonempty text rows, the scan found **0** replacement-character rows and **0** rows containing the checked mojibake markers.
- These are heuristic checks; they do not rule out every possible transcription or encoding issue.

## Mapping to Sameer's five target emotions

| Project target | Direct source label in GoEmotions part 1 | Finding |
|---|---|---|
| Bored | None | Unsupported; leave unmapped. |
| Confident | None | Unsupported; leave unmapped. |
| Confused | `confusion` | Direct concept-level label exists: 2,471 positive annotation rows across 2,199 distinct comment IDs in this file. |
| Curious | `curiosity` | Direct concept-level label exists: 3,267 positive annotation rows across 2,828 distinct comment IDs in this file. |
| Frustrated | None | Unsupported; leave unmapped. |

Do not map unrelated labels such as `neutral`, `annoyance`, `anger`, `optimism`, or `pride` to fill missing target categories. Even the direct `confusion` and `curiosity` labels come from general Reddit-comment annotations, not study-specific learner self-reports; they are only candidates for carefully bounded auxiliary analysis.

## Limitations and recommendation

1. This audit covers only `goemotions_1.csv`, not the remaining official raw files or the complete dataset. The 70,000 rows are rater annotations, not 70,000 distinct comments; this file contains 43,358 distinct nonempty comment IDs.
2. The source domain is Reddit conversation, not learner study challenges. Labels represent annotators' judgments about comment text, not the authors' confirmed internal states.
3. The five project labels are not fully covered. Only `confusion` and `curiosity` have direct source-label matches; Bored, Confident, and Frustrated remain unsupported.
4. The raw file contains author and other metadata fields. This report intentionally includes no metadata values or text samples; keep the source file access restricted and do not expose it in logs or public repositories.
5. The dataset-specific rights for Reddit comment text were not established by the repository README. Review intended use and applicable platform/data terms before training or redistribution.
6. No model was trained, and no model accuracy or performance is claimed.

**Recommendation:** Keep all five product labels unmixed with unrelated GoEmotions categories. If approved after rights review, use GoEmotions only as a separately identified auxiliary source for `Confused`/`Curious`; obtain appropriately approved study-specific labels or revise the model label set before claiming five-category coverage.
