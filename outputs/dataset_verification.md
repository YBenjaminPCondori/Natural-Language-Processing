# Dataset Verification

Created: 2026-05-03T16:53:06.238358+00:00

## Split Counts
| split | raw_rows | processed_rows |
| --- | --- | --- |
| train | 60000 | 28587 |
| validation | 10000 | 4670 |
| test | 10000 | 4732 |

## Integrity Checks
- Labels selected from training split only: `True`
- Validation/test labels subset of selected train labels: `True`
- Text+label cross-split overlaps after de-duplication: `{'train_vs_validation': 0, 'train_vs_test': 0, 'validation_vs_test': 0}`
- Label IDs consistent: `True`
