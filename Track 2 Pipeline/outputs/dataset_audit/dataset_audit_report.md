# Track 2 MMRDR Dataset Audit

## Integrity Checks

| modality | check | value | status |
| --- | --- | --- | --- |
| CFP | csv_rows | 11118 | info |
| CFP | jpg_files | 11118 | info |
| CFP | missing_image_paths | 0 | pass |
| CFP | extra_jpg_not_in_csv | 0 | pass |
| CFP | unknown_split_prefix | 0 | pass |
| CFP | invalid_grade | 0 | pass |
| CFP | exact_duplicate_images | 0 | not_run |
| UWF | csv_rows | 10404 | info |
| UWF | jpg_files | 10404 | info |
| UWF | missing_image_paths | 0 | pass |
| UWF | extra_jpg_not_in_csv | 0 | pass |
| UWF | unknown_split_prefix | 0 | pass |
| UWF | invalid_grade | 0 | pass |
| UWF | exact_duplicate_images | 0 | not_run |
| OCT | csv_rows | 2938 | info |
| OCT | jpg_files | 2938 | info |
| OCT | missing_image_paths | 0 | pass |
| OCT | extra_jpg_not_in_csv | 0 | pass |
| OCT | unknown_split_prefix | 0 | pass |
| OCT | invalid_grade | 0 | pass |
| OCT | exact_duplicate_images | 0 | not_run |

## Class Distribution

| modality | split | grade | count |
| --- | --- | --- | --- |
| CFP | test | 0 | 1323 |
| CFP | test | 1 | 259 |
| CFP | test | 2 | 409 |
| CFP | test | 3 | 100 |
| CFP | test | 4 | 134 |
| CFP | train | 0 | 5256 |
| CFP | train | 1 | 1043 |
| CFP | train | 2 | 1550 |
| CFP | train | 3 | 503 |
| CFP | train | 4 | 541 |
| UWF | test | 0 | 800 |
| UWF | test | 1 | 509 |
| UWF | test | 2 | 544 |
| UWF | test | 3 | 358 |
| UWF | test | 4 | 386 |
| UWF | train | 0 | 2572 |
| UWF | train | 1 | 1427 |
| UWF | train | 2 | 1622 |
| UWF | train | 3 | 1019 |
| UWF | train | 4 | 1167 |
| OCT | test | 0 | 214 |
| OCT | test | 1 | 36 |
| OCT | test | 2 | 312 |
| OCT | train | 0 | 792 |
| OCT | train | 1 | 184 |
| OCT | train | 2 | 1400 |

## Lesion Label Counts

| lesion | positive_count | modality |
| --- | --- | --- |
| microaneurysm | 4398 | CFP |
| hard_exudate | 2648 | CFP |
| intraretinal_hemorrhage | 2580 | CFP |
| vb_irma | 236 | CFP |
| neovascularization | 556 | CFP |
| vitreous_hemorrhage | 358 | CFP |
| retinal_detachment | 93 | CFP |
| microaneurysm | 6938 | UWF |
| hard_exudate | 3846 | UWF |
| intraretinal_hemorrhage | 3849 | UWF |
| vb_irma | 1010 | UWF |
| neovascularization | 1188 | UWF |
| vitreous_hemorrhage | 1219 | UWF |
| retinal_detachment | 238 | UWF |

## Split Policy

- `tr*.jpg` files are training/validation candidates.
- `ts*.jpg` files are locked test cases.
- Validation splits should be stratified and created only from `tr*.jpg` rows.
- Exact duplicate hashing is optional because it reads every image byte; run with `--hash-duplicates` for the full duplicate audit.