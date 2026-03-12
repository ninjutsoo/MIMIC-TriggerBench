# Normalization coverage (generated)

- Backend: files
- Root: `C:\Users\mrosh\OneDrive\Documents\GitHub\MIMIC-TriggerBench\physionet.org\files\mimiciv\3.1`
- Max rows scanned per table: 50000
- Top-K unmapped shown: 15

## Summary table

| table | rows_scanned | mapped | unmapped | ambiguous | mapped_rate |
|-------|------------:|------:|--------:|----------:|-----------:|
| `labevents` | 50000 | 5573 | 44427 | 0 | 0.111 |
| `chartevents` | 50000 | 3197 | 46803 | 0 | 0.064 |
| `inputevents` | 50000 | 20362 | 29638 | 8362 | 0.407 |
| `procedureevents` | 50000 | 726 | 49274 | 726 | 0.015 |
| `prescriptions` | 50000 | 5382 | 44618 | 1613 | 0.108 |
| `emar` | 50000 | 3358 | 44166 | 521 | 0.067 |

## `labevents`

- Rows scanned: 50000
- Mapped: 5573
- Unmapped: 44427
- Ambiguous (mapped but flagged): 0

### Top unmapped itemids

- `itemid=51221`: 1332
- `itemid=50983`: 1321
- `itemid=50902`: 1314
- `itemid=51265`: 1311
- `itemid=51222`: 1295
- `itemid=51301`: 1290
- `itemid=51248`: 1289
- `itemid=51249`: 1289
- `itemid=51250`: 1289
- `itemid=51277`: 1289
- `itemid=51279`: 1289
- `itemid=50882`: 1280
- `itemid=50868`: 1278
- `itemid=50893`: 865
- `itemid=50934`: 822

## `chartevents`

- Rows scanned: 50000
- Mapped: 3197
- Unmapped: 46803
- Ambiguous (mapped but flagged): 0

### Top unmapped itemids

- `itemid=227969`: 1615
- `itemid=220277`: 1073
- `itemid=220210`: 1033
- `itemid=220048`: 965
- `itemid=220180`: 951
- `itemid=224650`: 804
- `itemid=229381`: 591
- `itemid=228988`: 513
- `itemid=227958`: 492
- `itemid=224082`: 423
- `itemid=224080`: 412
- `itemid=224086`: 401
- `itemid=224093`: 374
- `itemid=228928`: 335
- `itemid=228957`: 334

## `inputevents`

- Rows scanned: 50000
- Mapped: 20362
- Unmapped: 29638
- Ambiguous (mapped but flagged): 8362

### Top unmapped itemids

- `itemid=225943`: 3260
- `itemid=226452`: 2187
- `itemid=222168`: 2044
- `itemid=225799`: 1641
- `itemid=221744`: 1378
- `itemid=226453`: 1248
- `itemid=225975`: 929
- `itemid=225942`: 913
- `itemid=225166`: 887
- `itemid=221668`: 813
- `itemid=225798`: 603
- `itemid=221794`: 557
- `itemid=225152`: 461
- `itemid=227522`: 457
- `itemid=226089`: 452

## `procedureevents`

- Rows scanned: 50000
- Mapped: 726
- Unmapped: 49274
- Ambiguous (mapped but flagged): 726

### Top unmapped itemids

- `itemid=224275`: 6375
- `itemid=225459`: 4542
- `itemid=224277`: 4012
- `itemid=225402`: 2666
- `itemid=225752`: 2453
- `itemid=225792`: 2234
- `itemid=224274`: 1625
- `itemid=224263`: 1620
- `itemid=227194`: 1610
- `itemid=225401`: 1576
- `itemid=229351`: 1472
- `itemid=221214`: 1437
- `itemid=229581`: 1329
- `itemid=228129`: 1193
- `itemid=225469`: 1058

## `prescriptions`

- Rows scanned: 50000
- Mapped: 5382
- Unmapped: 44618
- Ambiguous (mapped but flagged): 1613

### Top unmapped labels

- `0.9% Sodium Chloride`: 1755
- `Sodium Chloride 0.9%  Flush`: 1709
- `Potassium Chloride`: 1708
- `Acetaminophen`: 1455
- `Furosemide`: 1302
- `Heparin`: 1033
- `Magnesium Sulfate`: 964
- `Bag`: 937
- `5% Dextrose`: 916
- `Senna`: 891
- `Docusate Sodium`: 855
- `Metoprolol Tartrate`: 826
- `HYDROmorphone (Dilaudid)`: 795
- `Iso-Osmotic Dextrose`: 735
- `Ondansetron`: 613

## `emar`

- Rows scanned: 50000
- Mapped: 3358
- Unmapped: 44166
- Ambiguous (mapped but flagged): 521

### Top unmapped labels

- `Sodium Chloride 0.9%  Flush`: 5834
- `Heparin`: 2533
- `Acetaminophen`: 2066
- `Docusate Sodium`: 1140
- `HYDROmorphone (Dilaudid)`: 1064
- `OxyCODONE (Immediate Release)`: 1040
- `Senna`: 796
- `Metoprolol Tartrate`: 720
- `Gabapentin`: 718
- `Lactulose`: 624
- `Pantoprazole`: 590
- `Lidocaine 5% Patch`: 564
- `Furosemide`: 548
- `Atorvastatin`: 497
- `Aspirin`: 468
