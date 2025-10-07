## PPL Results


|   Model    |  Dataset   | Rank  | Compression Ratio |  PPL  |
| :--------: | :--------: | :---: | :---------------: | :---: |
| LLama-3-8B | WikiText-2 | Full  |         1         | 8.87  |
|            | WikiText-2 |  160  |       1.73        | 11.56 |
|            |  Pile-10k  | Full  |         1         | 6.43  |
|            |  Pile-10k  |  160  |       1.73        |       |


## RULER Results

|   Model    | Dataset | Rank  | Compression Ratio | Acc (%) |
| :--------: | :-----: | :---: | :---------------: | :-----: |
| LLama-3-8B |  QA-1   |  160  |       1.73        |  7.29   |
|            |  QA-1   |  320  |       1.52        |  8.33   |
|            |  QA-1   |  640  |       1.23        |  8.33   |
|            |  QA-1   |  960  |       1.03        |  9.38   |
|            |  QA-1   | Full  |         1         |   75    |
|            |  QA-2   |  160  |       1.73        |  11.5   |
|            |  QA-2   |  320  |       1.52        |  12.5   |
|            |  QA-2   |  640  |       1.23        |  13.5   |
|            |  QA-2   |  960  |       1.03        |  11.4   |
|            |  QA-2   | Full  |         1         |   51    |
|   Yi-9B    |  QA-1   |  160  |       1.52        |  14.6   |
|            |  QA-1   |  320  |       1.23        |  16.7   |
|            |  QA-1   | Full  |         1         |  56.3   |
|            |  QA-2   |  160  |       1.52        |  16.7   |
|            |  QA-2   |  320  |       1.23        |  13.5   |
|            |  QA-2   | Full  |         1         |  39.6   |


## ICL Results


|   Model    |  Dataset  | Rank  | Compression Ratio (%) | Acc (%) |
| :--------: | :-------: | :---: | :-------------------: | :-----: |
| LLama-3-8B | Banking77 | Full  |           1           |  61.7   |
|            | Banking77 |  160  |         1.73          |         |
|            | clinic150 | Full  |           1           |  58.9   |
|            | clinic150 |  160  |         1.73          |         |


