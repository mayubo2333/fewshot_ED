# Dataset Processing

- Preprocessing code for [Few-shot Event Detection: An Empirical Study and a Unified View](https://arxiv.org/abs/2305.01901)
- Please contact [Yubo Ma](mailto:yubo001@e.ntu.edu.sg) for questions and suggestions.

## Dataset preparation

We follow the preprocessing methods listed below and sincerely thank their previous work.

 | Dataset      | Preprocessing |
 | ----------- | ----------- |
 | ACE05 | [HMEAE](https://github.com/thunlp/HMEAE) |
 | MAVEN | [MAVEN](https://github.com/THU-KEG/MAVEN-dataset) |
 | ERE | [OmniEvent](https://github.com/THU-KEG/OmniEvent) |

 Please store the preprocessed data in ```./data``` folder with the structure below

```
data
  ├── ACE05_processed
  │   ├── train.json
  │   ├── dev.json
  │   └── test.json
  ├── MAVEN
  │   ├── train.jsonl
  │   ├── valid.jsonl
  │   └── test.jsonl
  └── ERE
      ├── processed
      │   ├── LDC2015E29.unified.jsonl
      │   ├── LDC2015E68.unified.jsonl 
      │   └── LDC2015E78.unified.jsonl 
      └── splits
          ├── train.doc.txt
          ├── dev.doc.txt
          └── test.doc.txt 
```

Then further preprocessing procedure for ERE dataset is necessary. Run
```
cd ./ERE
python data_split.py
```
The preprocessed data is then stored in ```./ERE/[train|dev|test].jsonl```


## Few-shot Dataset Construction
We conduct our empirical study on two task settings, (1) low-resource setting and (2) class-transfer setting. You could find detailed definition about them in our paper.

### Low-resource Setting
```
cd ./k_shot
bash run.sh [ACE|MAVEN|ERE]
```
You could find constructed few-shot dataset in ```./k_shot/fewshot_set```

### Class-transfer Setting
```
cd ./class_transfer
bash run.sh [ACE|MAVEN|ERE]
```
You could find constructed few-shot dataset in ```./class_transfer/fewshot_set```