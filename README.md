# Few-shot Event Detection: An Empirical Study and a Unified View
This is the implementation of the paper [Few-shot Event Detection: An Empirical Study and a Unified View](https://arxiv.org/abs/2305.01901). ACL'2023.

## Data
See details in [dataset_processing](./dataset_processing/) pages


## Requirements
* python 3.8.12
* Pytorch 1.7.0
* Transformers 4.10.0

You can install other dependencies by ```pip install -r requirements.txt```

## Code
Simplified source code (version 1). It includes the core parts of our work (i.e., the unified baseline proposed). The authors would find time cleaning the remaining code and make it publicly available as soon as possible.

To run this code,
```
DATA=[ACE|MAVEN|ERE] K=[2|5|10] idx=[0|1|2|3|4|5|6|7|8|9] bash run.sh
```

## Citation
Please cite our paper if you use it in your work:
```bibtex
@misc{ma2023fewshot,
      title={Few-shot Event Detection: An Empirical Study and a Unified View}, 
      author={Yubo Ma and Zehao Wang and Yixin Cao and Aixin Sun},
      year={2023},
      eprint={2305.01901},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
