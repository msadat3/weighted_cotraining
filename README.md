# Co-training for Low Resource Scientific Natural Language Inference

This repository contains the data and code for ACL 2024 paper "[Co-training for Low Resource Scientific Natural Language Inference](https://aclanthology.org/2024.acl-long.139/)."

## Abstract
Scientific Natural Language Inference (NLI) is the task of predicting the semantic relation between a pair of sentences extracted from research articles. The automatic annotation method based on distant supervision for the training set of SciNLI, the first and most popular dataset for this task, results in label noise which inevitably degenerates the performance of classifiers. In this paper, we propose a novel co-training method that assigns weights based on the training dynamics of the classifiers to the distantly supervised labels, reflective of the manner they are used in the subsequent training epochs. That is, unlike the existing semi-supervised learning (SSL) approaches, we consider the historical behavior of the classifiers to evaluate the quality of the automatically annotated labels. Furthermore, by assigning importance weights instead of filtering out examples based on an arbitrary threshold on the predicted confidence, we maximize the usage of automatically labeled data, while ensuring that the noisy labels have a minimal impact on model training. The proposed method obtains an improvement of **1.5%** in Macro F1 over the distant supervision baseline, and substantial improvements over several other strong SSL baselines.

## Dataset Description

The `data/` directory contains all datasets used in our experiments. 

The full SciNLI training set originally consisted of approximately 101,000 examples. From this collection, we manually curated 2,000 examples with the help of three expert annotators. These human-annotated examples form our high-quality labeled subset.

To avoid data leakage, we removed all other examples that originated from the same source papers as the human-annotated samples. After this filtering step, about 97,000 examples remained. These constitute our automatically annotated dataset.

**Files included in `data/`:**

- `train_1.tsv` — First split of the human-annotated dataset  
- `train_2.tsv` — Second split of the human-annotated dataset  
- `Automatically_annotated.tsv` — Automatically annotated dataset (≈97k examples)

The two human-annotated files together contain the full set of 2,000 examples. All files are provided in TSV format.

For more information about the manual annotation process, please refer to Appendix A of our paper.

## Training & Testing
### Requirements
```
numpy
pandas
scikit_learn
torch
transformers
```

### Running the Proposed Weighted Co-training Approach
```bash
python cotraining.py \
    --base <path-to-experiment-base> \        # REQUIRED: directory containing train/dev/test + automatically annotated data in tsv format
    --device_1 <device-for-model-1> \         # e.g., cuda:0 or cpu
    --device_2 <device-for-model-2> \         # e.g., cuda:1 or cpu
    --seed <random-seed> \                    # default: 1234
    --model_type <RoBERTa-or-Sci_BERT> \      # default: RoBERTa
    --dataset <dataset-name> \                # default: SciNLI
    --patience <early-stopping-patience> \    # default: 2
    --epoch_patience <epoch-patience> \       # default: 2
    --batch_size <batch-size> \               # default: 32
    --num_epochs <num-training-epochs> \      # default: 5
    --report_every <steps-between-logs> \     # default: 10
    --cuda_devices "<comma-separated-devices>" \  # default: "0,1"
    --lr_initial <initial-learning-rate> \        # default: 2e-5
    --lr_finetune <finetune-learning-rate>        # default: 2e-6

```


## Citation
If you use our code, data, or proposed approach in your research, please cite our paper:

```
@inproceedings{sadat-caragea-2024-co,
    title = "Co-training for Low Resource Scientific Natural Language Inference",
    author = "Sadat, Mobashir  and
      Caragea, Cornelia",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.139/",
    doi = "10.18653/v1/2024.acl-long.139",
    pages = "2538--2550",
}
```

## License
The data and code in the paper are licensed under the Attribution-ShareAlike 4.0 International license [(CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).