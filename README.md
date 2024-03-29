# (WWW '24) UnifiedSSR: A Unified Framework of Sequential Search and Recommendation

This is the Pytorch implementation of UnifiedSSR for joint learning of user behavior history in both search and recommendation scenarios.

## Environments

- python=3.8.17
- torch=1.13.1
- numpy=1.24.4
- pandas=2.0.3
- scikit-learn=1.3.0
- matplotlib=3.7.1
- nltk=3.8.1
- joblib=1.3.0

You can create the environment via `conda env create -f unifiedssr_env.yaml`.

## Run the Codes

1. Pretrain: Customize parameters in `utils/parser.py`, and then run `pretrain.py` to pretrain the model. The pretrained model will be saved in `models/`.
2. Finetune: Modify `args.trained_model_path` in `train.py` to specify the path to the pretrained model, and then run `train.py` to finetune the model.
3. Evaluate: Modify `args.tasks` in `predict.py` to specify the path to the trained model, and then run `predict.py` to evaluate the model. Note that evaluation can only be conducted on one task at a time.

We provide a pretrained model and task-specific finetuned models for the Amazon-CL dataset as follows:
1. Pretrained model: `models/Amazon_Clothing/pretrain_recommendation_search/model.pth`
2. Finetuned model for search: `models/Amazon_Clothing/pretrain_recommendation_search/finetune_search/model.pth`
3. Finetuned model for recommendation: `models/Amazon_Clothing/pretrain_recommendation_search/finetune_recommendation/model.pth`

## Datasets

* Original Dataset
  * Amazon dataset can be found in [here](https://nijianmo.github.io/amazon/index.html).
  * JDsearch dataset can be found in [here](https://github.com/rucliujn/JDsearch).
* Preprocess
  * Use the provided preprocessed Amazon-CL dataset in `datasets/Amazon_Clothing`.
  * Feel free to contact the author for more details of the data preprocessing.
  

Notes: The dataset and model files are large. Please download them from [Google Drive](https://drive.google.com/drive/folders/1GShl2vju5_uXHRgcd1UZinJhmgmDzSw_?usp=share_link) and place them in the project folder.

## Citation

If you find our codes helpful, please kindly cite the following papers:

```
@article{unifiedssr,
  author       = {Jiayi Xie and
                  Shang Liu and
                  Gao Cong and
                  Zhenzhong Chen},
  title        = {UnifiedSSR: {A} Unified Framework of Sequential Search and Recommendation},
  journal      = {CoRR},
  volume       = {abs/2310.13921},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2310.13921},
  doi          = {10.48550/ARXIV.2310.13921},
  eprinttype    = {arXiv},
  eprint       = {2310.13921}
}
```