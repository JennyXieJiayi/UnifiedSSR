# UnifiedSSR: A Unified Framework of Sequential Search and Recommendation

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
2. Finetuned model for search: `models/Amazon_Clothing/pretrain_recommendation_search/finetune_recommendation/model.pth`
3. Finetuned model for recommendation: `models/Amazon_Clothing/pretrain_recommendation_search/model.pth`

## Datasets

* Original Dataset
  * Amazon dataset can be found in [here](https://nijianmo.github.io/amazon/index.html).
  * JDsearch dataset can be found in [here](https://github.com/rucliujn/JDsearch).

* Preprocess
  * We provide the jupyter notebook for Amazon datasets preprocessing in `datasets/amazon_data_processing.ipynb`.
  * You can also directly use the provided preprocessed Amazon-CL dataset in `datasets/Amazon_Clothing`.

## Citation

If you find our codes helpful, please kindly cite the following papers:

```
@article{}
```