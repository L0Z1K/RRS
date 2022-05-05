# Modeling

<p align="left">
    <img alt="PyTorch Lightning" src="https://img.shields.io/badge/PyTorch Lightning-792EE5?style=for-the-badge&logo=PyTorch Lightning&logoColor=white" />
</p>

### Environments

```bash
NUM_LABELS=100
CKPT_DIR=model_chp
LOGS_DIR=tb_logs
REVIEW_JSON="../DataCrawling/data/review.json"
TRAIN_VAL_SPLIT=0.8
```

I used [KoGPT3 Pretrained Model](https://huggingface.co/kykim/gpt3-kor-small_based_on_gpt2) on HuggingFace. 

The Model predict the restaurant with some reviews. This is Classification Problem.

## Steps

1. `train.py` : Save a list of restaurant in `data/restaurant.json`.

```bash
$ python train.py --gpus 1 --max_epochs 100
```

## Results

<p align="center">
    <img alt="Train Loss" src="https://user-images.githubusercontent.com/64528476/166885418-c84be3c0-605c-4fbb-af7e-7bb6d11f8746.png" />
    <img alt="Valid Acc" src="https://user-images.githubusercontent.com/64528476/166885524-55c0a9b7-c9b3-459b-b396-ec29b2b8b550.png" />
</p>