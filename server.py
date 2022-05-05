from transformers import BertTokenizerFast, GPT2LMHeadModel, GPT2ForSequenceClassification
import logging
import argparse
import json
import pandas as pd
import random
import numpy as np
import streamlit as st

import re
import os
from pytorch_lightning import loggers as pl_loggers
import torch
import pytorch_lightning as pl

from torchmetrics import Accuracy
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import BertTokenizerFast
from torch.utils.data import Dataset, DataLoader, random_split

from train import ArgsBase, FoodDataset, FoodDataModule, Classification, KoGPTClassification

NUM_LABELS = 100
with open("image.json", "r") as f:
    image = json.load(f)
    restaurant = list(image.keys())[:NUM_LABELS]

with open("result.json", "r") as f:
    result = json.load(f)

@st.cache(allow_output_mutation=True)
def load_model(ckpt_path: str, args):
    model = KoGPTClassification(args)
    model.eval()
    new_state_dict = torch.load(ckpt_path)["state_dict"]

    model.load_state_dict(new_state_dict)
    return model

@st.cache(allow_output_mutation=True)
def load_tokenizer():
    return BertTokenizerFast.from_pretrained(
            "kykim/gpt3-kor-small_based_on_gpt2"
    )

def infer(model, tokenizer, text):
    tokens = (
        [tokenizer.cls_token]
        + tokenizer.tokenize(text)
        + [tokenizer.sep_token]
    )
    encoder_input_id = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = [1] * len(encoder_input_id)
    if len(encoder_input_id) < 64:
        while len(encoder_input_id) < 64:
            encoder_input_id += [tokenizer.pad_token_id]
            attention_mask += [0]
    else:
        encoder_input_id = encoder_input_id[: 64 - 1] + [
            tokenizer.sep_token_id
        ]
        attention_mask = attention_mask[: 64]

    input_ids = torch.LongTensor(encoder_input_id).reshape(1, -1)
    attention_mask = torch.FloatTensor(attention_mask).reshape(1, -1)
    y = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
    
    logits = y.logits[0]
    rank = torch.argsort(logits, descending=True)
    return rank

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="subtask for KoBART")
    parser.add_argument(
        "--cachedir", type=str, default=os.path.join(os.getcwd(), ".cache")
    )
    parser.add_argument("--subtask", type=str, default="NSMC", help="NSMC")
    parser = Classification.add_model_specific_args(parser)
    parser = ArgsBase.add_model_specific_args(parser)
    parser = FoodDataModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    logging.info(args)

    if args.default_root_dir is None:
        args.default_root_dir = args.cachedir

    # init model
    model = load_model("./.cache/model_chp/epoch=99-val_acc=0.531.ckpt", args)
    tokenizer = load_tokenizer()

    st.title("뭘 먹을지 고민된다면 AI가 추천해줄게요.")

    q = st.text_input("애매하게 얘기해도 좋으니 뭘 먹고 싶은지 얘기해보세요.", "달달한 디저트 좀 먹고 싶어")

    if st.button("Click"):
        rank = infer(model, tokenizer, q)   
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader(restaurant[rank[0]])
            st.image(image[restaurant[rank[0]]])
            st.markdown(f"[Details](https://www.mangoplate.com{result[restaurant[rank[0]]]})")
        with col2:
            st.subheader(restaurant[rank[1]])
            st.image(image[restaurant[rank[1]]])
            st.markdown(f"[Details](https://www.mangoplate.com{result[restaurant[rank[1]]]})")

        with col3:
            st.subheader(restaurant[rank[2]])
            st.image(image[restaurant[rank[2]]])
            st.markdown(f"[Details](https://www.mangoplate.com{result[restaurant[rank[2]]]})")

    