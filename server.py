from transformers import (
    BertTokenizerFast,
    GPT2LMHeadModel,
    GPT2ForSequenceClassification,
)
import logging
import os
import argparse
import json
import streamlit as st

import torch
import pytorch_lightning as pl

from transformers import BertTokenizerFast
from Modeling.train import ArgsBase, FoodDataModule, Classification, KoGPTClassification
from dotenv import load_dotenv


@st.cache(allow_output_mutation=True)
def load_model(ckpt_path: str, args):
    model = KoGPTClassification(args)
    model.eval()
    new_state_dict = torch.load(ckpt_path)["state_dict"]

    model.load_state_dict(new_state_dict)
    return model


@st.cache(allow_output_mutation=True)
def load_tokenizer():
    return BertTokenizerFast.from_pretrained("kykim/gpt3-kor-small_based_on_gpt2")


def infer(model, tokenizer, text):
    tokens = [tokenizer.cls_token] + tokenizer.tokenize(text) + [tokenizer.sep_token]
    encoder_input_id = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = [1] * len(encoder_input_id)
    if len(encoder_input_id) < 64:
        while len(encoder_input_id) < 64:
            encoder_input_id += [tokenizer.pad_token_id]
            attention_mask += [0]
    else:
        encoder_input_id = encoder_input_id[: 64 - 1] + [tokenizer.sep_token_id]
        attention_mask = attention_mask[:64]

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
    load_dotenv(verbose=True)

    NUM_LABELS = int(os.getenv("NUM_LABELS", 100))
    IMAGE_PATH = os.getenv("IMAGE_PATH", "./DataCrawling/data/image.json")
    RESTAURANT_PATH = os.getenv(
        "RESTAURANT_PATH", "./DataCrawling/data/restaurant.json"
    )
    CKPT_PATH = os.getenv("CKPT_PATH", "")
    if CKPT_PATH == "":
        raise ValueError("CKPT_PATH is not defined")

    with open(IMAGE_PATH, "r") as f:
        image = json.load(f)
        restaurant = list(image.keys())[:NUM_LABELS]

    with open(RESTAURANT_PATH, "r") as f:
        link = json.load(f)

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
    model = load_model(CKPT_PATH, args)
    tokenizer = load_tokenizer()

    st.title("뭘 먹을지 고민된다면 AI가 추천해줄게요.")

    q = st.text_input("애매하게 얘기해도 좋으니 뭘 먹고 싶은지 얘기해보세요.", "달달한 디저트 좀 먹고 싶어")

    if st.button("Click"):
        rank = infer(model, tokenizer, q)
        col = st.columns(3)
        for i, c in enumerate(col):
            with c:
                res_id = rank[i]
                st.subheader(restaurant[res_id])
                st.image(image[restaurant[res_id]])
                st.markdown(
                    f"[Details](https://www.mangoplate.com{link[restaurant[res_id]]})"
                )
