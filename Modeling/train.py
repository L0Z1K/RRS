import logging
import argparse
import json
import random
import re
import os
import torch

import numpy as np
import pytorch_lightning as pl

from torchmetrics import Accuracy
from pytorch_lightning import loggers as pl_loggers
from transformers import BertTokenizerFast, GPT2ForSequenceClassification
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import BertTokenizerFast
from torch.utils.data import Dataset, DataLoader, random_split
from dotenv import load_dotenv

load_dotenv(verbose=True)

NUM_LABELS = int(os.getenv("NUM_LABELS", 100))
CKPT_DIR = os.getenv("CKPT_DIR", "model_chp")
LOGS_DIR = os.getenv("LOGS_DIR", "tb_logs")


class ArgsBase:
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=64, help="")
        parser.add_argument("--max_seq_len", type=int, default=64, help="")
        return parser


class FoodDataset(Dataset):
    def __init__(self, file_path: str, max_seq_len: int = 64):
        self.regex = re.compile("[가-힣]+")
        self.data = self._init_data(file_path)
        self.tokenizer = BertTokenizerFast.from_pretrained(
            "kykim/gpt3-kor-small_based_on_gpt2"
        )

        self.max_seq_len = max_seq_len

    def _init_data(self, file_path: str):
        result = []
        with open(file_path, "r") as f:
            data = json.load(f)
            self.y = list(data.keys())
            for i, v in enumerate(data.values()):
                if i == NUM_LABELS:
                    break
                if len(v) < 5:
                    continue
                for vv in v:
                    for vvv in vv.split("\n\n"):
                        vvv = vvv.strip()
                        vvv = " ".join(self.regex.findall(vvv))
                        vvv = vvv.strip()
                        if vvv != "":
                            result.append([vvv, i])
        random.shuffle(result)
        return result

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        record = self.data[index]
        document, label = str(record[0]), int(record[1])
        tokens = (
            [self.tokenizer.cls_token]
            + self.tokenizer.tokenize(document)
            + [self.tokenizer.sep_token]
        )
        encoder_input_id = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(encoder_input_id)
        if len(encoder_input_id) < self.max_seq_len:
            while len(encoder_input_id) < self.max_seq_len:
                encoder_input_id += [self.tokenizer.pad_token_id]
                attention_mask += [0]
        else:
            encoder_input_id = encoder_input_id[: self.max_seq_len - 1] + [
                self.tokenizer.sep_token_id
            ]
            attention_mask = attention_mask[: self.max_seq_len]
        return {
            "input_ids": np.array(encoder_input_id, dtype=np.int_),
            "attention_mask": np.array(attention_mask, dtype=float),
            "labels": np.array(label, dtype=np.int_),
        }


class FoodDataModule(pl.LightningDataModule):
    def __init__(self, max_seq_len=64, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        return parser

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):
        # split dataset
        REVIEW_JSON = os.path.join("REVIEW_JSON", "../DataCrawling/data/review.json")
        dataset = FoodDataset(REVIEW_JSON)
        TRAIN_VAL_SPLIT = float(os.path.join("TRAIN_VAL_SPLIT", 0.8))
        train_size = int(len(dataset) * TRAIN_VAL_SPLIT)
        test_size = len(dataset) - train_size
        self.train, self.test = random_split(
            dataset=dataset,
            lengths=[train_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

    # return the dataloader for each split
    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train, batch_size=self.batch_size, num_workers=5, shuffle=True
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.test, batch_size=self.batch_size, num_workers=5, shuffle=False
        )
        return val_dataloader


class Classification(pl.LightningModule):
    def __init__(self, hparams, **kwargs) -> None:
        super(Classification, self).__init__()
        print(vars(hparams))
        self.hparams.update(vars(hparams))

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument(
            "--batch-size",
            type=int,
            default=32,
            help="batch size for training (default: 96)",
        )

        parser.add_argument(
            "--lr", type=float, default=5e-5, help="The initial learning rate"
        )

        parser.add_argument(
            "--warmup_ratio", type=float, default=0.1, help="warmup ratio"
        )

        return parser

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.hparams.lr, correct_bias=False
        )
        # warm up lr
        num_workers = (self.hparams.gpus if self.hparams.gpus is not None else 1) * (
            self.hparams.num_nodes if self.hparams.num_nodes is not None else 1
        )
        data_len = len(
            self.trainer._data_connector._train_dataloader_source.dataloader().dataset
        )
        logging.info(f"number of workers {num_workers}, data length {data_len}")
        num_train_steps = int(data_len * self.hparams.max_epochs)
        logging.info(f"num_train_steps : {num_train_steps}")
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        logging.info(f"num_warmup_steps : {num_warmup_steps}")
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )
        lr_scheduler = {
            "scheduler": scheduler,
            "monitor": "loss",
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]


class KoGPTClassification(Classification):
    def __init__(self, hparams, **kwargs):
        super(KoGPTClassification, self).__init__(hparams, **kwargs)
        self.model = GPT2ForSequenceClassification.from_pretrained(
            "kykim/gpt3-kor-small_based_on_gpt2",
            num_labels=NUM_LABELS,
        )
        self.model.config.pad_token_id = 0
        self.model.train()

        self.metric_acc = Accuracy()

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

    def training_step(self, batch, batch_idx):
        outs = self(batch["input_ids"], batch["attention_mask"], batch["labels"])
        loss = outs.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch["input_ids"], batch["attention_mask"])
        labels = batch["labels"]
        accuracy = self.metric_acc(
            torch.nn.functional.softmax(pred.logits, dim=1), labels
        )
        self.log("accuracy", accuracy)
        result = {"accuracy": accuracy}
        # Checkpoint model based on validation loss
        return result

    def validation_epoch_end(self, outputs):
        val_acc = torch.stack([i["accuracy"] for i in outputs]).mean()
        self.log("val_acc", val_acc, prog_bar=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="subtask for KoBART")
    parser.add_argument(
        "--cachedir", type=str, default=os.path.join(os.getcwd(), ".cache")
    )
    parser = Classification.add_model_specific_args(parser)
    parser = ArgsBase.add_model_specific_args(parser)
    parser = FoodDataModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    logging.info(args)

    if args.default_root_dir is None:
        args.default_root_dir = args.cachedir

    # init model
    model = KoGPTClassification(args)

    dm = FoodDataModule(
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        dirpath=args.default_root_dir,
        filename=CKPT_DIR + "/{epoch:02d}-{val_acc:.3f}",
        verbose=True,
        save_last=True,
        mode="max",
        save_top_k=1,
    )

    tb_logger = pl_loggers.TensorBoardLogger(
        os.path.join(args.default_root_dir, LOGS_DIR)
    )

    # train
    lr_logger = pl.callbacks.LearningRateMonitor()
    trainer = pl.Trainer.from_argparse_args(
        args, logger=tb_logger, callbacks=[checkpoint_callback, lr_logger]
    )
    trainer.fit(model, dm)
