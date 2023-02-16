#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
import re

import numpy as np
import torch
from datasets import load_dataset
from openprompt.plms import load_plm
from openprompt.utils.reproduciblity import set_seed


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


set_seeds()


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_type", default="mix",
                        type=str, help="prompt's training style")
    parser.add_argument("--template_type", default="ptuning", type=str, choices=['man', 'soft', 'mix', 'ptuning', 'ptr'],
                        help="The way prompt templates are constructed")
    parser.add_argument("--dataset", default="go_emotions", type=str, choices=["go_emotions", "emotion"],
                        help="The dataset")
    parser.add_argument("--data_condition", default="", type=str, choices=['full_data', 'fewshot', 'DA'],
                        help="Data scale for model training")
    parser.add_argument("--few_shot_train", default=True,
                        type=bool, help="few-shot or not")
    parser.add_argument("--model", default="BERT", type=str, choices=["TARA", "Bert", "calibration", "Bert_large", "roberta-base", "roberta-large", "ALBERT"],
                        help="pretrained model")
    parser.add_argument("--model_lr", default=1e-5, type=int)
    parser.add_argument("--template_lr", default=5e-4, type=int)
    parser.add_argument("--template_lr", default=5e-4, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--use_cuda", default=True, type=bool)
    parser.add_argument("--bank_size", default=32, type=int,
                        help="Parameters required for TML")

    args = parser.parse_args()
    return args


def train(args):
    """Main training function."""
    if args.model == "TARA":
        plm, tokenizer, model_config, WrapperClass = load_plm("TARA", "bert-base-cased")
    elif args.model == "calibration":
        plm, tokenizer, model_config, WrapperClass = load_plm("BertForCalibration", "bert-base-cased")
    elif args.model == "Bert":
        plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")
    elif args.model == "Bert_large":
        plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-large-cased")
    elif args.model == "roberta-base":
        plm, tokenizer, model_config, WrapperClass = load_plm("roberta", "roberta-base")
    elif args.model == "roberta-large":
        plm, tokenizer, model_config, WrapperClass = load_plm("roberta", "roberta-large")
    elif args.model == "ALBERT":
        plm, tokenizer, model_config, WrapperClass = load_plm("albert", "albert-base-v2")
    else:
        print("Other models are not yet supported")


if __name__ == "__main__":
    args = setup_args()
    train(args)
