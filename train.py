#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import datetime
import random
import re
import time
from csv import reader

import numpy as np
import torch
from datasets import load_dataset
from openprompt import PromptDataLoader, PromptForClassification
from openprompt.plms import load_plm
from openprompt.prompts import (ManualTemplate, ManualVerbalizer,
                                MixedTemplate, PTRTemplate, PtuningTemplate,
                                SoftTemplate)
from openprompt.utils.reproduciblity import set_seed
from pytorch_metric_learning import (distances, losses, miners, reducers,
                                     testers)
from sklearn.metrics import (accuracy_score, classification_report,
                             precision_recall_fscore_support)
from torch import linalg as LA
from torch.nn.utils.parametrizations import spectral_norm
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from transformers.optimization import AdamW

from data.reader import Reader
from models.TML import TML

min_f1 = 0.0


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
                        type=str, choices=['man', 'mix', 'PCA', 'TML', 'TARA', 'calibration'], help="prompt's training style")
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


def compute_metrics(labels, preds):
    assert len(preds) == len(labels)
    results = dict()

    results["accuracy"] = accuracy_score(labels, preds)
    results["macro_precision"], results["macro_recall"], results[
        "macro_f1"], _ = precision_recall_fscore_support(
        labels, preds, average="macro")
    results["micro_precision"], results["micro_recall"], results[
        "micro_f1"], _ = precision_recall_fscore_support(
        labels, preds, average="micro")
    results["weighted_precision"], results["weighted_recall"], results[
        "weighted_f1"], _ = precision_recall_fscore_support(
        labels, preds, average="weighted")

    return results


def train(args):
    """Main training function."""
    if args.model == "TARA":
        plm, tokenizer, model_config, WrapperClass = load_plm(
            "TARA", "bert-base-cased")
    elif args.model == "calibration":
        plm, tokenizer, model_config, WrapperClass = load_plm(
            "BertForCalibration", "bert-base-cased")
    elif args.model == "Bert":
        plm, tokenizer, model_config, WrapperClass = load_plm(
            "bert", "bert-base-cased")
    elif args.model == "Bert_large":
        plm, tokenizer, model_config, WrapperClass = load_plm(
            "bert", "bert-large-cased")
    elif args.model == "roberta-base":
        plm, tokenizer, model_config, WrapperClass = load_plm(
            "roberta", "roberta-base")
    elif args.model == "roberta-large":
        plm, tokenizer, model_config, WrapperClass = load_plm(
            "roberta", "roberta-large")
    elif args.model == "ALBERT":
        plm, tokenizer, model_config, WrapperClass = load_plm(
            "albert", "albert-base-v2")
    else:
        print("Other models are not yet supported")

    if args.template_type == "man":
        # manual
        # template_text = '{"placeholder":"text_a"} Is was {"mask"}.'
        # template_text = '{"placeholder":"text_a"} I am {"mask"}.'
        template_text = '{"placeholder":"text_a"} The emotional aspect of this text is {"mask"}.'
        mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)
    elif args.template_type == "soft":
        # Soft
        # template_text = '{"placeholder":"text_a"} {"soft": "It was"} {"mask"}.'
        # template_text = '{"placeholder":"text_a"} {"soft": None, "duplicate": 10, "same": True} {"mask"}.'
        template_text = '{"placeholder":"text_a"} {"soft": "The emotional aspect of this text is"} {"mask"}.'
        mytemplate = SoftTemplate(
            model=plm, tokenizer=tokenizer, text=template_text)
    elif args.template_type == "mix":
        template_text = '{"placeholder":"text_a"} {"soft": "The"} emotional {"soft": "aspect of this text is"} {"mask"}.'
        # template_text = '{"placeholder":"text_a"} {"soft": "The"} emotional {"soft": "aspect of this text is"} {"mask"} {"soft"}.'
        mytemplate = MixedTemplate(
            model=plm, tokenizer=tokenizer, text=template_text)
    elif args.template_type == "ptuning":
        # template_text = '{"placeholder":"text_a"} {"soft": "It was"} {"mask"}.'
        template_text = '{"placeholder":"text_a"} {"soft": "The"} emotional {"soft": "aspect of this text is"} {"mask"}.'
        mytemplate = PtuningTemplate(
            model=plm, tokenizer=tokenizer, text=template_text)
    elif args.template_type == "ptr":
        template_text = '{"placeholder":"text_a"} {"soft": "The"} emotional {"soft": "aspect of this text is"} {"mask"}.'
        mytemplate = PTRTemplate(
            model=plm, tokenizer=tokenizer, text=template_text)

    # get datasets & DataLoader
    data_reader = Reader(args.dataset, args.data_condition,
                         args.num_examples_per_label)
    dataset, labels = data_reader.process_data()

    if args.data_condition == 'fewshot':
        train_dataloader = PromptDataLoader(
            dataset=dataset['support'],
            template=mytemplate,
            tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass,
            max_seq_length=128,
            decoder_max_length=3,
            batch_size=args.batch_size,
            shuffle=True,
            teacher_forcing=False,
            predict_eos_token=False,
            truncate_method="tail"
        )
    elif args.data_condition == 'full_data':
        train_dataloader = PromptDataLoader(
            dataset=dataset['train'],
            template=mytemplate,
            tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass,
            max_seq_length=128,
            decoder_max_length=3,
            batch_size=args.batch_size,
            shuffle=True,
            teacher_forcing=False,
            predict_eos_token=False,
            truncate_method="tail"
        )

    # Evaluate
    validation_dataloader = PromptDataLoader(
        dataset=dataset["validation"],
        template=mytemplate,
        tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass,
        max_seq_length=128,
        decoder_max_length=3,
        batch_size=args.batch_size,
        shuffle=False,
        teacher_forcing=False,
        predict_eos_token=False,
        truncate_method="tail"
    )

    # Verbalizer
    label_words = {i: [i] for i in labels}
    myverbalizer = ManualVerbalizer(tokenizer,
                                    classes=labels,
                                    label_words=label_words)

    prompt_model = PromptForClassification(
        plm=plm, template=mytemplate, verbalizer=myverbalizer)
    if args.use_cuda:
        prompt_model = prompt_model.cuda()

    if args.data_condition == "full_data":
        model_save_pth = args.model+"_"+str(args.training_type)+"_template_"+args.template_type + \
            "_"+args.data_condition+"_"+"_bankSize_" + \
            str(args.bank_size)+"_dataset_"+args.dataset+".pth"
    elif args.data_condition == "fewshot":
        model_save_pth = args.model+"_"+str(args.training_type)+"_template_"+args.template_type+"_"+args.data_condition+"_"+str(
            args.num_examples_per_label)+"_bankSize_"+str(args.bank_size)+"_dataset_"+args.dataset+".pth"

    # optimizer
    loss_func = torch.nn.CrossEntropyLoss()
    no_decay = ['bias', 'LayerNorm.weight']
    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters
    optimizer_grouped_parameters2 = [
        {'params': [p for n, p in prompt_model.template.named_parameters(
        ) if "raw_embedding" not in n]}
    ]

    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=args.model_lr)
    optimizer2 = AdamW(optimizer_grouped_parameters2, lr=args.template_lr)

    tot_step = len(train_dataloader)*5
    scheduler1 = get_linear_schedule_with_warmup(optimizer1, 0, tot_step)
    scheduler2 = get_linear_schedule_with_warmup(optimizer2, 0, tot_step)

    if args.training_type in ['PCA', 'TML']:
        distance = distances.CosineSimilarity()
        reducer = reducers.ThresholdReducer(low=0)
        loss_func_metric = losses.TripletMarginLoss(
            margin=0.2, distance=distance, reducer=reducer)
        mining_func = miners.TripletMarginMiner(
            margin=0.2, distance=distance, type_of_triplets="semihard"
        )
        level_switch = TML(args.dataset, labels)

    if args.few_shot_train:
        step_index = int(args.bank_size/args.batch_size)
        logits_stack = []
        labels_stack = []
        start_time = time.time()
        for epoch in tqdm(range(args.epoch_num)):
            tot_loss = 0
            for step, inputs in enumerate(train_dataloader):
                prompt_model.train()
                if args.use_cuda:
                    inputs = inputs.cuda()

                logits, _ = prompt_model(inputs)
                labels = inputs['label']

                loss = loss_func(logits, labels)

                # PCA
                if args.training_type == 'PCA':
                    indices_tuple1 = mining_func(logits, labels)
                    auxiliary_loss = loss_func_metric(
                        logits, labels, indices_tuple1)
                    loss = loss + auxiliary_loss

                if args.training_type == 'TML':
                    for logit, label in zip(logits.cpu().tolist(), labels.cpu().tolist()):
                        logits_stack.append(logit)
                        labels_stack.append(label)
                    if step % step_index == 1:
                        if args.dataset == "go_emotions":
                            Hierarchical_labels_mid = list(
                                map(level_switch.label_change_mid, labels_stack))
                            Hierarchical_labels_mid = torch.Tensor(
                                Hierarchical_labels_mid).to(labels.device)
                            logits_stack = torch.Tensor(
                                logits_stack).to(labels.device)

                            indices_tuple = mining_func(
                                logits_stack, Hierarchical_labels_mid)
                            loss2 = loss_func_metric(
                                logits_stack, Hierarchical_labels_mid, indices_tuple)

                            Hierarchical_labels_high = list(
                                map(level_switch.label_change_high, labels_stack))
                            Hierarchical_labels_high = torch.Tensor(
                                Hierarchical_labels_high).to(labels.device)
                            indices_tuple = mining_func(
                                logits_stack, Hierarchical_labels_high)
                            loss3 = loss_func_metric(
                                logits_stack, Hierarchical_labels_high, indices_tuple)

                            loss_TML = 1/7*loss2 + 1/4*loss3

                        elif args.dataset == "emotion":
                            Hierarchical_labels = list(
                                map(level_switch.label_change, labels_stack))
                            Hierarchical_labels = torch.Tensor(
                                Hierarchical_labels).to(labels.device)
                            logits_stack = torch.Tensor(
                                logits_stack).to(labels.device)

                            indices_tuple = mining_func(
                                logits_stack, Hierarchical_labels)
                            loss2 = loss_func_metric(
                                logits_stack, Hierarchical_labels, indices_tuple)

                            loss_TML = 1/4*loss2
                        loss = loss + loss_TML

                if args.training_type == 'TARA':
                    if step % step_index == 1:
                        w1 = prompt_model.plm.fc1.weight.cpu()
                        diff_matrix1 = torch.matmul(
                            w1, w1.transpose(1, 0))-torch.eye(w1.shape[0])
                        loss_w1 = LA.matrix_norm(diff_matrix1)
                        w2 = prompt_model.plm.fc2.weight.cpu()
                        loss_w2 = LA.matrix_norm(w2)-1
                        w3 = prompt_model.plm.fc3.weight.cpu()
                        diff_matrix3 = torch.matmul(w3,w3.transpose(1,0))-torch.eye(w3.shape[0])
                        loss_w3 = LA.matrix_norm(diff_matrix3)
                        loss_collapse =  loss_w1 + loss_w2 + loss_w3
                        loss = loss + loss_collapse

                loss.backward()
                tot_loss += loss.item()
                optimizer1.step()
                optimizer1.zero_grad()
                optimizer2.step()
                optimizer2.zero_grad()
                if step % 100 == 1:
                    print("Epoch {}, average loss: {}".format(
                        epoch, tot_loss/(step+1)), flush=True)

                if step % 500 == 1:
                    # evaluation
                    prompt_model.eval()

                    allpreds = []
                    alllabels = []
                    for step, inputs in enumerate(validation_dataloader):
                        if args.use_cuda:
                            inputs = inputs.cuda()
                        logits, _ = prompt_model(inputs)
                        # logits = prompt_model(inputs)
                        labels = inputs['label']
                        alllabels.extend(labels.cpu().tolist())
                        allpreds.extend(torch.argmax(
                            logits, dim=-1).cpu().tolist())

                    dev_res = compute_metrics(alllabels, allpreds)
                    acc_score = dev_res["accuracy"]
                    f1_score = dev_res["weighted_f1"]

                    # acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
                    print("Dev acc: "+str(acc_score))
                    print("Dev f1:"+str(f1_score))

                    if min_f1 < f1_score:
                        min_f1 = float(f1_score)
                        print("save model in epoch:"+str(epoch))
                        torch.save(prompt_model.state_dict(), model_save_pth)

    end_time = time.time()
    complete_time = end_time - start_time
    print("running time: "+str(datetime.timedelta(seconds=complete_time)))

    # Test
    # Test Zero shot
    test_dataloader = PromptDataLoader(
        dataset=dataset["test"],
        template=mytemplate,
        tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass,
        max_seq_length=256,
        decoder_max_length=3,
        batch_size=args.batch_size,
        shuffle=False,
        teacher_forcing=False,
        predict_eos_token=False,
        truncate_method="tail"
    )
    test_model = prompt_model

    allpreds = []
    all_logits = []
    alllabels = []
    all_mask = []

    pbar = tqdm(test_dataloader)
    for step, inputs in enumerate(pbar):
        test_model.eval()
        if args.use_cuda:
            inputs = inputs.cuda()

        logits, _ = test_model(inputs)

        labels = inputs['label']

        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        all_logits.extend(logits.cpu().tolist())

    result = compute_metrics(alllabels, allpreds)
    print(result)
    print(classification_report(alllabels, allpreds, target_names=labels))


if __name__ == "__main__":
    args = setup_args()
    train(args)
