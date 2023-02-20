#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datasets import load_dataset
from openprompt.data_utils import InputExample
from openprompt.utils.reproduciblity import set_seed
from openprompt.data_utils.data_sampler import FewShotSampler


class Reader(object):

    def __init__(self, name, division, num_examples_per_label=100) -> None:
        if name not in ["go_emotions", "emotion"]:
            print("Other datasets are not yet supported")

        if name == "go_emotions":
            emotions = load_dataset("go_emotions", "simplified")

            df_train = emotions['train'].to_pandas()
            df_dev = emotions['validation'].to_pandas()
            df_test = emotions['test'].to_pandas()

            # Adjusting the order to determine if the hierarchy can be retained
            # Since the dataset and the paper are in a different tag order, reset
            labels_ = ['amusement', 'excitement', 'joy', 'love', 'desire', 'optimism', 'caring', 'pride', 'admiration', 'gratitude', 'relief', 'approval', 'realization', 'surprise',
                       'curiosity', 'confusion', 'fear', 'nervousness', 'remorse', 'embarrassment', 'disappointment', 'sadness', 'grief', 'disgust', 'anger', 'annoyance', 'disapproval', 'neutral']
            labels_cols = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
                           "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"]

            change_dict = {}
            for idx, item in enumerate(labels_cols):
                change_dict[idx] = labels_.index(item)

            # print(change_dict)
            # print(labels_)
            # print(len(labels_))

            df_train["labels_num"] = list(
                map(len, df_train["labels"].values.tolist()))
            df_train = df_train.drop(
                df_train[df_train["labels_num"] != 1].index)
            # print(df_train["labels"].values.tolist()[0])
            df_train["label"] = list(
                map(lambda a: change_dict[a.tolist()[0]], df_train["labels"].values.tolist()))
            df_train["idx"] = df_train.index

            df_dev["labels_num"] = list(
                map(len, df_dev["labels"].values.tolist()))
            df_dev = df_dev.drop(df_dev[df_dev["labels_num"] != 1].index)
            df_dev["label"] = list(
                map(lambda a: change_dict[a.tolist()[0]], df_dev["labels"].values.tolist()))
            df_dev["idx"] = df_dev.index

            df_test["labels_num"] = list(
                map(len, df_test["labels"].values.tolist()))
            df_test = df_test.drop(df_test[df_test["labels_num"] != 1].index)
            df_test["label"] = list(
                map(lambda a: change_dict[a.tolist()[0]], df_test["labels"].values.tolist()))
            df_test["idx"] = df_test.index

        elif name == "emotion":
            emo = load_dataset("emotion")

            label_cols = ["sadness", "joy", "love",
                          "anger", "fear", "surprise"]
            labels_ = label_cols

            df_train = emo['train'].to_pandas()
            df_train["idx"] = df_train.index

            df_dev = emo['validation'].to_pandas()
            df_dev["idx"] = df_dev.index

            df_test = emo['test'].to_pandas()
            df_test["idx"] = df_test.index

        self.raw_dataset = {"train": df_train,
                            "validation": df_dev, "test": df_test, "name": name}
        self.labels = labels_cols if name == "go_emotions" else labels_
        self.div = division
        self.num_examples_per_label = num_examples_per_label

    def get_class_distribution(self) -> None:
        train_data = self.raw_dataset["train"]

        print("Training set data volume: ")
        print(len(train_data))

        max_num = 0

        if self.raw_dataset["name"] == "emotion":
            for i in range(len(self.labels)):
                cur_num = len(train_data[train_data["label"] == i])
                if max_num < cur_num:
                    max_num = cur_num

        labels_dict = {}
        for i in range(len(self.labels)):
            labels_dict[i] = len(train_data[train_data["label"] == i])/max_num

        print("Class distribution in training data: ")
        print(labels_dict)

        return labels_dict

    def process_data(self,) -> dict:
        # 'full_data', 'fewshot', 'DA'
        dataset = {}
        for split in ['train', 'validation', 'test']:
            dataset[split] = []
            for _, data in self.raw_dataset[split].iterrows():
                input_example = InputExample(
                    text_a=data["text"], label=data["label"], guid=data["idx"])
                dataset[split].append(input_example)

        if self.div == "fewshot":
            support_sampler = FewShotSampler(
                num_examples_per_label=self.num_examples_per_label, also_sample_dev=False)
            dataset['support'] = support_sampler(dataset['train'], seed=42)

        return dataset, self.labels


