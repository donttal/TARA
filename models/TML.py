
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class TML(object):

    def __init__(self, dataset, labels) -> None:
        self.labels = labels
        self.Hierarchical_class_high = {}
        self.Hierarchical_class_mid  = {}
        self.Hierarchical_class = {}
        if dataset == 'go_emotions':
            Hierarchical_class_high = {
            0: ["amusement", "excitement", "joy", "love", "desire", "optimism", "caring", "pride", "admiration", "gratitude", "relief", "approval"],
            1: ["fear", "nervousness", "remorse", "embarrassment", "disappointment", "sadness", "grief", "disgust", "anger", "annoyance", "disapproval"],
            2: ["realization", "surprise", "curiosity", "confusion"]
            }

            Hierarchical_class_mid = {
            0: ["anger", "annoyance", "disapproval"],
            1: ["disgust"],
            2: ["fear", "nervousness"],
            3: ["joy", "amusement", "approval", "excitement", "gratitude",  "love", "optimism", "relief", "pride", "admiration", "desire", "caring"],
            4: ["sadness", "disappointment", "embarrassment", "grief",  "remorse"],
            5: ["surprise", "realization", "confusion", "curiosity"]
            }

        elif dataset == 'emotion':
            Hierarchical_class = {
            0: ["joy", "love"],
            1: ["fear",  "sadness", "anger"],
            2: ["surprise"],
            } 

        def label_change(self, x):
            for key, item in self.Hierarchical_class.items():
                if self.labels[x] in item:
                    return int(key) 
            return int(len(Hierarchical_class))

        def label_change_mid(x):
            for key, item in self.Hierarchical_class_mid.items():
                if self.labels[x] in item:
                    return int(key) 
            return int(len(Hierarchical_class_mid))

        def label_change_high(x):
            for key, item in Hierarchical_class_high.items():
                if self.labels[x] in item:
                    return int(key) 
        return int(len(Hierarchical_class_high))