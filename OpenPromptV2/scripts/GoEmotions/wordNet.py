'''
Author: Hong Jing Li
Date: 2022-06-27 17:06:30
LastEditors: Hong Jing Li
LastEditTime: 2022-06-27 18:39:21
Contact: hongjing.li.work@gmail.com
'''
from nltk.corpus import wordnet

word_lst = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
            'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']

synonyms_double_lst = {}
synonyms_lst = []

for i in word_lst:
    for syn in wordnet.synsets(i):
        for lm in syn.lemmas():
            synonyms_lst.append(lm.name())
    synonyms_double_lst[i]=list(set(synonyms_lst))
    synonyms_lst = []

print(synonyms_double_lst)

