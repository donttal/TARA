'''
Author: Hong Jing Li
Date: 2022-06-27 16:26:19
LastEditors: Hong Jing Li
LastEditTime: 2022-06-27 20:59:45
Contact: hongjing.li.work@gmail.com
'''
import re
from tqdm import tqdm
from BackTranslation import BackTranslation

label_cols = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
              'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']


trans = BackTranslation(url=[
    'translate.google.com',
    'translate.google.co.kr',
], proxies={'http': '127.0.0.1:1234', 'http://host.name': '127.0.0.1:4012'})

# languages = ['af', 'sq', 'am', 'hy', 'az', 'eu', 'be', 'bn', 'bs',
#              'bg', 'ca', 'ceb', 'ny', 'zh-cn', 'zh-tw', 'co',
#              'hr', 'cs', 'da', 'nl', 'eo', 'et', 'tl', 'fi',
#              'fr', 'fy', 'gl', 'ka', 'de', 'el', 'gu', 'ht', 'ha',
#              'haw', 'he', 'hi', 'hmn', 'hu', 'is', 'ig', 'id',
#              'ga', 'it', 'ja', 'jw', 'kn', 'kk', 'ko', 'ku', 'ky', 'lo',
#              'la', 'lv', 'lb', 'mk', 'mg', 'ms', 'ml', 'mt', 'mi', 'mr',
#              'mn', 'my', 'ne', 'no', 'or', 'ps', 'fa', 'pl', 'pt', 'pa',
#              'ro', 'ru', 'sm', 'gd', 'sr', 'st', 'sn', 'sd', 'si', 'sk', 'sl',
#              'so', 'es', 'su', 'sw', 'sv', 'tg', 'ta', 'te', 'th', 'tr',
#              'ur', 'uk', 'ug', 'uz', 'vi', 'cy', 'xh', 'yi', 'yo', 'zu',
#              'ar', 'es']

languages = ['fr', 'th', 'tr', 'ur', 'ru', 'bg', 'de', 'ar', 'zh-cn', 'hi',
             'sw', 'vi', 'es', 'el']

# labels = "realization", "surprise", "curiosity", "confusion"
label_words = {}
for i in tqdm(label_cols):
    #print(i + "\n")
    new_words = set()
    for j in languages:
        try:
            # print(j)
            result = trans.translate(i, src='en', tmp=j)
            new_word = result.result_text.lower()
            new_word = re.sub(r'[\W\s]', '', new_word)
            # print(new_word)
            new_words.add(new_word)
        except NameError:
            print(j+"can not as translate target")
        except TypeError:
            print(j+"can not as translate target")
    # print(new_words)
    label_words[i] = list(new_words)
    # print("\n")
print(label_words)
