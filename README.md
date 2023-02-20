# TARA
The code of research paper [Distinguishability Calibration to In-Context Learning](https://arxiv.org/abs/2302.06198).

This paper has been accepted at the The 17th Conference of the European Chapter of the Association for Computational Linguistics(EACL 2023 findings).

We currently provide a fully operational version of .ipynb.

## Requirements
- pytorch-metric-learning>=1.6.3
- numpy
- transformers>=4.10.0
- sentencepiece
- tqdm
- pytorch>=1.10.0
- nltk
- datasets
- rouge==1.0.0

### OpenPromptForTARA install 
OpenPromptForTARA is a variant version for TARA based on [OpenPrompt](
OpenPrompt). Thanks for their development contribution.
```bash
cd OpenPromptForTARA
pip install -r requirements.txt
python setup.py install
```

## Train
```bash
sh train.sh conf/your_train.conf
```

### Configuration
```conf
# training_type
training_type = "mix"

# prompt_type
template_type = "ptuning"

# Data
dataset = "go_emotions"
data_condition = "fewshot"
num_examples_per_label = 50

# zero-shot
few_shot_train = True

# train
model = "BERT"
model_lr = 2e-5
template_lr = 1e-3
epoch_num = 3
batch_size = 8
use_cuda = True

# TML
bank_size = 32
```