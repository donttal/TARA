#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import re

import numpy as np
import torch
from datasets import load_dataset
from openprompt.utils.reproduciblity import set_seed


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

set_seeds()
