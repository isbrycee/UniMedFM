import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from transformers import AutoTokenizer, AutoModel

def init_M3D_CLIP_model(model_path): # 
    device = torch.device("cuda") # or cpu

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
        local_files_only=True
    )
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    model = model.to(device=device)

    return tokenizer, model