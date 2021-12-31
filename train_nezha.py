import os
import warnings
warnings.simplefilter("ignore")
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
import random
import numpy as np


class Config:
    text_col = 'text'
    target_col = 'label'
    max_len = 256
    batch_size = 2
    target_size = 25
    seed = 42
    n_fold = 5
    learning_rate = 5e-5
    epochs = 10
    warmup_proportion = 0.1
    weight_decay = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    data_path = 'input/data'
    model_path = 'input/nezha-base-www'


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def concat_text(row):
    return "标题:" + str(row['name']) + ',' + row['content']


CFG = Config()
set_seed(CFG.seed)


def process_data():
    train = pd.read_csv(os.path.join(CFG.data_path, 'train.csv'))
    test = pd.read_csv(os.path.join(CFG.data_path, 'testa_nolabel.csv'))
    train.fillna('无内容', inplace=True)
    test.fillna('无内容', inplace=True)
    train['text'] = train.apply(lambda row: concat_text(row), axis=1)
    test['text'] = test.apply(lambda row: concat_text(row), axis=1)
    return train, test






if __name__ == '__main__':
    train, test = process_data()
    print(train.head())