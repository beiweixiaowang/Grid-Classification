import pandas as pd
import numpy as np
import warnings
warnings.simplefilter("ignore")
from scipy.special import softmax
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from tqdm.notebook import tqdm
from sklearn.metrics import *
from sklearn.model_selection import *
import re
import random
import torch
pd.options.display.max_colwidth = 200


class Args:
    model_name = 'bert'
    model_path = 'hfl/chinese-bert-wwm-ext'
    n_fold = 5
    seed = 42
    batch_size = 2
    learning_rate = 5e-5
    max_len = 256
    nums_labels = 25
    epochs = 10
    weight_decay = 0.01
    warmup_ratio = 0.1
    warmup_steps = 0







def seed_all(seed_value):
    random.seed(seed_value)  # Python
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


def concat_text(row):
    return str(row['name']) + ', ' + row['content']


def process_data(data):
    data.fillna("无内容", inplace=True)
    data['text'] = data.apply(lambda row: concat_text(row), axis=1)
    return data


def f1_multiclass(labels, preds):
    return f1_score(labels, preds, average='micro')


def train_model(args):
    err = []
    acc = []
    y_pred_tot = []
    fold = StratifiedKFold(n_splits=args.n_fold, shuffle=True, random_state=args.seed)
    i = 1
    model_args = ClassificationArgs(num_train_epochs=args.epochs,
                                    train_batch_size=args.batch_size,
                                    reprocess_input_data=True,
                                    overwrite_output_dir=True,
                                    warmup_ratio=args.warmup_ratio,
                                    warmup_steps=args.warmup_steps,
                                    weight_decay=args.weight_decay

                                    )


if __name__ == '__main__':

