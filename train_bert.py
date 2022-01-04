import pandas as pd
import numpy as np
import warnings
warnings.simplefilter("ignore")
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn.metrics import *
from sklearn.model_selection import *
import random
import torch
pd.options.display.max_colwidth = 200


class Args:
    model_name = 'bert'
    model_path = 'hfl/chinese-bert-wwm-ext'
    n_fold = 5
    seed = 42
    batch_size = 2
    learning_rate = 4e-5
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


def train_model(args, train, test):
    err = []
    acc = []
    y_pred_tot = []
    # id_list = test['id'].tolists()

    train_data = train[['text', 'label']]
    test_data = test[['text']]

    fold = StratifiedKFold(n_splits=args.n_fold, shuffle=True, random_state=args.seed)
    model_args = ClassificationArgs(num_train_epochs=args.epochs,
                                    train_batch_size=args.batch_size,
                                    reprocess_input_data=True,
                                    overwrite_output_dir=True,
                                    warmup_ratio=args.warmup_ratio,
                                    warmup_steps=args.warmup_steps,
                                    weight_decay=args.weight_decay,
                                    save_model_every_epoch=False,
                                    save_eval_checkpoints=False
                                    )
    for train_index, test_index in fold.split(train_data, train_data['label']):
        train_trn, train_val = train_data.iloc[train_index], train_data.iloc[test_index]
        model = ClassificationModel(args.model_name, args.model_path, use_cuda=True,
                                    num_labels=args.nums_labels, args=model_args)
        model.train_model(train_trn)
        result, raw_outputs_val, wrong_predictions = model.eval_model(train_val, acc=accuracy_score)
        print(result)
        err.append(result['eval_loss'])
        acc.append(result['acc'])
        raw_outputs_test = model.predict(test_data['text'].tolist())[1]
        y_pred_tot.append(raw_outputs_test)
    print("Mean LogLoss: ", np.mean(err))
    print("Mean acc:", np.mean(acc))
    final = pd.DataFrame(np.mean(y_pred_tot, 0))
    final.insert(0, 'id', test['id'].tolist())
    print(final.shape)
    return final


def main():
    Args.model_name = 'roberta'
    Args.model_path = 'hfl/chinese-roberta-wwm-ext'
    train = pd.read_csv("input/data/train.csv")
    test = pd.read_csv("input/data/testa_nolabel.csv")
    train = process_data(train)
    test = process_data(test)
    print(train.shape, test.shape)
    result = train_model(Args, train, test)
    print(result.shape)
    result.to_csv("robert.csv", index=False)


if __name__ == '__main__':
    main()
