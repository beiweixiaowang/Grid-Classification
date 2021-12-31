import os
import warnings
warnings.simplefilter("ignore")
import random
from functools import partial
import numpy as np
import paddle
import paddle as P
import paddle.nn.functional as F
import paddlenlp as ppnlp
import pandas as pd
from paddle.io import Dataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import MapDataset
from paddlenlp.transformers import LinearDecayWithWarmup
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


# =============================== 初始化 ========================
class Config:
    text_col = 'text'
    target_col = 'label'
    # 最大长度大小
    max_len = 256   # len(text) or tokenizer:256覆盖95%
    # 模型运行批处理大小
    batch_size = 2
    target_size = 25
    seed = 71
    n_fold = 5
    # 训练过程中的最大学习率
    learning_rate = 5e-5
    # 训练轮次
    epochs = 10  # 3
    # 学习率预热比例
    warmup_proportion = 0.1
    # 权重衰减系数，类似模型正则项策略，避免模型过拟合
    weight_decay = 0.01
    model_name = "ernie-gram-zh"
    print_freq = 100


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def concat_text(row):
    return str(row['name']) + ',' + row['content']


CFG = Config()
seed_torch(seed=CFG.seed)


# y = train[CFG.target_col]
# class_weight = 'balanced'
# classes = train[CFG.target_col].unique()  # 标签类别
# weight = compute_class_weight(class_weight=class_weight,classes= classes, y=y)


# print(weight)
train = pd.read_csv('input/data/train.csv')
test = pd.read_csv('input/data/testa_nolabel.csv')
train.fillna('', inplace=True)
test.fillna('', inplace=True)
train['text'] = train.apply(lambda row: concat_text(row), axis=1)
test['text'] = test.apply(lambda row: concat_text(row), axis=1)

# CV split：5折 StratifiedKFold 分层采样
folds = train.copy()
Fold = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[CFG.target_col])):
    folds.loc[val_index, 'fold'] = int(n)
folds['fold'] = folds['fold'].astype(int)


# ====================================== 数据集以及转换函数==============================
# Torch
class CustomDataset(Dataset):

    def __init__(self, df):
        super().__init__()
        self.data = df.values.tolist()
        self.texts = df[CFG.text_col]
        self.labels = df[CFG.target_col]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """
        索引数据
        :param idx:
        :return:
        """
        text = str(self.texts[idx])
        label = self.labels[idx]
        example = {'text': text, 'label': label}

        return example


def convert_example(example, tokenizer, max_seq_length=512, is_test=False):
    """
    创建Bert输入
    ::
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
    Returns:
        input_ids(obj:`list[int]`): The list of token ids.
        token_type_ids(obj: `list[int]`): List of sequence pair mask.
        label(obj:`numpy.array`, data type of int64, optional): The input label if not is_test.
    """
    encoded_inputs = tokenizer(text=example["text"], max_seq_len=max_seq_length)
    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        return input_ids, token_type_ids


def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)


# tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained(CFG.model_name)
tokenizer = ppnlp.transformers.ErnieGramTokenizer.from_pretrained(CFG.model_name)

trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=CFG.max_len)
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
    Stack(dtype="int64")  # label
): [data for data in fn(samples)]


# ====================================== 训练、验证与预测函数 ==============================

@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader):
    """
    验证函数
    """
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        logits = model(input_ids, token_type_ids)
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
        accu = metric.accumulate()
    print("eval loss: %.5f, accu: %.5f" % (np.mean(losses), accu))
    model.train()
    metric.reset()
    return accu


def predict(model, data, tokenizer, batch_size=1):
    """
    预测函数
    """
    examples = []
    for text in data:
        input_ids, segment_ids = convert_example(
            text,
            tokenizer,
            max_seq_length=CFG.max_len,
            is_test=True)
        examples.append((input_ids, segment_ids))

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input id
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # segment id
    ): fn(samples)

    # Seperates data into some batches.
    batches = []
    one_batch = []
    for example in examples:
        one_batch.append(example)
        if len(one_batch) == batch_size:
            batches.append(one_batch)
            one_batch = []
    if one_batch:
        # The last batch whose size is less than the config batch_size setting.
        batches.append(one_batch)

    results = []
    model.eval()
    for batch in tqdm(batches):
        input_ids, segment_ids = batchify_fn(batch)
        input_ids = paddle.to_tensor(input_ids)
        segment_ids = paddle.to_tensor(segment_ids)
        logits = model(input_ids, segment_ids)
        probs = F.softmax(logits, axis=1)
        results.append(probs.numpy())
    return np.vstack(results)


def inference():
    output_path = "./output/ernie/"
    model_paths = [
        'ernie-gram-zh_fold0.bin',
        'ernie-gram-zh_fold1.bin',
        'ernie-gram-zh_fold2.bin',
        'ernie-gram-zh_fold3.bin',
        'ernie-gram-zh_fold4.bin',
    ]

    model = ppnlp.transformers.ErnieGramForSequenceClassification.from_pretrained(CFG.model_name,
                                                                                  num_classes=25)
    fold_preds = []
    for model_path in model_paths:
        model.load_dict(P.load(os.path.join(output_path, model_path)))

        pred = predict(model, test.to_dict(orient='records'), tokenizer, 16)

        fold_preds.append(pred)
    preds = np.mean(fold_preds, axis=0) # 五折概率进行平均
    np.save("preds.npy",preds)
    labels = np.argmax(preds, axis=1)
    test['label'] = labels
    test[['id', 'label']].to_csv('paddle.csv', index=None)



def train():
    # ====================================  交叉验证训练 ==========================
    for fold in range(5):
        print(f"===============training fold_nth:{fold + 1}======================")
        trn_idx = folds[folds['fold'] != fold].index
        val_idx = folds[folds['fold'] == fold].index

        train_folds = folds.loc[trn_idx].reset_index(drop=True)
        valid_folds = folds.loc[val_idx].reset_index(drop=True)

        train_dataset = CustomDataset(train_folds)
        train_ds = MapDataset(train_dataset)

        dev_dataset = CustomDataset(valid_folds)
        dev_ds = MapDataset(dev_dataset)

        train_data_loader = create_dataloader(
            train_ds,
            mode='train',
            batch_size=CFG.batch_size,
            batchify_fn=batchify_fn,
            trans_fn=trans_func)
        dev_data_loader = create_dataloader(
            dev_ds,
            mode='dev',
            batch_size=CFG.batch_size,
            batchify_fn=batchify_fn,
            trans_fn=trans_func)

        model = ppnlp.transformers.ErnieGramForSequenceClassification.from_pretrained(CFG.model_name,
                                                                                      num_classes=25)

        num_training_steps = len(train_data_loader) * CFG.epochs
        lr_scheduler = LinearDecayWithWarmup(CFG.learning_rate, num_training_steps, CFG.warmup_proportion)
        optimizer = paddle.optimizer.AdamW(
            learning_rate=lr_scheduler,
            parameters=model.parameters(),
            weight_decay=CFG.weight_decay,
            apply_decay_param_fun=lambda x: x in [
                p.name for n, p in model.named_parameters()
                if not any(nd in n for nd in ["bias", "norm"])
            ])

        criterion = paddle.nn.loss.CrossEntropyLoss()
        metric = paddle.metric.Accuracy()

        global_step = 0
        best_val_acc = 0
        for epoch in range(1, CFG.epochs + 1):
            for step, batch in enumerate(train_data_loader, start=1):
                input_ids, segment_ids, labels = batch
                logits = model(input_ids, segment_ids)
                # probs_ = paddle.to_tensor(logits, dtype="float64")
                loss = criterion(logits, labels)
                probs = F.softmax(logits, axis=1)
                correct = metric.compute(probs, labels)
                metric.update(correct)
                acc = metric.accumulate()

                global_step += 1
                if global_step % CFG.print_freq == 0:
                    print("global step %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f" % (
                        global_step, epoch, step, loss, acc))
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()
            acc = evaluate(model, criterion, metric, dev_data_loader)
            if acc > best_val_acc:
                best_val_acc = acc
                P.save(model.state_dict(), f'./output/ernie/{CFG.model_name}_fold{fold}.bin')
            print('Best Val acc %.5f' % best_val_acc)
        del model


if __name__ == '__main__':
    train()
    inference()


# Focalloss
# class_weights
# ernie>chinese_roberta_wwm
# nezha