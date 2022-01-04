import warnings
import pandas as pd
from transformers import (AutoModelForMaskedLM,
                          AutoTokenizer, LineByLineTextDataset,
                          DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)

warnings.filterwarnings('ignore')

train_data = pd.read_csv('./input/data/train.csv')
test_data = pd.read_csv('./input/data/testa_nolabel.csv')
def concat_text(row):
    return str(row['name']) + ', ' + row['content']


def process_data(data):
    data.fillna("无内容", inplace=True)
    data['text'] = data.apply(lambda row: concat_text(row), axis=1)
    return data


train_data = process_data(train_data)
test_data = process_data(test_data)


#train_data['text'] = train_data['title']
#test_data['text'] = test_data['title']
#data = pd.concat([train_data, test_data])

data=train_data.append(test_data)
data['text'] = data['text'].apply(lambda x: x.replace('\n', ''))

text = '\n'.join(data.text.tolist())

with open('text.txt', 'w', encoding='utf-8') as f:
    f.write(text)

model_name = 'input/bert'
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained('output/pre_bert')

train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="text.txt",  # mention train text file here
    block_size=256)

valid_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="text.txt",  # mention valid text file here
    block_size=256)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.2)

training_args = TrainingArguments(
    output_dir="output/pre_bert",  # select model path for checkpoint
    overwrite_output_dir=True,
    num_train_epochs=5,
    do_eval=True,
    seed=42,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,
    #evaluation_strategy='steps',
    save_total_limit=2,
    eval_steps=200,
    #metric_for_best_model='eval_loss',
    #greater_is_better=False,
    #load_best_model_at_end=True,
    prediction_loss_only=True,
    #report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset)

trainer.train()
trainer.save_model(f'output/pre_bert')
print("ok")