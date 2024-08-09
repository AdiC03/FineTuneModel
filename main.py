# Licenses:
'''
Dataset:
McAuley, Julian, and Jure Leskovec. "Hidden factors and hidden topics: understanding rating dimensions with review text." In Proceedings of the 7th ACM conference on Recommender systems, pp. 165-172. 2013.

Xiang Zhang, Junbo Zhao, Yann LeCun. Character-level Convolutional Networks for Text Classification. Advances in Neural Information Processing Systems 28 (NIPS 2015)
'''

'''
GPT2 Model:
@article{radford2019language,
  title={Language Models are Unsupervised Multitask Learners},
  author={Radford, Alec and Wu, Jeff and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  year={2019}
}
'''

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, TaskType, PeftConfig, PeftModel

model = AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=2,
                                                           id2label = {0: "negative", 1: "positive"},
                                                           label2id = {"negative": 0, "positive": 1})


tokenizer = AutoTokenizer.from_pretrained("gpt2")
model.config.pad_token_id = model.config.eos_token_id

dataset = load_dataset("fancyzhx/amazon_polarity", split="train[:4000]", trust_remote_code=True)
test_dataset = load_dataset("fancyzhx/amazon_polarity", split="test[:1000]", trust_remote_code=True)

tokenizer.pad_token = tokenizer.eos_token

def preprocess_func(example):
    return tokenizer(example["title"], padding="max_length", truncation=True)

tokenized_ds = dataset.map(preprocess_func, batched=True)
tokenized_test_ds = test_dataset.map(preprocess_func, batched=True)

for param in model.parameters():
    param.requires_grad = True

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    learning_rate=2e-3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    eval_dataset=tokenized_test_ds,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

gpt_eval = trainer.evaluate()

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

peft_model = get_peft_model(model, peft_config)

lora_training_args = TrainingArguments(
    output_dir='./peft_results',
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    learning_rate=2e-3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

lora_trainer = Trainer(
    model=peft_model,
    args=lora_training_args,
    train_dataset=tokenized_ds,
    eval_dataset=tokenized_test_ds,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

lora_trainer.train()

peft_model.save_pretrained("./peft_model")

saved_model_config = PeftConfig.from_pretrained("./peft_model")
saved_model = AutoModelForSequenceClassification.from_pretrained(saved_model_config.base_model_name_or_path)
saved_peft_model = PeftModel.from_pretrained(saved_model, "./peft_model")

reloaded_lora_training_args = TrainingArguments(
    output_dir='./lora_results_inference',
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    learning_rate=2e-3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

reloaded_lora_trainer = Trainer(
    model=saved_peft_model,
    args=reloaded_lora_training_args,
    train_dataset=tokenized_ds,
    eval_dataset=tokenized_test_ds,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

fine_tune_eval = reloaded_lora_trainer.evaluate()

print("GPT eval: {}", gpt_eval)
print("Reloaded eval:", fine_tune_eval)