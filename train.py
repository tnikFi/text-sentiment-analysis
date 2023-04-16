from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
from datasets import load_metric
from sklearn.metrics import accuracy_score, f1_score
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = (AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6).to(device))

emotions = load_dataset('emotion')

def tokenize(batch):
   return tokenizer(batch["text"], padding=True, truncation=True)
 
emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Print the features of the encoded emotions
print(emotions_encoded["train"].features)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

batch_size = 64
logging_steps = len(emotions_encoded["train"]) // batch_size

training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    weight_decay=0.01,
    evaluation_strategy="epoch",
    num_train_epochs=8,
    save_strategy="epoch",
    disable_tqdm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=emotions_encoded["train"],
    eval_dataset=emotions_encoded["validation"],
    compute_metrics=compute_metrics,
)

trainer.train()

# Evaluate the model and print the results
results = trainer.evaluate()
print(results)