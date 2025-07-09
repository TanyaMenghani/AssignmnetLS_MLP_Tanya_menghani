from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import GPT2LMHeadModel
from transformers import Trainer, TrainingArguments

dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')

def tokenize(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize, batched=True)



training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
)

trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset["train"])

trainer.train()
model.save_pretrained("./fine_tuned_model")
eval_dataset = tokenized_datasets["validation"]
results = trainer.evaluate(eval_dataset)
print(f"Perplexity: {results['perplexity']}")
print(f"Top-k Accuracy: {results['eval_accuracy']}")

