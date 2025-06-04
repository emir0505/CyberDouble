import os
import json
import random
import torch
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTTrainingArguments

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

MODEL_SAVE_PATH = "fine_tuned_phi4_model"
TEST_DATA_PATH = "test_dataset"

def load_siberianpersonachat():
    print("Загрузка датасета siberianpersonachat...")
    dataset = load_dataset("IlyaGusev/siberianpersonachat")
    return dataset

def prepare_dataset(dataset):
    print("Подготовка и разделение датасета...")

    all_data = list(dataset['train']) + list(dataset['validation'])
    train_data, temp_data = train_test_split(all_data, test_size=0.3, random_state=SEED)
    validation_data, test_data = train_test_split(temp_data, test_size=1/3, random_state=SEED)

    print(f"Train: {len(train_data)}, Val: {len(validation_data)}, Test: {len(test_data)}")

    os.makedirs(TEST_DATA_PATH, exist_ok=True)
    with open(os.path.join(TEST_DATA_PATH, "test_data.json"), "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    return train_data, validation_data, test_data

def format_dialog(dialog):
    return "\n".join(
        [("Human: " if i % 2 == 0 else "Assistant: ") + turn for i, turn in enumerate(dialog["dialog"])]
    ).strip()

def preprocess(data_list):
    return [{"text": format_dialog(item)} for item in data_list]

def fine_tune_model():
    print("Запуск дообучения с использованием Unsloth + TRL...")

    dataset = load_siberianpersonachat()
    train_data, val_data, test_data = prepare_dataset(dataset)

    train_dataset = Dataset.from_list(preprocess(train_data))
    val_dataset = Dataset.from_list(preprocess(val_data))


    model_name = "microsoft/phi-4-mini-instruct"
    max_seq_length = 512
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True
    )

    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, max_length=max_seq_length)

    tokenized_train = train_dataset.map(tokenize_function, remove_columns=["text"])
    tokenized_val = val_dataset.map(tokenize_function, remove_columns=["text"])

    training_args = SFTTrainingArguments(
        output_dir=MODEL_SAVE_PATH,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=100,
        evaluation_strategy="steps",
        save_steps=1000,
        eval_steps=1000,
        save_total_limit=2,
        load_best_model_at_end=True,
        warmup_steps=100,
        weight_decay=0.01,
        bf16=True,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        dataset_text_field="text",
        max_seq_length=max_seq_length
    )

    trainer.train()

    print(f"Сохраняем модель в {MODEL_SAVE_PATH}...")
    trainer.model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)

    return MODEL_SAVE_PATH, TEST_DATA_PATH

if __name__ == "__main__":
    fine_tune_model()
