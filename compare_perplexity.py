import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

ORIGINAL_MODEL_NAME = "microsoft/phi-4-mini-instruct"
FINE_TUNED_MODEL_PATH = "fine_tuned_phi4_model"
TEST_DATA_PATH = os.path.join("test_dataset", "test_data.json")

def load_test_data(test_data_path):
    print(f"Загрузка тестовой выборки из {test_data_path}...")
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    return test_data

def format_dialog(dialog):
    formatted_text = ""
    for i, turn in enumerate(dialog["dialog"]):
        role = "Human: " if i % 2 == 0 else "Assistant: "
        formatted_text += role + turn + "\n"
    return formatted_text.strip()

def calculate_perplexity(model, tokenizer, text):
    encodings = tokenizer(text, return_tensors="pt")
    
    if torch.cuda.is_available():
        encodings = {k: v.to("cuda") for k, v in encodings.items()}
        model = model.to("cuda")
    
    max_length = min(tokenizer.model_max_length, 2048) 
    
    input_ids = encodings.input_ids[:, :max_length]
    attention_mask = encodings.attention_mask[:, :max_length] if hasattr(encodings, 'attention_mask') else None
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        
    loss = outputs.loss
    perplexity = torch.exp(loss).item()
    
    return perplexity

def compare_models():
    print("Начало сравнения моделей по метрике perplexity...")
    
    test_data = load_test_data(TEST_DATA_PATH)
    print(f"Загружено {len(test_data)} тестовых примеров")
    
    print(f"Загрузка оригинальной модели {ORIGINAL_MODEL_NAME}...")
    original_tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL_NAME)
    original_model = AutoModelForCausalLM.from_pretrained(
        ORIGINAL_MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    print(f"Загрузка дообученной модели из {FINE_TUNED_MODEL_PATH}...")
    fine_tuned_tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_PATH)
    fine_tuned_model = AutoModelForCausalLM.from_pretrained(
        FINE_TUNED_MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    original_perplexities = []
    fine_tuned_perplexities = []
    
    print("Расчет perplexity для каждого диалога в тестовой выборке...")
    for dialog in tqdm(test_data):
        formatted_text = format_dialog(dialog)
        
        original_ppl = calculate_perplexity(original_model, original_tokenizer, formatted_text)
        original_perplexities.append(original_ppl)
        
        fine_tuned_ppl = calculate_perplexity(fine_tuned_model, fine_tuned_tokenizer, formatted_text)
        fine_tuned_perplexities.append(fine_tuned_ppl)
    
    avg_original_ppl = np.mean(original_perplexities)
    avg_fine_tuned_ppl = np.mean(fine_tuned_perplexities)
    
    print("\nРезультаты сравнения моделей по метрике perplexity:")
    print(f"Оригинальная модель ({ORIGINAL_MODEL_NAME}): {avg_original_ppl:.4f}")
    print(f"Дообученная модель: {avg_fine_tuned_ppl:.4f}")
    print(f"Разница (оригинальная - дообученная): {avg_original_ppl - avg_fine_tuned_ppl:.4f}")
    
    if avg_fine_tuned_ppl < avg_original_ppl:
        print("Дообученная модель показывает лучшие результаты (меньшее значение perplexity).")
    else:
        print("Оригинальная модель показывает лучшие результаты (меньшее значение perplexity).")
    
    results = {
        "original_model": {
            "name": ORIGINAL_MODEL_NAME,
            "avg_perplexity": float(avg_original_ppl),
            # "perplexities": [float(p) for p in original_perplexities]
        },
        "fine_tuned_model": {
            "name": FINE_TUNED_MODEL_PATH,
            "avg_perplexity": float(avg_fine_tuned_ppl),
            # "perplexities": [float(p) for p in fine_tuned_perplexities]
        }
    }
    
    with open("results/perplexity_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("Результаты сохранены в файл perplexity_results.json")

if __name__ == "__main__":
    compare_models()
