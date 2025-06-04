import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

ORIGINAL_MODEL_NAME = "microsoft/phi-4-mini-instruct"
FINE_TUNED_MODEL_PATH = "fine_tuned_phi4_model"
TEST_DATA_PATH = os.path.join("test_dataset", "test_data.json")
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

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

def extract_assistant_responses(dialog):
    responses = []
    for i, turn in enumerate(dialog["dialog"]):
        if i % 2 == 1:
            responses.append(turn)
    return responses

def generate_model_responses(model, tokenizer, dialog):
    responses = []
    
    if torch.cuda.is_available():
        model = model.to("cuda")
    
    for i, turn in enumerate(dialog["dialog"]):
        if i % 2 == 0:
            prompt = f"Human: {turn}\nAssistant:"
            
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
            
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            
            assistant_response = generated_text.split("Assistant:")[-1].strip()
            responses.append(assistant_response)
    
    return responses

def calculate_cosine_similarities(embedding_model, original_responses, fine_tuned_responses, reference_responses):
    print("Создание векторных представлений для ответов...")
    reference_embeddings = embedding_model.encode(reference_responses)
    original_embeddings = embedding_model.encode(original_responses)
    fine_tuned_embeddings = embedding_model.encode(fine_tuned_responses)
    
    original_similarities = []
    fine_tuned_similarities = []
    
    for i in range(len(reference_responses)):
        original_sim = cosine_similarity(
            [reference_embeddings[i]], 
            [original_embeddings[i]]
        )[0][0]
        original_similarities.append(original_sim)
        
        fine_tuned_sim = cosine_similarity(
            [reference_embeddings[i]], 
            [fine_tuned_embeddings[i]]
        )[0][0]
        fine_tuned_similarities.append(fine_tuned_sim)
    
    return original_similarities, fine_tuned_similarities

def compare_models():
    print("Начало сравнения моделей по метрике cosine similarity...")
    
    test_data = load_test_data(TEST_DATA_PATH)
    print(f"Загружено {len(test_data)} тестовых примеров")
    
    print(f"Загрузка модели для векторных представлений: {EMBEDDING_MODEL_NAME}...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
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
    
    all_original_similarities = []
    all_fine_tuned_similarities = []
    
    print("Обработка диалогов и расчет косинусного сходства...")
    for dialog in tqdm(test_data):
        reference_responses = extract_assistant_responses(dialog)
        
        original_responses = generate_model_responses(original_model, original_tokenizer, dialog)
        
        fine_tuned_responses = generate_model_responses(fine_tuned_model, fine_tuned_tokenizer, dialog)
        
        original_similarities, fine_tuned_similarities = calculate_cosine_similarities(
            embedding_model, 
            original_responses, 
            fine_tuned_responses, 
            reference_responses
        )
        
        all_original_similarities.extend(original_similarities)
        all_fine_tuned_similarities.extend(fine_tuned_similarities)
    
    avg_original_similarity = np.mean(all_original_similarities)
    avg_fine_tuned_similarity = np.mean(all_fine_tuned_similarities)
    
    print("\nРезультаты сравнения моделей по метрике cosine similarity:")
    print(f"Оригинальная модель ({ORIGINAL_MODEL_NAME}): {avg_original_similarity:.4f}")
    print(f"Дообученная модель: {avg_fine_tuned_similarity:.4f}")
    print(f"Разница (дообученная - оригинальная): {avg_fine_tuned_similarity - avg_original_similarity:.4f}")
    
    if avg_fine_tuned_similarity > avg_original_similarity:
        print("Дообученная модель показывает лучшие результаты (большее значение cosine similarity).")
    else:
        print("Оригинальная модель показывает лучшие результаты (большее значение cosine similarity).")
    
    results = {
        "original_model": {
            "name": ORIGINAL_MODEL_NAME,
            "avg_similarity": float(avg_original_similarity),
            "similarities": [float(s) for s in all_original_similarities]
        },
        "fine_tuned_model": {
            "name": FINE_TUNED_MODEL_PATH,
            "avg_similarity": float(avg_fine_tuned_similarity),
            "similarities": [float(s) for s in all_fine_tuned_similarities]
        }
    }
    
    with open("results/cosine_similarity_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("Результаты сохранены в файл cosine_similarity_results.json")

if __name__ == "__main__":
    compare_models()
