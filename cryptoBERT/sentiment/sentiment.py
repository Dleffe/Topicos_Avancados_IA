import json
import os
import torch
from transformers import pipeline

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def analyze_sentiment(input_file: str, output_file: str):

    print(f"\nCarregando o arquivo {input_file} para análise...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Arquivo {input_file} não encontrado.")
        return

    if not data:
        print("Arquivo JSON está vazio. Nenhuma análise para fazer.")
        return

    print("Carregando o modelo CryptoBERT (Isso pode demorar um pouco na primeira execução para baixar o modelo)...")
    
    sentiment_pipeline = pipeline(
        "text-classification", 
        model="ElKulako/cryptobert", 
        truncation=True, 
        max_length=512
    )

    print(f"Iniciando inferência matemática em {len(data)} textos...")
    
    platform_weights = {
        "Telegram": 1.2,
        "Google News": 1.0,
        "Reddit": 0.8
    }
    
    total_weighted_score = 0.0
    total_weights = 0.0
    
    sentiment_counts = {"Bullish": 0, "Bearish": 0, "Neutral": 0}
    
    for item in data:
        text = item.get("Content", "")
        platform = item.get("Platform", "Google News")
        weight = platform_weights.get(platform, 1.0)
        
        if not text:
            item["Sentiment"] = "Unknown"
            item["Confidence"] = 0.0
            item["Calculated_Score"] = 0.0
            continue
            
        try:
            result = sentiment_pipeline(text)[0]
            label = result["label"]
            confidence = result["score"]
            
            item["Sentiment"] = label
            item["Confidence"] = round(confidence, 4)
            
            direction = 0.0
            if label == "Bullish":
                direction = 1.0
                sentiment_counts["Bullish"] += 1
            elif label == "Bearish":
                direction = -1.0
                sentiment_counts["Bearish"] += 1
            else:
                sentiment_counts["Neutral"] += 1
                
            item_score = direction * confidence * weight
            item["Calculated_Score"] = round(item_score, 4)
            
            total_weighted_score += item_score
            total_weights += weight
                
        except Exception as e:
            item["Sentiment"] = "Error"
            item["Confidence"] = 0.0
            item["Calculated_Score"] = 0.0
            print(f"Erro ao analisar o texto: {e}")
            
    macro_index = 0.0
    if total_weights > 0:
        macro_index = total_weighted_score / total_weights
        
    macro_index = round(macro_index, 4)
    
    final_output = {
        "Macro_Sentiment_Index": macro_index,
        "Total_Analyzed": len(data),
        "Breakdown": {
            "Bullish_Count": sentiment_counts["Bullish"],
            "Bearish_Count": sentiment_counts["Bearish"],
            "Neutral_Count": sentiment_counts["Neutral"]
        },
        "Data": data
    }
            
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
        
    print(f"\n" + "="*50)
    print(f"🧠 MACRO SENTIMENT INDEX (Neuro-Fuzzy Input): {macro_index:+.4f}")
    print(f"   [-1.0 = Max Bearish  |  +1.0 = Max Bullish]")
    print("="*50 + "\n")
    
    print(f"Análise finalizada com sucesso! Salvo em: {output_file}")

if __name__ == "__main__":
    analyze_sentiment("output_bitcoin.json", "sentiment_bitcoin.json")
