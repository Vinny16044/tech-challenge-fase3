
from pathlib import Path
import json
import os

from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / ".env")

LANGUAGE_KEY = os.getenv("AZURE_LANGUAGE_KEY")
LANGUAGE_ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT")

if not LANGUAGE_KEY or not LANGUAGE_ENDPOINT:
    raise ValueError(
        "Configure AZURE_LANGUAGE_KEY e "
        "AZURE_LANGUAGE_ENDPOINT no arquivo .env"
    )

client = TextAnalyticsClient(
    endpoint=LANGUAGE_ENDPOINT,
    credential=AzureKeyCredential(LANGUAGE_KEY)
)

# Texto fictício para testar o contexto clínico
text = (
    "Estou sentindo muito cansaço e dificuldade para respirar. "
    "Também tenho tosse persistente e estou preocupado "
    "com a minha saúde."
)

print("Enviando texto para Azure AI Language...")

documents = [text]

sentiment = client.analyze_sentiment(
    documents=documents,
    language="pt"
)[0]

key_phrases = client.extract_key_phrases(
    documents=documents,
    language="pt"
)[0]

entities = client.recognize_entities(
    documents=documents,
    language="pt"
)[0]

for result in [sentiment, key_phrases, entities]:
    if result.is_error:
        raise RuntimeError(
            f"Erro Azure Language: {result.error}"
        )

output = {
    "source": "synthetic_clinical_text",
    "language": "pt",
    "text": text,
    "sentiment": {
        "label": sentiment.sentiment,
        "confidence": {
            "positive": sentiment.confidence_scores.positive,
            "neutral": sentiment.confidence_scores.neutral,
            "negative": sentiment.confidence_scores.negative
        }
    },
    "key_phrases": list(key_phrases.key_phrases),
    "entities": [
        {
            "text": entity.text,
            "category": entity.category,
            "subcategory": entity.subcategory,
            "confidence": entity.confidence_score
        }
        for entity in entities.entities
    ]
}

output_path = OUTPUT_DIR / "language_analysis.json"

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(
        output,
        file,
        ensure_ascii=False,
        indent=4
    )

print("\nSentimento:", output["sentiment"]["label"])
print("Frases-chave:", output["key_phrases"])
print("Entidades:", output["entities"])
print("\nResultado salvo em:", output_path)
