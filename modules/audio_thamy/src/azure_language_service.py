
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential


def analyze_text(transcription, language_key, language_endpoint):
    """Analisa uma transcrição utilizando Azure AI Language."""

    if not transcription or not transcription.strip():
        raise ValueError("A transcrição está vazia.")

    if not language_key or not language_endpoint:
        raise ValueError(
            "Credenciais do Azure AI Language incompletas."
        )

    client = TextAnalyticsClient(
        endpoint=language_endpoint,
        credential=AzureKeyCredential(language_key)
    )

    sentiment = client.analyze_sentiment(
        documents=[transcription],
        language="pt"
    )[0]

    phrases = client.extract_key_phrases(
        documents=[transcription],
        language="pt"
    )[0]

    entities = client.recognize_entities(
        documents=[transcription],
        language="pt"
    )[0]

    for result in [sentiment, phrases, entities]:
        if result.is_error:
            raise RuntimeError(
                f"Falha no Azure AI Language: {result.error}"
            )

    return {
        "provider": "azure_ai_language",
        "sentiment": {
            "label": sentiment.sentiment,
            "confidence": {
                "positive": sentiment.confidence_scores.positive,
                "neutral": sentiment.confidence_scores.neutral,
                "negative": sentiment.confidence_scores.negative
            }
        },
        "key_phrases": list(phrases.key_phrases),
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
