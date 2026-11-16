
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.azure_language_service import analyze_text


@pytest.fixture
def mock_language():
    with patch(
        "src.azure_language_service.TextAnalyticsClient"
    ) as client_class:

        client = MagicMock()
        client_class.return_value = client

        sentiment = SimpleNamespace(
            is_error=False,
            sentiment="negative",
            confidence_scores=SimpleNamespace(
                positive=0.1,
                neutral=0.2,
                negative=0.7
            )
        )

        phrases = SimpleNamespace(
            is_error=False,
            key_phrases=["tosse", "cansaço"]
        )

        entities = SimpleNamespace(
            is_error=False,
            entities=[
                SimpleNamespace(
                    text="tosse",
                    category="Event",
                    subcategory=None,
                    confidence_score=0.8
                )
            ]
        )

        client.analyze_sentiment.return_value = [sentiment]
        client.extract_key_phrases.return_value = [phrases]
        client.recognize_entities.return_value = [entities]

        yield client


def test_analise_sucesso(mock_language):
    result = analyze_text(
        "Estou com tosse e cansaço.",
        "chave_simulada",
        "https://exemplo.cognitiveservices.azure.com/"
    )

    assert result["provider"] == "azure_ai_language"
    assert result["sentiment"]["label"] == "negative"
    assert "tosse" in result["key_phrases"]
    assert len(result["entities"]) == 1


def test_transcricao_vazia():
    with pytest.raises(ValueError, match="vazia"):
        analyze_text(
            "",
            "chave_simulada",
            "https://exemplo.cognitiveservices.azure.com/"
        )


def test_credenciais_incompletas():
    with pytest.raises(ValueError, match="Credenciais"):
        analyze_text(
            "Texto de teste.",
            "",
            "https://exemplo.cognitiveservices.azure.com/"
        )


def test_erro_na_analise(mock_language):
    mock_language.analyze_sentiment.return_value = [
        SimpleNamespace(
            is_error=True,
            error="Erro simulado"
        )
    ]

    with pytest.raises(RuntimeError, match="Falha"):
        analyze_text(
            "Texto de teste.",
            "chave_simulada",
            "https://exemplo.cognitiveservices.azure.com/"
        )


def test_falha_de_conexao(mock_language):
    mock_language.analyze_sentiment.side_effect = (
        ConnectionError("Serviço indisponível")
    )

    with pytest.raises(ConnectionError, match="indisponível"):
        analyze_text(
            "Texto de teste.",
            "chave_simulada",
            "https://exemplo.cognitiveservices.azure.com/"
        )
