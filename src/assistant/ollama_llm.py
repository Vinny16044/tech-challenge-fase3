from __future__ import annotations

from langchain_ollama import ChatOllama

from src.finetune.prompts import SYSTEM_ASSISTENTE


class OllamaMedicalLLM:
    """
    Wrapper para execução do modelo médico local via Ollama.

    Responsável: Paola
    """

    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        temperature: float = 0.0,
        num_predict: int = 300,
    ) -> None:
        self.model_name = model_name

        self.client = ChatOllama(
            model=model_name,
            temperature=temperature,
            num_predict=num_predict,
            repeat_penalty=1.15,  # evita degeneração/loops do modelo 3B
            validate_model_on_init=True,
        )

    def invoke(
        self,
        prompt: str,
    ) -> str:
        """
        Envia o contexto clínico para a LLM local.
        """

        response = self.client.invoke(
            [
                (
                    "system",
                    SYSTEM_ASSISTENTE,
                ),
                (
                    "human",
                    prompt,
                ),
            ]
        )

        content = response.content

        if isinstance(content, str):
            return content.strip()

        return str(content).strip()