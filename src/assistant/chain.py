from __future__ import annotations

import argparse
import json
from typing import Any

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from src.assistant.grounding import build_verified_facts
from src.assistant.ollama_llm import OllamaMedicalLLM
from src.assistant.prontuario_repository import (
    ProntuarioRepository,
)
from src.assistant.protocol_retriever import (
    ProtocolRetriever,
)
from src.finetune.inference import audit_log


CONTEXT_PROMPT = PromptTemplate.from_template(
    """
Você é um assistente de apoio clínico para oncologia,
especializado em fornecer informações baseadas em protocolos
internos e fatos verificados do prontuário do paciente.

Responda usando SOMENTE os fatos verificados e os protocolos
fornecidos abaixo.

Os FATOS VERIFICADOS foram obtidos diretamente do banco de dados
e possuem prioridade sobre qualquer interpretação do modelo.

REGRAS OBRIGATÓRIAS:

1. Não invente dados.
2. Não transforme um alerta em exame.
3. Não classifique exame realizado como pendente.
4. Não diga que determinado exame está pendente há mais de
30 dias se essa identificação não estiver explícita nos fatos.
5. Se existir alerta de exame pendente há mais de 30 dias,
informe obrigatoriamente:
- a prioridade do alerta;
- o código do protocolo associado;
- a quantidade de exames mencionada no alerta;
- e, se o alerta não identificar qual exame originou a condição,
     escreva explicitamente:
     "Os dados disponíveis não identificam qual exame pendente
     originou especificamente este alerta."
6. Cite os códigos dos protocolos utilizados.
7. Não forneça diagnóstico definitivo.
8. Não forneça prescrição médica autônoma.
9. Caso a informação solicitada não conste dos FATOS VERIFICADOS
nem do PRONTUÁRIO (ex.: medicações em uso, resultados não
registrados), comece a resposta exatamente com:
"Não há informações disponíveis sobre <o que foi solicitado>
no prontuário." — e não acrescente suposições sobre o dado ausente.
10. A decisão clínica final pertence ao médico responsável.
11. Responda todas as partes da pergunta do usuário.
12. Não cite protocolos irrelevantes apenas porque foram
    recuperados pelo sistema.
13. Seja CONCISO: no máximo dois parágrafos curtos. Nunca repita
    a mesma informação e nunca invente exemplos, datas ou nomes.
14. Se a pergunta for uma saudação ou pedir uma visão geral do
    paciente, responda em 2-4 frases com um resumo objetivo dos
    dados do PRONTUÁRIO (idade, estágio, pendências e alertas),
    sem listar protocolos.

PERGUNTA:
{question}

FATOS VERIFICADOS:
{verified_facts}

PRONTUÁRIO COMPLETO:
{patient_context}

PROTOCOLOS INTERNOS RECUPERADOS:
{protocol_context}

FONTES:
{sources}

INSTRUÇÕES ESPECÍFICAS:

Quando a pergunta envolver exames pendentes:

1. Informe primeiro, de forma objetiva, somente os exames
   marcados como pendentes nos FATOS VERIFICADOS.

2. Informe qual protocolo está associado a esses exames.

3. Explique objetivamente o que esse protocolo determina
   para a situação apresentada.

4. Se houver alerta registrado, informe o alerta em uma seção
   separada. Nunca trate um alerta como exame.

5. Se o alerta informar que existe exame pendente há mais de
   30 dias, mas não identificar qual exame, deixe isso explícito.

6. Não omita nenhuma parte da pergunta.

Responda em português do Brasil, de forma clara, objetiva e
baseada exclusivamente nas informações fornecidas.
""".strip()
)


class MedicalAssistantChain:
    """
    Pipeline LangChain principal.
    Responsável: Paola
    """

    def __init__(
        self,
        offline: bool = False,
        model_name: str = "llama3.2:3b",
    ) -> None:
        self.offline = offline
        self.model_name = (
            "offline"
            if offline
            else model_name
        )

        self.protocol_retriever = ProtocolRetriever(
            k=3,
        )

        self.prontuario_repository = (
            ProntuarioRepository()
        )

        self.llm: OllamaMedicalLLM | None = None

        if not self.offline:
            self.llm = OllamaMedicalLLM(
                model_name=model_name,
            )

        self.chain = (
            RunnableLambda(
                self._retrieve_context
            )
            | RunnableLambda(
                self._generate_answer
            )
        )

    def _retrieve_context(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        question = str(
            payload.get("question", "")
        ).strip()

        paciente_id = payload.get(
            "paciente_id"
        )

        if not question:
            raise ValueError(
                "A pergunta não pode ser vazia."
            )

        documents = (
            self.protocol_retriever.search(
                question
            )
        )

        sources = sorted(
            {
                str(
                    document.metadata["id"]
                )
                for document in documents
            }
        )

        protocol_context = "\n\n".join(
            (
                f"[{document.metadata['id']}] "
                f"{document.metadata['titulo']}\n"
                f"{document.page_content}"
            )
            for document in documents
        )

        patient_context = (
            self.prontuario_repository
            .build_context(
                paciente_id
            )
        )

        verified_facts = (
            build_verified_facts(
                self.prontuario_repository,
                paciente_id,
            )
        )

        result = {
            "question": question,
            "paciente_id": paciente_id,
            "patient_context": patient_context,
            "protocol_context": protocol_context,
            "sources": sources,
            "model_name": self.model_name,
            "verified_facts": verified_facts,
        }

        audit_log(
            {
                "etapa": (
                    "langchain_retrieval"
                ),
                "paciente_id": paciente_id,
                "pergunta": question,
                "fontes_recuperadas": sources,
                "fatos_verificados": (verified_facts
                                    ),
            }
        )

        return result

    def _generate_answer(
        self,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        verified_facts = context[
            "verified_facts"
        ]


        prompt = CONTEXT_PROMPT.format(
            question=context["question"],
            verified_facts=(
                verified_facts["summary"]
            ),
            patient_context=(
                context[
                    "patient_context"
                ]
            ),
            protocol_context=(
                context[
                    "protocol_context"
                ]
            ),
            sources=(
                ", ".join(
                    context["sources"]
                )
                or "Nenhuma fonte recuperada"
            ),
        )

        if self.offline:
            answer = (
                "[MODO OFFLINE]\n"
                "Pipeline LangChain executado "
                "com sucesso.\n\n"
                f"Paciente: "
                f"{context['paciente_id'] or 'não informado'}\n"
                "Fontes recuperadas: "
                f"{', '.join(context['sources']) or 'nenhuma'}\n\n"
            )

        else:
            if self.llm is None:
                raise RuntimeError(
                    "LLM não inicializada."
                )

            answer = self.llm.invoke(
                prompt
            )

        audit_log(
            {
                "etapa": (
                    "langchain_resposta"
                ),
                "paciente_id": (
                    context["paciente_id"]
                ),
                "modelo": self.model_name,
                "fontes_recuperadas": (
                    context["sources"]
                ),
                "fatos_verificados": (
                    verified_facts),
                "resposta": answer,
            }
        )

        return {
            **context,
            "prompt": prompt,
            "answer": answer,
        }

    def invoke(
        self,
        question: str,
        paciente_id: str | None = None,
    ) -> dict[str, Any]:

        return self.chain.invoke(
            {
                "question": question,
                "paciente_id": paciente_id,
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Assistente médico — LangChain"
        )
    )

    parser.add_argument(
        "--question",
        "-q",
        required=True,
        help="Pergunta clínica.",
    )

    parser.add_argument(
        "--paciente-id",
        "-p",
        default=None,
        help=(
            "ID sintético do paciente. "
            "Exemplo: P0001"
        ),
    )

    parser.add_argument(
        "--offline",
        action="store_true",
    )

    parser.add_argument(
        "--model",
        default="llama3.2:3b",
        help=(
            "Modelo disponível no Ollama. "
            "Padrão: llama3.2:3b"
        ),
    )

    args = parser.parse_args()

    assistant = MedicalAssistantChain(
        offline=args.offline,
        model_name=args.model,
    )

    result = assistant.invoke(
        question=args.question,
        paciente_id=args.paciente_id,
    )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()