from __future__ import annotations

import re
from typing import Any

from src.assistant.grounding import build_verified_facts
from src.assistant.chain import CONTEXT_PROMPT
from src.assistant.ollama_llm import OllamaMedicalLLM
from src.assistant.prontuario_repository import (
    ProntuarioRepository,
)
from src.assistant.protocol_retriever import (
    ProtocolRetriever,
)
from src.finetune.inference import audit_log
from src.graph.state import MedicalAssistantState


class MedicalGraphNodes:
    """
    Implementação dos nós utilizados pelo LangGraph.
    Responsável: Paola
    """

    def __init__(
        self,
        offline: bool = False,
        model_name: str = "llama3.2:3b",
        interactive_review: bool = False,
    ) -> None:
        self.offline = offline
        self.interactive_review = interactive_review

        self.model_name = "offline" if offline else model_name

        self.protocol_retriever = ProtocolRetriever(
            k=3,
        )

        self.prontuario_repository = ProntuarioRepository()

        self.llm: OllamaMedicalLLM | None = None

        if not self.offline:
            self.llm = OllamaMedicalLLM(
                model_name=model_name,
            )

    def validate_input(
        self,
        state: MedicalAssistantState,
    ) -> dict[str, Any]:
        question = str(
            state.get(
                "question",
                "",
            )
        ).strip()

        if not question:
            raise ValueError("A pergunta clínica " "não pode ser vazia.")

        paciente_id = state.get("paciente_id")

        audit_log(
            {
                "etapa": ("langgraph_validate_input"),
                "paciente_id": paciente_id,
                "pergunta": question,
            }
        )

        return {
            "question": question,
            "paciente_id": paciente_id,
            "model_name": self.model_name,
            "status": "INPUT_VALIDATED",
        }

    def load_patient_context(
        self,
        state: MedicalAssistantState,
    ) -> dict[str, Any]:
        """
        Consulta prontuário e extrai fatos clínicos
        verificados diretamente do banco.

        """
        paciente_id = state.get("paciente_id")

        patient_context = self.prontuario_repository.build_context(paciente_id)

        verified_facts = build_verified_facts(self.prontuario_repository, paciente_id)

        audit_log(
            {
                "etapa": ("langgraph_patient_context"),
                "paciente_id": paciente_id,
                "fatos_verificados": (verified_facts),
            }
        )

        return {
            "patient_context": (patient_context),
            "verified_facts": verified_facts,
            "status": "PATIENT_LOADED",
        }

    def retrieve_protocols(
        self,
        state: MedicalAssistantState,
    ) -> dict[str, Any]:
        question = state["question"]

        documents = self.protocol_retriever.search(question)

        sources = sorted({str(document.metadata["id"]) for document in documents})

        protocol_context = "\n\n".join(
            (
                f"[{document.metadata['id']}] "
                f"{document.metadata['titulo']}\n"
                f"{document.page_content}"
            )
            for document in documents
        )

        audit_log(
            {
                "etapa": ("langgraph_protocol_retrieval"),
                "paciente_id": state.get("paciente_id"),
                "fontes_recuperadas": sources,
            }
        )

        return {
            "protocol_context": (protocol_context),
            "sources": sources,
            "status": ("PROTOCOLS_RETRIEVED"),
        }

    def build_prompt(
        self,
        state: MedicalAssistantState,
    ) -> dict[str, Any]:
        """
        Monta o prompt usando fatos verificados,
        prontuário e protocolos recuperados.
        """

        sources = state.get(
            "sources",
            [],
        )

        verified_facts = state.get(
            "verified_facts",
            {},
        )

        verified_summary = verified_facts.get(
            "summary",
            "Nenhum fato clínico verificado disponível.",
        )

        prompt = CONTEXT_PROMPT.format(
            question=state["question"],
            verified_facts=verified_summary,
            patient_context=state.get(
                "patient_context",
                "Nenhum prontuário disponível.",
            ),
            protocol_context=state.get(
                "protocol_context",
                "Nenhum protocolo recuperado.",
            ),
            sources=(", ".join(sources) if sources else "Nenhuma fonte recuperada"),
        )

        audit_log(
            {
                "etapa": "langgraph_build_prompt",
                "paciente_id": state.get("paciente_id"),
                "fontes": sources,
                "fatos_verificados": verified_facts,
            }
        )

        return {
            "prompt": prompt,
            "status": "PROMPT_READY",
        }

    def generate_answer(
        self,
        state: MedicalAssistantState,
    ) -> dict[str, Any]:
        if self.offline:
            sources = state.get(
                "sources",
                [],
            )

            answer = (
                "[MODO OFFLINE]\n"
                "Fluxo LangGraph executado até "
                "a etapa de geração.\n\n"
                f"Paciente: "
                f"{state.get('paciente_id') or 'não informado'}\n"
                "Protocolos recuperados: "
                f"{', '.join(sources) or 'nenhum'}\n\n"
                "A geração pela LLM está "
                "desativada neste teste."
            )

        else:
            if self.llm is None:
                raise RuntimeError("LLM não inicializada.")

            answer = self.llm.invoke(state["prompt"])

        audit_log(
            {
                "etapa": ("langgraph_generate_answer"),
                "paciente_id": state.get("paciente_id"),
                "modelo": self.model_name,
                "resposta": answer,
            }
        )

        return {
            "answer": answer,
            "model_name": self.model_name,
            "status": "ANSWER_GENERATED",
        }

    def apply_grounding_guardrail(
        self,
        state: MedicalAssistantState,
    ) -> dict[str, Any]:
        """
        Garante que informações críticas verificadas
        no banco não sejam omitidas pela LLM.
        """

        answer = str(
            state.get(
                "answer",
                "",
            )
        ).strip()

        verified_facts = state.get(
            "verified_facts",
            {},
        )

        # Se um paciente FOI informado mas não existe no banco,
        # não permitimos que a LLM invente exames, alertas ou
        # dados clínicos. (Sem paciente informado, a pergunta é
        # apenas sobre protocolos — a resposta da LLM é mantida.)
        paciente_informado = bool(state.get("paciente_id"))

        if paciente_informado and not verified_facts.get(
            "patient_found",
            False,
        ):
            paciente_id = str(
                state.get(
                    "paciente_id",
                    "",
                )
            )

            safe_answer = (
                f"Paciente {paciente_id} não encontrado "
                "na base de prontuários.\n\n"
                "Não é possível informar exames pendentes, "
                "alertas ou condutas clínicas para este paciente "
                "porque não existem dados de prontuário associados "
                "ao identificador informado."
            )

            audit_log(
                {
                    "etapa": (
                        "langgraph_patient_not_found_guardrail"
                    ),
                    "paciente_id": paciente_id,
                    "patient_found": False,
                }
            )

            return {
                "answer": safe_answer,
                "status": "GROUNDING_GUARDRAIL_APPLIED",
            }

        answer_lower = answer.lower()

        answer_mentions_alert = (
            "alerta" in answer_lower
            or "prot-onco-005" in answer_lower
            or "mais de 30 dias" in answer_lower
        )

        alerts = verified_facts.get(
            "alerts",
            [],
        )

        for alert in alerts:
            detail = str(
                alert.get(
                    "detalhe",
                    "",
                )
            ).lower()

            if (
                "mais de 30 dias" not in detail
                or not answer_mentions_alert
            ):
                continue

            warning_fragments = [
                "não identificam qual exame pendente originou",
                "não identifica qual exame pendente originou",
                "não é possível identificar qual exame",
            ]

            warning_already_present = any(
                fragment in answer.lower()
                for fragment in warning_fragments
            )

            if not warning_already_present:
                answer += (
                    "\n\n**Observação de segurança:**\n"
                    "Os dados disponíveis não identificam "
                    "qual exame pendente originou "
                    "especificamente este alerta."
                )

        audit_log(
            {
                "etapa": (
                    "langgraph_grounding_guardrail"
                ),
                "paciente_id": state.get(
                    "paciente_id"
                ),
                "resposta_ajustada": answer,
            }
        )

        return {
            "answer": answer,
            "status": "GROUNDING_GUARDRAIL_APPLIED",
        }

    def validate_grounding(
        self,
        state: MedicalAssistantState,
    ) -> dict[str, Any]:
        """
        Compara a resposta da LLM com os fatos
        verificados diretamente no banco.
        """

        answer = str(
            state.get(
                "answer",
                "",
            )
        ).lower()

        question = str(
            state.get(
                "question",
                "",
            )
        ).lower()

        asks_about_pending_exams = (
            "exame" in question
            and (
                "pendente" in question
                or "pendência" in question
            )
        )

        asks_about_alerts = (
            "alerta" in question
            or "alertas" in question
        )

        verified_facts = state.get(
            "verified_facts",
            {},
        )

        pending_exams = verified_facts.get(
            "pending_exams",
            [],
        )

        alerts = verified_facts.get(
            "alerts",
            [],
        )

        issues: list[str] = []

        # 1. Todos os exames realmente pendentes
        # precisam aparecer na resposta.
        if asks_about_pending_exams:
            for exam in pending_exams:
                exam_name = str(
                    exam.get(
                        "exame",
                        "",
                    )
                ).strip()

                if (
                    exam_name
                    and exam_name.lower() not in answer
                ):
                    issues.append(
                        f"exame_pendente_omitido:{exam_name}"
                    )

        # 2. Valida os protocolos dos exames,
        # sem repetir o mesmo protocolo.
        if asks_about_pending_exams:
            exam_protocols = {
                str(
                    exam.get(
                        "protocolo",
                        "",
                    )
                ).strip()
                for exam in pending_exams
                if exam.get("protocolo")
            }

            for protocol in exam_protocols:
                if protocol.lower() not in answer:
                    issues.append(
                        f"protocolo_exame_omitido:{protocol}"
                    )

        # 3. Validação dos alertas existentes.
        if asks_about_alerts:
            for alert in alerts:
                protocol = str(
                    alert.get(
                        "protocolo",
                        "",
                    )
                ).strip()

                priority = str(
                    alert.get(
                        "prioridade",
                        "",
                    )
                ).strip()

                detail = str(
                    alert.get(
                        "detalhe",
                        "",
                    )
                ).lower()

                if (
                    protocol
                    and protocol.lower() not in answer
                ):
                    issues.append(
                        f"protocolo_alerta_omitido:{protocol}"
                    )

                if (
                    priority
                    and priority.lower() not in answer
                ):
                    issues.append(
                        f"prioridade_alerta_omitida:{priority}"
                    )

                # Se o alerta fala em mais de 30 dias,
                # a resposta precisa deixar claro que
                # não sabemos qual exame originou o alerta.
                if "mais de 30 dias" in detail:
                    warning_fragments = [
                        (
                            "não identificam qual exame "
                            "pendente originou"
                        ),
                        (
                            "não identifica qual exame "
                            "pendente originou"
                        ),
                        (
                            "não é possível identificar "
                            "qual exame"
                        ),
                    ]

                    warning_found = any(
                        fragment in answer
                        for fragment in warning_fragments
                    )

                    if not warning_found:
                        issues.append(
                            "alerta_30d_sem_ressalva"
                        )

        # Remove duplicações preservando a ordem.
        issues = list(
            dict.fromkeys(
                issues
            )
        )

        response_consistent = (
            len(issues) == 0
        )

        audit_log(
            {
                "etapa": (
                    "langgraph_grounding_validation"
                ),
                "paciente_id": state.get(
                    "paciente_id"
                ),
                "response_consistent": (
                    response_consistent
                ),
                "consistency_issues": issues,
            }
        )

        return {
            "response_consistent": (
                response_consistent
            ),
            "consistency_issues": issues,
            "status": "GROUNDING_VALIDATED",
        }

    def safety_check(
        self,
        state: MedicalAssistantState,
    ) -> dict[str, Any]:
        """
        Detecta pedidos clínicos sensíveis.

        Perguntas relacionadas a prescrição,
        dosagem, mudança de medicamento ou
        diagnóstico definitivo exigem revisão.

        Também detecta doses concretas geradas
        pelo modelo.
        """

        question = state.get(
            "question",
            "",
        ).lower()

        answer = state.get(
            "answer",
            "",
        ).lower()

        question_rules = {
            "prescricao": (
                r"\b(" r"prescrev\w*|" r"prescriç\w*|" r"prescric\w*" r")\b"
            ),
            "dosagem": (r"\b(" r"dose|" r"doses|" r"dosagem" r")\b"),
            "alteracao_medicamento": (
                r"\b("
                r"suspender|"
                r"interromper|"
                r"trocar medicamento|"
                r"alterar medicamento"
                r")\b"
            ),
            "diagnostico_direto": (
                r"\b("
                r"diagnostique|"
                r"diagnóstico definitivo|"
                r"diagnostico definitivo"
                r")\b"
            ),
        }

        reasons: list[str] = []

        for reason, pattern in question_rules.items():
            if re.search(
                pattern,
                question,
                flags=re.IGNORECASE,
            ):
                reasons.append(reason)

        concrete_dose_pattern = r"\b\d+(?:[.,]\d+)?\s*" r"(?:mg|ml|mcg|µg)\b"

        if re.search(
            concrete_dose_pattern,
            answer,
            flags=re.IGNORECASE,
        ):
            reasons.append("dose_concreta_gerada")

        reasons = list(dict.fromkeys(reasons))

        if not state.get(
            "response_consistent",
            True,
        ):
            reasons.append("grounding_inconsistency")
        requires_human_review = bool(reasons)
        reasons = list(dict.fromkeys(reasons))

        requires_human_review = bool(reasons)

        audit_log(
            {
                "etapa": ("langgraph_safety_check"),
                "paciente_id": state.get("paciente_id"),
                "requires_human_review": (requires_human_review),
                "risk_reasons": reasons,
            }
        )

        return {
            "requires_human_review": (requires_human_review),
            "risk_reasons": reasons,
            "status": "SAFETY_CHECKED",
        }
    def calculate_confidence(
        self,
        state: MedicalAssistantState,
    ) -> dict[str, Any]:
        """
        Calcula um índice determinístico de confiança operacional.

        O score não representa probabilidade clínica.
        Ele mede se a resposta está suficientemente
        fundamentada e segura para entrega automática.

        Responsável: Paola
        """

        score = 0.0
        reasons: list[str] = []

        verified_facts = state.get(
            "verified_facts",
            {},
        )

        sources = state.get(
            "sources",
            [],
        )

        consistency_issues = state.get(
            "consistency_issues",
            [],
        )

        response_consistent = state.get(
            "response_consistent",
            False,
        )

        requires_human_review = state.get(
            "requires_human_review",
            False,
        )

        # Grounding validado.
        if response_consistent:
            score += 0.40
            reasons.append(
                "grounding_consistente"
            )
        else:
            reasons.append(
                "grounding_inconsistente"
            )

        # Ausência de inconsistências.
        if not consistency_issues:
            score += 0.15
            reasons.append(
                "sem_inconsistencias"
            )
        else:
            reasons.append(
                "inconsistencias_detectadas"
            )

        # O sistema conseguiu verificar explicitamente
        # se o paciente existe ou não no banco.
        if isinstance(
            verified_facts.get(
                "patient_found"
            ),
            bool,
        ):
            score += 0.15
            reasons.append(
                "status_paciente_verificado"
            )

        # Houve recuperação de fontes internas.
        if sources:
            score += 0.10
            reasons.append(
                "fontes_recuperadas"
            )
        else:
            reasons.append(
                "sem_fontes_recuperadas"
            )

        # Resposta liberada pelo safety check.
        if not requires_human_review:
            score += 0.20
            reasons.append(
                "sem_risco_para_entrega_automatica"
            )
        else:
            reasons.append(
                "revisao_humana_obrigatoria"
            )

            # Uma resposta que exige revisão humana
            # nunca pode receber confiança alta.
            score = min(
                score,
                0.49,
            )

        score = round(
            min(
                max(
                    score,
                    0.0,
                ),
                1.0,
            ),
            2,
        )

        if score >= 0.80:
            level = "ALTA"
        elif score >= 0.50:
            level = "MEDIA"
        else:
            level = "BAIXA"

        audit_log(
            {
                "etapa": (
                    "langgraph_confidence"
                ),
                "paciente_id": state.get(
                    "paciente_id"
                ),
                "confidence_score": score,
                "confidence_level": level,
                "confidence_reasons": reasons,
            }
        )

        return {
            "confidence_score": score,
            "confidence_level": level,
            "confidence_reasons": reasons,
            "status": "CONFIDENCE_CALCULATED",
        }

    def human_review(
        self,
        state: MedicalAssistantState,
    ) -> dict[str, Any]:
        if not self.interactive_review:
            audit_log(
                {
                    "etapa": ("langgraph_human_review_required"),
                    "paciente_id": state.get("paciente_id"),
                    "motivos": state.get(
                        "risk_reasons",
                        [],
                    ),
                    "status": ("PENDING_HUMAN_REVIEW"),
                }
            )

            return {
                "reviewer_name": None,
                "review_decision": None,
                "review_notes": None,
                "delivery_allowed": False,
                "status": ("PENDING_HUMAN_REVIEW"),
            }

        print()
        print("=" * 60)
        print("REVISÃO HUMANA OBRIGATÓRIA")
        print("=" * 60)

        print(f"Paciente: " f"{state.get('paciente_id') or 'não informado'}")

        print("Motivos: " f"{', '.join(state.get('risk_reasons', []))}")

        print()
        print("Pergunta:")
        print(
            state.get(
                "question",
                "",
            )
        )

        print()
        print("Resposta gerada:")
        print(
            state.get(
                "answer",
                "",
            )
        )

        print()
        print("Fontes:")
        print(
            ", ".join(
                state.get(
                    "sources",
                    [],
                )
            )
            or "Nenhuma"
        )

        print("=" * 60)

        reviewer_name = input("Nome do profissional responsável: ").strip()

        while not reviewer_name:
            print("O nome do profissional " "é obrigatório.")

            reviewer_name = input("Nome do profissional responsável: ").strip()

        decision_input = input("Decisão [A]provar / [R]ejeitar: ").strip().lower()

        while decision_input not in {
            "a",
            "aprovar",
            "aprovado",
            "r",
            "rejeitar",
            "rejeitado",
        }:
            print("Informe A para aprovar " "ou R para rejeitar.")

            decision_input = input("Decisão [A]provar / [R]ejeitar: ").strip().lower()

        review_notes = input("Observações da revisão: ").strip()

        approved = decision_input in {
            "a",
            "aprovar",
            "aprovado",
        }

        review_decision = "APPROVED" if approved else "REJECTED"

        status = "HUMAN_APPROVED" if approved else "HUMAN_REJECTED"

        audit_log(
            {
                "etapa": ("langgraph_human_review_completed"),
                "paciente_id": state.get("paciente_id"),
                "profissional": (reviewer_name),
                "decisao": (review_decision),
                "observacoes": (review_notes),
                "motivos_risco": state.get(
                    "risk_reasons",
                    [],
                ),
                "fontes": state.get(
                    "sources",
                    [],
                ),
                "resposta_validada": (
                    state.get(
                        "answer",
                        "",
                    )
                ),
            }
        )

        return {
            "reviewer_name": (reviewer_name),
            "review_decision": (review_decision),
            "review_notes": (review_notes),
            "delivery_allowed": approved,
            "status": status,
        }

    def finalize(
        self,
        state: MedicalAssistantState,
    ) -> dict[str, Any]:
        audit_log(
            {
                "etapa": ("langgraph_completed"),
                "paciente_id": state.get("paciente_id"),
                "modelo": self.model_name,
                "fontes": state.get(
                    "sources",
                    [],
                ),
                "delivery_allowed": True,
            }
        )

        return {
            "delivery_allowed": True,
            "status": "COMPLETED",
        }

    @staticmethod
    def route_after_safety(
        state: MedicalAssistantState,
    ) -> str:
        if state.get(
            "requires_human_review",
            False,
        ):
            return "human_review"

        return "finalize"
