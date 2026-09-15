"""
Exporta o modelo fine-tunado para GGUF (Ollama) SEM retreinar.
Responsável: Vinicius (Frente 1)

Carrega o adapter LoRA já treinado (models/lora-llama32-3b-medico), mescla com o
modelo base e exporta em GGUF quantizado q4_k_m — o formato que o Ollama consome.
Na primeira execução o Unsloth baixa/compila o llama.cpp para fazer a conversão
(exige build-essential e cmake instalados no WSL).

Uso (WSL2, GPU):
  python -m src.finetune.export_gguf
  python -m src.finetune.export_gguf --quant q8_0   # mais fiel, arquivo maior

Depois, no Ollama (Windows ou WSL):
  cd models/gguf
  ollama create medico-fase3 -f Modelfile
  ollama run medico-fase3 "Qual o protocolo para BI-RADS 4?"

E o grafo da Frente 3 passa a usar o modelo fine-tunado:
  python -m src.graph.workflow -q "..." -p P0001 --model medico-fase3
"""

from __future__ import annotations

import argparse
import glob
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODELFILE = """# Modelo fine-tunado do Tech Challenge Fase 3 (Llama 3.2 3B + LoRA médico)
# O system prompt NÃO é fixado aqui: quem o injeta é o pipeline
# (src/assistant/ollama_llm.py usa prompts.SYSTEM_ASSISTENTE).
FROM ./{gguf_name}
PARAMETER temperature 0.2
PARAMETER num_ctx 4096
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta adapter treinado para GGUF")
    parser.add_argument("--quant", default="q4_k_m",
                        help="método de quantização GGUF (padrão: q4_k_m)")
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "models", "gguf"))
    args = parser.parse_args()

    with open(os.path.join(ROOT, "configs", "finetune.yaml"), encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    adapter_dir = os.path.join(ROOT, cfg["output"]["adapter_dir"])
    if not os.path.isdir(adapter_dir):
        raise SystemExit(f"Adapter não encontrado em {adapter_dir}. Rode o treino antes.")

    from unsloth import FastLanguageModel

    print(f"Carregando base + adapter de {adapter_dir} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_dir,
        max_seq_length=cfg["model"]["max_seq_length"],
        load_in_4bit=True,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Exportando GGUF ({args.quant}) para {args.out_dir} ...")
    model.save_pretrained_gguf(args.out_dir, tokenizer, quantization_method=args.quant)

    ggufs = sorted(glob.glob(os.path.join(args.out_dir, "*.gguf")),
                   key=os.path.getmtime, reverse=True)
    if not ggufs:
        raise SystemExit("Nenhum .gguf gerado — veja o log acima.")
    gguf_name = os.path.basename(ggufs[0])

    modelfile_path = os.path.join(args.out_dir, "Modelfile")
    with open(modelfile_path, "w", encoding="utf-8") as fh:
        fh.write(MODELFILE.format(gguf_name=gguf_name))

    print("\nOK! Próximos passos:")
    print(f"  1. cd {os.path.relpath(args.out_dir, ROOT)}")
    print("  2. ollama create medico-fase3 -f Modelfile")
    print("  3. ollama run medico-fase3 \"Qual o protocolo de triagem para nódulo mamário?\"")
    print("  4. python -m src.graph.workflow -q \"...\" -p P0001 --model medico-fase3")


if __name__ == "__main__":
    main()
