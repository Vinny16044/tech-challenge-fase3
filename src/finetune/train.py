"""
Treino QLoRA do Llama 3.2 3B Instruct com Unsloth — RTX 3070 (8 GB) / WSL2.
Responsável: Vinicius (Frente 1)

Uso (dentro do WSL2, com o venv ativo):
  python -m src.finetune.train                      # treina e salva o adapter LoRA
  python -m src.finetune.train --merge              # + salva o modelo mesclado (fp16)
  python -m src.finetune.train --export-gguf        # + exporta GGUF q4_k_m p/ Ollama

O que é QLoRA aqui: o modelo base fica congelado e quantizado em 4 bits (~2,2 GB
de VRAM para o 3B); treinamos apenas adaptadores LoRA (matrizes de posto baixo
injetadas nas projeções de atenção e MLP). Cabe na 3070 com folga.
"""

from __future__ import annotations

import argparse
import json
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG = os.path.join(ROOT, "configs", "finetune.yaml")


def load_config() -> dict:
    with open(CONFIG, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning QLoRA (Unsloth)")
    parser.add_argument("--merge", action="store_true", help="salvar modelo mesclado fp16")
    parser.add_argument("--export-gguf", action="store_true", help="exportar GGUF q4_k_m (Ollama)")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="limitar passos (0 = épocas completas); útil para smoke test")
    args = parser.parse_args()

    cfg = load_config()

    # Imports pesados só aqui, para que --help funcione em qualquer máquina
    import torch
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    m, l, t, o = cfg["model"], cfg["lora"], cfg["training"], cfg["output"]

    # Auto-ajuste para GPUs pequenas: com menos de 7 GB de VRAM (ex. RTX 4050
    # 6 GB), reduzimos sequência e batch sem precisar editar o config.
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if vram_gb < 7 and m["max_seq_length"] > 768:
        print(f"[auto] GPU com {vram_gb:.1f} GB — ajustando: seq 768, batch 1, accum 16")
        m["max_seq_length"] = 768
        t["per_device_train_batch_size"] = 1
        t["gradient_accumulation_steps"] = 16

    # 1. Modelo base 4-bit ------------------------------------------------
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=m["base_model"],
        max_seq_length=m["max_seq_length"],
        load_in_4bit=m["load_in_4bit"],
    )
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    # 2. Adapters LoRA ----------------------------------------------------
    model = FastLanguageModel.get_peft_model(
        model,
        r=l["r"],
        lora_alpha=l["lora_alpha"],
        lora_dropout=l["lora_dropout"],
        target_modules=l["target_modules"],
        use_gradient_checkpointing="unsloth",  # economiza VRAM
        random_state=t["seed"],
    )

    # 3. Dataset (JSONL com campo "messages") ----------------------------
    train_file = os.path.join(ROOT, t["train_file"])
    if not os.path.exists(train_file):
        raise SystemExit(
            f"Dataset não encontrado: {train_file}\n"
            "Rode antes: python -m src.finetune.prepare_dataset"
        )
    dataset = load_dataset("json", data_files=train_file, split="train")

    def to_text(example):
        return {"text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False)}

    dataset = dataset.map(to_text, remove_columns=dataset.column_names)
    split = dataset.train_test_split(test_size=t["eval_holdout"], seed=t["seed"])

    # 4. Treino -----------------------------------------------------------
    sft_args = SFTConfig(
        output_dir=os.path.join(ROOT, "models", "checkpoints"),
        dataset_text_field="text",
        max_seq_length=m["max_seq_length"],
        num_train_epochs=t["num_train_epochs"],
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=float(t["learning_rate"]),
        warmup_ratio=t["warmup_ratio"],
        lr_scheduler_type=t["lr_scheduler_type"],
        weight_decay=t["weight_decay"],
        logging_steps=t["logging_steps"],
        eval_strategy="epoch",
        save_strategy="epoch",
        seed=t["seed"],
        # Ampere+ (30xx/40xx) treina em bf16; GPUs antigas (ex. T4 do Colab) usam fp16
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        args=sft_args,
    )

    stats = trainer.train()
    print(f"Treino concluído. Loss final: {stats.training_loss:.4f}")

    # 5. Salvar -----------------------------------------------------------
    adapter_dir = os.path.join(ROOT, o["adapter_dir"])
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Adapter LoRA salvo em {adapter_dir}")

    with open(os.path.join(adapter_dir, "train_stats.json"), "w", encoding="utf-8") as fh:
        json.dump({"training_loss": stats.training_loss,
                   "metrics": stats.metrics}, fh, ensure_ascii=False, indent=2)

    if args.merge or args.export_gguf:
        merged_dir = os.path.join(ROOT, o["merged_dir"])
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        print(f"Modelo mesclado salvo em {merged_dir}")

    if args.export_gguf:
        gguf_dir = os.path.join(ROOT, o["merged_dir"] + "-gguf")
        model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method="q4_k_m")
        print(f"GGUF exportado em {gguf_dir} — importe no Ollama com um Modelfile "
              f"(FROM ./model-q4_k_m.gguf) para a Frente 3 consumir.")


if __name__ == "__main__":
    main()
