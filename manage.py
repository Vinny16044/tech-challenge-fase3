"""
Comando único do projeto — roda TUDO a partir do PowerShell (ou do WSL).

A regra que ninguém mais precisa decorar: treino/export exigem Linux+GPU (WSL),
o resto roda no Windows. Este script detecta onde está e faz o roteamento:
chamado no Windows, os comandos de GPU são executados automaticamente dentro do
WSL (sincronizando o código antes); chamado no WSL, rodam direto.

Uso (sempre da raiz do projeto):
  python manage.py doctor      # diagnóstico do ambiente (comece por aqui)
  python manage.py setup       # cria venv + instala dependências (win e wsl)
  python manage.py db          # cria a base de prontuários no MySQL
  python manage.py dataset     # gera o dataset de fine-tuning
  python manage.py train       # treina (roteia para o WSL sozinho)
  python manage.py evaluate    # avaliação base vs. fine-tunado (WSL)
  python manage.py export      # exporta GGUF (WSL)
  python manage.py ollama      # copia o GGUF do WSL e cria o modelo no Ollama
  python manage.py ask "Quais exames estão pendentes?" -p P0001
  python manage.py validate    # suíte de validação da Frente 3

Configuração opcional por variável de ambiente (ou .env):
  WSL_DISTRO=Ubuntu-24.04      # nome da distro (wsl -l -v)
  WSL_PROJECT=~/tech-challenge-fase3
  OLLAMA_MODEL=medico-fase3
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

WSL_DISTRO = os.environ.get("WSL_DISTRO", "Ubuntu-24.04")
WSL_PROJECT = os.environ.get("WSL_PROJECT", "~/tech-challenge-fase3")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "medico-fase3")

IS_WINDOWS = platform.system() == "Windows"
IS_WSL = "microsoft" in platform.release().lower()


# ----------------------------------------------------------------------
# infra
# ----------------------------------------------------------------------

def run(cmd: list[str] | str, check: bool = True, **kw) -> int:
    printable = cmd if isinstance(cmd, str) else " ".join(cmd)
    print(f"\n$ {printable}")
    result = subprocess.run(cmd, shell=isinstance(cmd, str), **kw)
    if check and result.returncode != 0:
        sys.exit(result.returncode)
    return result.returncode


def in_wsl(bash_cmd: str, check: bool = True) -> int:
    """Executa um comando bash dentro do WSL, no diretório do projeto."""
    full = f"cd {WSL_PROJECT} && {bash_cmd}"
    return run(["wsl", "-d", WSL_DISTRO, "--", "bash", "-lc", full], check=check)


def sync_to_wsl() -> None:
    """Copia o código do Windows para o clone do WSL (sem .venv/models/.git)."""
    win_path = ROOT.replace("\\", "/")
    drive = win_path[0].lower()
    mnt = f"/mnt/{drive}{win_path[2:]}"
    print(f"Sincronizando código -> WSL ({WSL_PROJECT}) ...")
    in_wsl(
        f'mkdir -p {WSL_PROJECT} && '
        f'rsync -a --delete-excluded '
        f'--exclude .venv --exclude .git --exclude models --exclude logs '
        f'--exclude results --exclude __pycache__ --exclude "*.pyc" '
        f'"{mnt}/" {WSL_PROJECT}/ 2>/dev/null || '
        f'cp -r "{mnt}/src" "{mnt}/configs" "{mnt}/data" '
        f'"{mnt}/requirements-finetune.txt" "{mnt}/manage.py" {WSL_PROJECT}/'
    )


def venv_python() -> str:
    """Python do venv local (cria se não existir)."""
    py = os.path.join(ROOT, ".venv",
                      "Scripts" if IS_WINDOWS else "bin",
                      "python.exe" if IS_WINDOWS else "python")
    return py if os.path.exists(py) else sys.executable


def gpu_task(module: str) -> None:
    """Roda um módulo que exige GPU: direto no WSL/Linux, ou delegando do Windows."""
    if IS_WINDOWS:
        print(f"[manage] Comando de GPU — executando dentro do WSL ({WSL_DISTRO}).")
        sync_to_wsl()
        in_wsl(f"source .venv/bin/activate && python -m {module}")
    else:
        run([venv_python(), "-m", module])


# ----------------------------------------------------------------------
# comandos
# ----------------------------------------------------------------------

def cmd_doctor(_args) -> None:
    ok = lambda msg: print(f"  [ok]   {msg}")        # noqa: E731
    bad = lambda msg: print(f"  [FALTA] {msg}")      # noqa: E731

    print(f"Ambiente: {'Windows' if IS_WINDOWS else 'WSL/Linux' if IS_WSL else 'Linux'}"
          f" | Python {platform.python_version()}")

    # Ollama
    if shutil.which("ollama"):
        ok("ollama instalado")
        r = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if OLLAMA_MODEL in (r.stdout or ""):
            ok(f"modelo '{OLLAMA_MODEL}' criado no Ollama")
        else:
            bad(f"modelo '{OLLAMA_MODEL}' ainda não criado — rode: python manage.py ollama")
    else:
        bad("ollama não encontrado no PATH (instale de ollama.com)")

    # MySQL
    try:
        sys.path.insert(0, ROOT)
        from src.data.build_prontuarios import get_engine
        from sqlalchemy import text
        with get_engine().connect() as conn:
            n = conn.execute(text("SELECT COUNT(*) FROM pacientes")).scalar()
        ok(f"MySQL acessível — base de prontuários com {n} pacientes")
    except Exception as exc:  # noqa: BLE001
        bad(f"MySQL/base de prontuários: {type(exc).__name__} — rode: python manage.py db "
            f"(confira DATABASE_URL no .env)")

    # WSL + GPU (só faz sentido checar a partir do Windows)
    if IS_WINDOWS:
        r = subprocess.run(["wsl", "-d", WSL_DISTRO, "--", "bash", "-lc", "nvidia-smi -L"],
                           capture_output=True, text=True)
        if r.returncode == 0 and "GPU" in (r.stdout or ""):
            ok(f"WSL '{WSL_DISTRO}' com GPU: {r.stdout.strip().splitlines()[0]}")
        else:
            bad(f"WSL '{WSL_DISTRO}' sem GPU visível (confira o nome com: wsl -l -v)")
        r = subprocess.run(["wsl", "-d", WSL_DISTRO, "--", "bash", "-lc",
                            f"test -d {WSL_PROJECT}/.venv && echo sim"],
                           capture_output=True, text=True)
        if "sim" in (r.stdout or ""):
            ok("venv de treino existe no WSL")
        else:
            bad("venv de treino não existe no WSL — rode: python manage.py setup")
    elif IS_WSL:
        import importlib.util
        if importlib.util.find_spec("torch"):
            ok("torch instalado (ative o venv para treinar)")
        else:
            bad("torch não encontrado — rode: python manage.py setup")


def cmd_setup(_args) -> None:
    if IS_WINDOWS:
        # deps do lado Windows (dados + LangChain/grafo)
        if not os.path.exists(os.path.join(ROOT, ".venv")):
            run([sys.executable, "-m", "venv", os.path.join(ROOT, ".venv")])
        py = venv_python()
        run([py, "-m", "pip", "install", "-q", "-r", "requirements-data.txt"])
        if os.path.exists(os.path.join(ROOT, "requirements-langchain.txt")):
            run([py, "-m", "pip", "install", "-q", "-r", "requirements-langchain.txt"])
        print("[manage] Dependências do Windows ok. Preparando o WSL (treino)...")
        sync_to_wsl()
        in_wsl("python3 -m venv .venv 2>/dev/null; source .venv/bin/activate && "
               "pip install -q -r requirements-finetune.txt")
        print("[manage] Setup completo (Windows + WSL).")
    else:
        run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-finetune.txt"])


def cmd_db(_args) -> None:
    run([venv_python(), "-m", "src.data.build_prontuarios"])


def cmd_dataset(_args) -> None:
    # repeat 5: mantém o português dominante frente aos 400 exemplos em inglês
    run([venv_python(), "-m", "src.finetune.prepare_dataset", "--synthetic-repeat", "5"])


def cmd_train(_args) -> None:
    gpu_task("src.finetune.train")


def cmd_evaluate(_args) -> None:
    gpu_task("src.finetune.evaluate")


def cmd_export(_args) -> None:
    gpu_task("src.finetune.export_gguf")


def cmd_ollama(_args) -> None:
    """Copia GGUF+Modelfile do WSL e cria o modelo no Ollama do Windows."""
    if not shutil.which("ollama"):
        sys.exit("ollama não encontrado no PATH — instale de https://ollama.com")

    if IS_WINDOWS:
        import glob as _glob
        base = fr"\\wsl.localhost\{WSL_DISTRO}\home"
        users = [d for d in _glob.glob(os.path.join(base, "*"))]
        gguf, modelfile = None, None
        for user_dir in users:
            proj = os.path.join(user_dir, os.path.basename(WSL_PROJECT.rstrip("/")))
            for sub in ("models\\gguf_gguf", "models\\gguf"):
                found = _glob.glob(os.path.join(proj, sub, "*.gguf"))
                if found:
                    gguf = found[0]
                    mf = os.path.join(proj, sub, "Modelfile")
                    modelfile = mf if os.path.exists(mf) else None
                    break
            if gguf:
                break
        if not gguf:
            sys.exit("Nenhum .gguf encontrado no WSL — rode antes: python manage.py export")

        dest = os.path.join(os.path.expanduser("~"), "medico-gguf")
        os.makedirs(dest, exist_ok=True)
        gguf_name = os.path.basename(gguf)
        local = os.path.join(dest, gguf_name)
        # copia sempre que o export do WSL for mais novo que a cópia local
        if (not os.path.exists(local)
                or os.path.getmtime(gguf) > os.path.getmtime(local)):
            print(f"Copiando {gguf_name} (~2 GB) para {dest} ...")
            shutil.copy2(gguf, dest)
        if modelfile:
            content = open(modelfile, encoding="utf-8").read().splitlines()
            content = [f"FROM ./{gguf_name}" if line.strip().upper().startswith("FROM")
                       else line for line in content]
        else:
            content = [f"FROM ./{gguf_name}", "PARAMETER temperature 0.2",
                       "PARAMETER num_ctx 4096"]
        with open(os.path.join(dest, "Modelfile"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(content) + "\n")
        run(["ollama", "create", OLLAMA_MODEL, "-f", "Modelfile"], cwd=dest)
    else:
        for sub in ("models/gguf_gguf", "models/gguf"):
            mf = os.path.join(ROOT, sub, "Modelfile")
            if os.path.exists(mf):
                run(["ollama", "create", OLLAMA_MODEL, "-f", mf])
                break
        else:
            sys.exit("Modelfile não encontrado — rode antes: python manage.py export")

    print(f"\n[manage] Pronto! Teste com: ollama run {OLLAMA_MODEL} \"sua pergunta\"")


def cmd_ask(args) -> None:
    cmd = [venv_python(), "-m", "src.graph.workflow",
           "-q", args.question, "--model", OLLAMA_MODEL]
    if args.paciente:
        cmd += ["-p", args.paciente]
    run(cmd)


def cmd_validate(_args) -> None:
    # garante que a suíte use o mesmo modelo do comando `ask`
    os.environ["OLLAMA_MODEL"] = OLLAMA_MODEL
    run([venv_python(), "-m", "src.validation.run_validation"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Comando único do Tech Challenge Fase 3")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("doctor", "setup", "db", "dataset", "train",
                 "evaluate", "export", "ollama", "validate"):
        sub.add_parser(name)
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--paciente", "-p", default=None)

    args = parser.parse_args()
    globals()[f"cmd_{args.cmd}"](args)


if __name__ == "__main__":
    main()
