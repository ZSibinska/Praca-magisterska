from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from langchain_community.llms import CTransformers
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


# ============================================================
# KONFIGURACJA EKSPERYMENTU — WARIANT BEZ RAG
# ============================================================

MODEL_PATH = r"ścieżka/do/modelu.gguf"
MODEL_TYPE = "mistral"

OUT_DIR = "badanie4_raport_bezRAG"

USER_FILE = "wypowiedz.txt"

MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.2
TOP_P = 0.9
REPETITION_PENALTY = 1.12
CONTEXT_LENGTH = 4096
GPU_LAYERS = 0

EXPERIMENT_VARIANT = "BEZ_RAG"

# Wariant bez RAG dostaje tylko nazwy cech, bez rozbudowanych definicji,
# źródeł, przykładów ani wskaźników z bazy wiedzy.
ANALYZED_FEATURES = [
    "zmniejszona długość wypowiedzi",
    "zmniejszona różnorodność leksykalna",
    "zmniejszona gęstość leksykalna",
    "obniżona spójność wypowiedzi",
    "zwiększona liczba zdań",
    "uproszczona struktura składniowa",
    "zmniejszone użycie wybranych kategorii słów",
    "zwiększone użycie przymiotników",
    "gadatliwość",
    "obniżona jakość pisania",
    "problemy z organizacją narracji",
    "nadmiar informacji i dygresyjność",
    "niepłynność wypowiedzi",
    "mniejsza złożoność i informatywność wypowiedzi",
]

# ============================================================
# FUNKCJE POMOCNICZE
# ============================================================

def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_model_output(text: str) -> str:
    if not text:
        return ""

    cleaned = text.strip()
    prefixes = [
        r"^###\s*Instruction:\s*",
        r"^###\s*Response:\s*",
        r"^###\s*Answer:\s*",
        r"^Assistant:\s*",
        r"^Odpowiedź:\s*",
        r"^Analiza:\s*",
        r"^Raport:\s*",
    ]
    for pattern in prefixes:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def truncate_text(text: str, max_chars: int) -> str:
    text = normalize_spaces(text)
    if len(text) <= max_chars:
        return text
    shortened = text[:max_chars].rsplit(" ", 1)[0]
    return shortened + " ..."


def word_count(text: str) -> int:
    return len(normalize_spaces(text).split())


def build_llm() -> CTransformers:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Nie znaleziono modelu GGUF: {MODEL_PATH}")

    return CTransformers(
        model=MODEL_PATH,
        model_type=MODEL_TYPE,
        config={
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "repetition_penalty": REPETITION_PENALTY,
            "context_length": CONTEXT_LENGTH,
            "gpu_layers": GPU_LAYERS,
        },
    )


# ============================================================
# PROMPT — WARIANT BEZ RAG
# ============================================================

REPORT_PROMPT_NO_RAG = PromptTemplate.from_template(
    """
Jesteś systemem badawczym analizującym cechy językowe wypowiedzi użytkownika.

Model ma korzystać wyłącznie z:
1. wypowiedzi użytkownika,
2. krótkiej listy analizowanych cech,
3. własnej wiedzy językowej.

BARDZO WAŻNE:
- nie stawiaj diagnozy klinicznej,
- nie pisz, że użytkownik ma ADHD,
- nie podawaj procentów,
- stosuj ostrożny, opisowy język,
- analizuj wyłącznie cechy językowe i narracyjne,
- nie dopisuj informacji, których nie ma w wypowiedzi,
- każdą cechę oznacz jako: TAK, NIE albo NIEJEDNOZNACZNE,
- jeśli wskazujesz cechę, podaj krótki cytat lub parafrazę fragmentu wypowiedzi,
- zachowaj dokładnie tę strukturę raportu.

Analizowane cechy:
{features}

Struktura raportu:
1. Krótkie podsumowanie danych
2. Tabela identyfikacji cech językowych
3. Fragmenty wypowiedzi warte uwagi
4. Dominujące sygnały widoczne w wypowiedzi
5. Ostrożny wniosek badawczy
6. Czy warto rozważyć konsultację ze specjalistą i dlaczego
7. Zastrzeżenie metodologiczne

Wypowiedź użytkownika:
{user_prompt}

Raport:
""".strip()
)


def generate_report(user_prompt: str) -> str:
    llm = build_llm()
    chain = REPORT_PROMPT_NO_RAG | llm | StrOutputParser()
    raw = chain.invoke(
        {
            "features": "\n".join(f"- {feature}" for feature in ANALYZED_FEATURES),
            "user_prompt": truncate_text(user_prompt, 3500),
        }
    )
    return clean_model_output(raw)


def save_outputs(user_prompt: str, report: str) -> Dict[str, str]:
    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"badanie4_no_rag_{timestamp}"

    payload = {
        "timestamp": timestamp,
        "experiment_variant": EXPERIMENT_VARIANT,
        "rag_enabled": False,
        "model_path": MODEL_PATH,
        "model_type": MODEL_TYPE,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "repetition_penalty": REPETITION_PENALTY,
        "context_length": CONTEXT_LENGTH,
        "analyzed_features": ANALYZED_FEATURES,
        "user_prompt": user_prompt,
        "user_prompt_word_count": word_count(user_prompt),
        "report": report,
    }

    json_path = out_dir / f"{base_name}.json"
    txt_path = out_dir / f"{base_name}.txt"

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    txt_path.write_text(
        "=== EKSPERYMENT 3 — WARIANT BEZ RAG ===\n\n"
        f"Model: {MODEL_PATH}\n"
        f"Temperatura: {TEMPERATURE}\n"
        f"Liczba słów w wypowiedzi: {word_count(user_prompt)}\n\n"
        "=== WYPOWIEDŹ UŻYTKOWNIKA ===\n"
        f"{user_prompt}\n\n"
        "=== RAPORT MODELU ===\n"
        f"{report}\n",
        encoding="utf-8",
    )

    return {"json_path": str(json_path), "txt_path": str(txt_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eksperyment 4 — wariant bez RAG")
    parser.add_argument("--text", type=str, default=None, help="Wypowiedź użytkownika przekazana bezpośrednio w konsoli")
    parser.add_argument("--input-file", type=str, default=None, help="Ścieżka do pliku TXT z wypowiedzią użytkownika")
    return parser.parse_args()


def load_user_prompt(args: argparse.Namespace) -> str:
    if args.text:
        return args.text.strip()

    script_dir = Path(__file__).resolve().parent
    input_path = Path(args.input_file) if args.input_file else script_dir / USER_FILE

    if not input_path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku z wypowiedzią użytkownika: {input_path}. "
            "Podaj ścieżkę argumentem --input-file albo utwórz plik wskazany w USER_FILE."
        )

    text = input_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Plik z wypowiedzią użytkownika jest pusty: {input_path}")

    return text


def main() -> None:
    args = parse_args()
    user_prompt = load_user_prompt(args)

    print("Uruchamiam eksperyment 4 — wariant BEZ RAG...")
    report = generate_report(user_prompt)
    paths = save_outputs(user_prompt, report)

    print("Zapisano wyniki:")
    print(f"- TXT:  {paths['txt_path']}")


if __name__ == "__main__":
    main()
