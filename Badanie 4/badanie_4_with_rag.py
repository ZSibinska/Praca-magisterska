from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from langchain_community.llms import CTransformers
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


# ============================================================
# KONFIGURACJA EKSPERYMENTU — WARIANT Z RAG
# ============================================================

MODEL_PATH = r"ścieżka/do/modelu.gguf"
MODEL_TYPE = "mistral"

OUT_DIR = "badanie4_raport_RAG"

USER_FILE = "wypowiedz.txt"
RAG_FILE_PATH = r"data\adhd_cechy_jezykowe.txt"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.2
TOP_P = 0.9
REPETITION_PENALTY = 1.12
CONTEXT_LENGTH = 4096
GPU_LAYERS = 0

TOP_K_RAG = 5
FETCH_K_RAG = 12
MMR_LAMBDA = 0.75
MAX_RAG_CONTEXT_CHARS = 3200

EXPERIMENT_VARIANT = "RAG"

# Lista cech jest taka sama jak w wariancie bez RAG.
# Różnica polega na tym, że tutaj model dostaje dodatkowo kontekst z bazy wiedzy.
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


def resolve_rag_file_path() -> Path:
    """Zwraca ścieżkę do pliku RAG względem folderu skryptu."""
    script_dir = Path(__file__).resolve().parent
    configured_path = Path(RAG_FILE_PATH)

    if configured_path.is_absolute() and configured_path.exists():
        return configured_path

    candidates = [
        script_dir / RAG_FILE_PATH,
        script_dir / RAG_FILE_PATH.replace("\\", "/"),
        script_dir / "data" / "adhd_cechy_jezykowe.txt",
        script_dir / "adhd_cechy_jezykowe.txt",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Nie znaleziono pliku RAG. Sprawdzono między innymi: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def extract_section_metadata(text: str, base_metadata: Dict | None = None) -> Dict:
    metadata = dict(base_metadata or {})
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line in lines:
        if line.startswith("## "):
            metadata["section_title"] = line.replace("## ", "").strip()
        elif line.startswith("Źródło:"):
            metadata["source_citation"] = line.replace("Źródło:", "").strip()
        elif line.startswith("Typ:"):
            metadata["trait_type"] = line.replace("Typ:", "").strip()

    metadata["source_file"] = str(resolve_rag_file_path())
    return metadata


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
# RAG
# ============================================================

def load_structured_rag_documents(file_path: Path) -> List[Document]:
    if not file_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku RAG: {file_path}")

    text = file_path.read_text(encoding="utf-8")

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2")],
        strip_headers=False,
    )
    header_docs = header_splitter.split_text(text)

    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=180,
        separators=["\n---\n", "\n\n", "\n- ", "\n", ". ", " ", ""],
        add_start_index=True,
    )

    final_docs: List[Document] = []

    for doc in header_docs:
        content = doc.page_content.strip()
        if not content:
            continue

        if content.startswith("# ") and "## " not in content and len(content.splitlines()) <= 2:
            continue

        metadata = extract_section_metadata(content, doc.metadata)

        if len(content) <= 1400:
            final_docs.append(Document(page_content=content, metadata=metadata))
        else:
            final_docs.extend(
                fallback_splitter.create_documents([content], metadatas=[metadata])
            )

    return final_docs


def build_vector_store() -> InMemoryVectorStore:
    rag_file = resolve_rag_file_path()
    docs = load_structured_rag_documents(rag_file)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(docs)
    return vector_store


def retrieve_rag_context(user_prompt: str) -> List[Document]:
    vector_store = build_vector_store()
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": TOP_K_RAG,
            "fetch_k": FETCH_K_RAG,
            "lambda_mult": MMR_LAMBDA,
        },
    )
    query = truncate_text(user_prompt, 2500)
    return retriever.invoke(query)


def format_rag_context(docs: List[Document]) -> str:
    if not docs:
        return "Brak pobranego kontekstu z bazy wiedzy."

    blocks = []
    for index, doc in enumerate(docs, start=1):
        title = doc.metadata.get("section_title", "Brak tytułu")
        trait_type = doc.metadata.get("trait_type", "Brak typu")
        source = doc.metadata.get("source_citation", "Brak źródła")
        blocks.append(
            f"[{index}] {title}\n"
            f"Typ: {trait_type}\n"
            f"Źródło: {source}\n"
            f"Treść:\n{doc.page_content}"
        )

    return truncate_text("\n\n".join(blocks), MAX_RAG_CONTEXT_CHARS)


def serialize_docs(docs: List[Document]) -> List[Dict]:
    return [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in docs
    ]


# ============================================================
# PROMPT — WARIANT Z RAG
# ============================================================

REPORT_PROMPT_WITH_RAG = PromptTemplate.from_template(
    """
Jesteś systemem badawczym analizującym cechy językowe wypowiedzi użytkownika.

Model ma korzystać z:
1. wypowiedzi użytkownika,
2. krótkiej listy analizowanych cech,
3. kontekstu pobranego z bazy wiedzy RAG.

BARDZO WAŻNE:
- nie stawiaj diagnozy klinicznej,
- nie pisz, że użytkownik ma ADHD,
- nie podawaj procentów,
- stosuj ostrożny, opisowy język,
- analizuj wyłącznie cechy językowe i narracyjne,
- nie dopisuj informacji, których nie ma w wypowiedzi,
- przy interpretacji cech korzystaj z kontekstu RAG,
- nie powołuj się na cechy, których nie ma na liście analizowanych cech,
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

Kontekst z RAG:
{rag_context}

Wypowiedź użytkownika:
{user_prompt}

Raport:
""".strip()
)


def generate_report(user_prompt: str, retrieved_docs: List[Document]) -> str:
    llm = build_llm()
    chain = REPORT_PROMPT_WITH_RAG | llm | StrOutputParser()
    raw = chain.invoke(
        {
            "features": "\n".join(f"- {feature}" for feature in ANALYZED_FEATURES),
            "rag_context": format_rag_context(retrieved_docs),
            "user_prompt": truncate_text(user_prompt, 3500),
        }
    )
    return clean_model_output(raw)


def save_outputs(user_prompt: str, retrieved_docs: List[Document], report: str) -> Dict[str, str]:
    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"badanie4_with_rag_{timestamp}"

    payload = {
        "timestamp": timestamp,
        "experiment_variant": EXPERIMENT_VARIANT,
        "rag_enabled": True,
        "model_path": MODEL_PATH,
        "model_type": MODEL_TYPE,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "repetition_penalty": REPETITION_PENALTY,
        "context_length": CONTEXT_LENGTH,
        "rag_file_path": str(resolve_rag_file_path()),
        "embedding_model_name": EMBEDDING_MODEL_NAME,
        "top_k_rag": TOP_K_RAG,
        "fetch_k_rag": FETCH_K_RAG,
        "mmr_lambda": MMR_LAMBDA,
        "analyzed_features": ANALYZED_FEATURES,
        "user_prompt": user_prompt,
        "user_prompt_word_count": word_count(user_prompt),
        "retrieved_context": serialize_docs(retrieved_docs),
        "report": report,
    }

    json_path = out_dir / f"{base_name}.json"
    txt_path = out_dir / f"{base_name}.txt"

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    retrieved_context_text = format_rag_context(retrieved_docs)
    txt_path.write_text(
        "=== EKSPERYMENT 3 — WARIANT Z RAG ===\n\n"
        f"Model: {MODEL_PATH}\n"
        f"Temperatura: {TEMPERATURE}\n"
        f"Plik RAG: {resolve_rag_file_path()}\n"
        f"Model osadzeń: {EMBEDDING_MODEL_NAME}\n"
        f"Liczba słów w wypowiedzi: {word_count(user_prompt)}\n\n"
        "=== WYPOWIEDŹ UŻYTKOWNIKA ===\n"
        f"{user_prompt}\n\n"
        "=== POBRANY KONTEKST RAG ===\n"
        f"{retrieved_context_text}\n\n"
        "=== RAPORT MODELU ===\n"
        f"{report}\n",
        encoding="utf-8",
    )

    return {"json_path": str(json_path), "txt_path": str(txt_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eksperyment 4 — wariant z RAG")
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

    print("Uruchamiam eksperyment 4 — wariant Z RAG...")
    retrieved_docs = retrieve_rag_context(user_prompt)
    report = generate_report(user_prompt, retrieved_docs)
    paths = save_outputs(user_prompt, retrieved_docs, report)

    print("Zapisano wyniki:")
    print(f"- TXT:  {paths['txt_path']}")


if __name__ == "__main__":
    main()
