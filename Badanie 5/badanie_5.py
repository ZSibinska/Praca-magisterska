import os
import re
from pathlib import Path
from functools import lru_cache
from datetime import datetime
from typing import List, Dict, Optional

from langchain_community.llms import CTransformers
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# KONFIGURACJA
# ============================================================

MODEL_PATH = r"ścieżka/do/modelu.gguf"
MODEL_TYPE = "mistral"

# Folder, w którym znajduje się ten plik .py
SCRIPT_DIR = Path(__file__).resolve().parent

# ZMIENIASZ RĘCZNIE DLA KOLEJNYCH URUCHOMIEŃ
# Ważne: ścieżki są liczone względem folderu, w którym leży skrypt.
RAG_FILE_PATH = SCRIPT_DIR / "data" / "rag_badanie_5_color_fruit.txt"
INPUT_TEXT_PATH = SCRIPT_DIR / "input_badanie_5" / "color_fruit.txt"

# Dla kolejnych uruchomień podmień dwie linie powyżej np. na:
# RAG_FILE_PATH = SCRIPT_DIR / "data" / "rag_experiment_5_xq.txt"
# INPUT_TEXT_PATH = SCRIPT_DIR / "input_badanie_5" / "xq.txt"
# albo:
# RAG_FILE_PATH = SCRIPT_DIR / "data" / "rag_experiment_5_color_fruit.txt"
# INPUT_TEXT_PATH = SCRIPT_DIR / "input_badanie_5" / "color_fruit.txt"

# Raport zapisze się do folderu "Wyniki" obok tego pliku .py
REPORTS_DIR = SCRIPT_DIR / "Wyniki"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CONTEXT_LENGTH = 4096
GPU_LAYERS = 0

MAX_NEW_TOKENS = 1400
TEMPERATURE = 0.0
TOP_P = 0.9
REPETITION_PENALTY = 1.12


# ============================================================
# NARZĘDZIA
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
        r"^Chat:\s*",
        r"^Odpowiedź:\s*",
        r"^Analiza:\s*",
        r"^Raport:\s*",
        r"^Wynik:\s*",
    ]

    for pattern in prefixes:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

    return cleaned.strip()


def truncate_text(text: str, max_chars: int) -> str:
    text = normalize_spaces(text)

    if len(text) <= max_chars:
        return text

    shortened = text[:max_chars]

    if " " in shortened:
        shortened = shortened.rsplit(" ", 1)[0]

    return shortened + " ..."


def load_text_file(file_path: Path) -> str:
    if not file_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {file_path}")

    text = file_path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(f"Plik jest pusty: {file_path}")

    return normalize_spaces(text)


# ============================================================
# WCZYTYWANIE RAG
# ============================================================

def extract_section_metadata(text: str, base_metadata: Optional[Dict] = None) -> Dict:
    metadata = dict(base_metadata or {})
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line in lines:
        if line.startswith("## "):
            metadata["section_title"] = line.replace("## ", "").strip()
        elif line.startswith("### "):
            metadata["subsection_title"] = line.replace("### ", "").strip()
        elif line.startswith("Typ:"):
            metadata["trait_type"] = line.replace("Typ:", "").strip()
        elif line.startswith("Cecha:"):
            metadata["feature_name"] = line.replace("Cecha:", "").strip()
        elif line.startswith("Źródło:"):
            metadata["source_citation"] = line.replace("Źródło:", "").strip()

    metadata["source_file"] = str(RAG_FILE_PATH)

    return metadata


def load_structured_rag_documents(file_path: Path) -> List[Document]:
    text = file_path.read_text(encoding="utf-8")

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ],
        strip_headers=False,
    )

    header_docs = header_splitter.split_text(text)

    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=180,
        separators=[
            "\n---\n",
            "\n\n",
            "\n- ",
            "\n",
            ". ",
            " ",
            "",
        ],
        add_start_index=True,
    )

    final_docs: List[Document] = []

    for doc in header_docs:
        content = doc.page_content.strip()

        if not content:
            continue

        base_metadata = dict(doc.metadata or {})
        enriched_metadata = extract_section_metadata(content, base_metadata)

        if len(content) <= 1400:
            final_docs.append(
                Document(
                    page_content=content,
                    metadata=enriched_metadata,
                )
            )
        else:
            split_docs = fallback_splitter.create_documents(
                [content],
                metadatas=[enriched_metadata],
            )
            final_docs.extend(split_docs)

    if not final_docs:
        final_docs = fallback_splitter.create_documents([text])

    return final_docs


# ============================================================
# PROMPT
# ============================================================

ANALYSIS_PROMPT = PromptTemplate.from_template(
    """
Jesteś systemem badawczym analizującym wypowiedź użytkownika na podstawie kontekstu RAG.

Twoje zadanie:
1. Przejrzyj wszystkie cechy opisane w RAG jedna po drugiej.
2. Dla każdej cechy sprawdź, czy jej warunki są spełnione dosłownie w wypowiedzi użytkownika.
3. Sprawdzaj zarówno cechy znaczeniowe, jak i formalne cechy zapisu.
4. Formalne cechy zapisu mogą dotyczyć między innymi: kolejności znaków, końcówek słów, powtarzalnych wzorców tekstowych, sekwencji wyrazów albo układu interpunkcji.
5. W raporcie wypisz tylko te cechy, których warunki są jednoznacznie spełnione.
6. Przy każdej cesze podaj dokładny fragment wypowiedzi, który ją uzasadnia.
7. Nie zgaduj cech i nie dopasowuj fragmentu do najbliższej znanej kategorii.
8. Nie uznawaj cechy za obecną, jeśli fragment wypowiedzi nie spełnia dokładnej definicji z RAG.
9. Jeżeli cecha wymaga powtórzenia słowa, to to samo słowo lub bardzo podobna fraza musi wystąpić co najmniej dwa razy.
10. Jeżeli słowo występuje tylko raz, nie wolno oznaczyć cechy jako powtórzenia.
11. Oceń, czy na podstawie wykrytych cech z RAG widzisz wskazanie dla ADHD.
12. Nie stawiaj diagnozy. Nie pisz, że użytkownik ma ADHD.
13. Traktuj RAG jako bazę wiedzy systemu.
14. Nie dodawaj cech, których nie ma w RAG.

Przygotuj odpowiedź jako krótki raport tekstowy w poniższej strukturze:

1. Cechy wykryte w wypowiedzi
- nazwa cechy:
- fragment wypowiedzi:
- uzasadnienie na podstawie RAG:

2. Czy widoczne jest wskazanie dla ADHD według reguł z RAG?
Odpowiedz: TAK albo NIE.
Krótko uzasadnij odpowiedź.

3. Krótkie podsumowanie
Napisz 2-3 zdania podsumowania.

4. Zastrzeżenie metodologiczne
Napisz, że wynik nie jest diagnozą i zależy od zawartości RAG.

Wypowiedź użytkownika:
{input_text}

Kontekst RAG:
{rag_context}

Raport:
""".strip()
)


# ============================================================
# KOMPONENTY LANGCHAIN
# ============================================================

@lru_cache(maxsize=1)
def build_llm() -> CTransformers:
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Nie znaleziono modelu: {MODEL_PATH}")

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


@lru_cache(maxsize=1)
def build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def build_vector_store() -> InMemoryVectorStore:
    if not RAG_FILE_PATH.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku RAG: {RAG_FILE_PATH}")

    rag_docs = load_structured_rag_documents(RAG_FILE_PATH)

    embeddings = build_embeddings()
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(rag_docs)

    return vector_store


# ============================================================
# SILNIK EKSPERYMENTU
# ============================================================

class SimpleRagExperimentEngine:
    def __init__(self):
        self.llm = build_llm()

        self.rag_docs = load_structured_rag_documents(RAG_FILE_PATH)
        self.vector_store = build_vector_store()

        self.output_parser = StrOutputParser()
        self.analysis_chain = ANALYSIS_PROMPT | self.llm | self.output_parser

    def retrieve_context(self, input_text: str) -> List[Document]:
        """
        W eksperymencie 5 podajemy cały RAG.
        Dzięki temu model widzi wszystkie cechy opisane w bazie wiedzy.
        """
        return self.rag_docs

    def format_retrieved_context(self, docs: List[Document]) -> str:
        if not docs:
            return "Brak kontekstu RAG."

        lines = []

        for idx, doc in enumerate(docs, start=1):
            section_title = doc.metadata.get("section_title", "Brak tytułu")
            trait_type = doc.metadata.get("trait_type", "Brak typu")

            lines.append(
                f"{idx}. [{section_title}] ({trait_type})\n"
                f"{doc.page_content}"
            )

        return "\n\n".join(lines)

    def analyze(self, input_text: str, retrieved_context: List[Document]) -> str:
        rag_context = self.format_retrieved_context(retrieved_context)

        raw = self.analysis_chain.invoke(
            {
                "input_text": truncate_text(input_text, 1800),
                "rag_context": truncate_text(rag_context, 6000),
            }
        )

        cleaned = clean_model_output(raw)

        if cleaned:
            return cleaned

        return (
            "Nie udało się wygenerować raportu modelu.\n\n"
            "Wynik nieważny technicznie."
        )

    def create_report_payload(
        self,
        input_text: str,
        retrieved_context: List[Document],
        model_report: str,
    ) -> Dict:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        serialized_context = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in retrieved_context
        ]

        return {
            "timestamp": timestamp,
            "rag_file_path": str(RAG_FILE_PATH),
            "input_text_path": str(INPUT_TEXT_PATH),
            "input_text": input_text,
            "retrieved_context": serialized_context,
            "model_report": model_report,
        }

    def save_report_file(self, report_data: Dict) -> str:
        timestamp = report_data["timestamp"]

        txt_path = REPORTS_DIR / f"badanie5_{timestamp}.txt"

        with open(txt_path, "w", encoding="utf-8") as f:
            self.write_txt_report(f, report_data)

        return str(txt_path)

    def write_txt_report(self, f, report_data: Dict) -> None:
        f.write("=== RAPORT ANALIZY WYPOWIEDZI Z WYKORZYSTANIEM RAG ===\n\n")

        f.write("1. KONFIGURACJA\n")
        f.write("-" * 80 + "\n")
        f.write(f"Data uruchomienia: {report_data['timestamp']}\n")
        f.write(f"Plik RAG: {report_data['rag_file_path']}\n")
        f.write(f"Plik wejściowy: {report_data['input_text_path']}\n\n")

        f.write("2. WYPOWIEDŹ UŻYTKOWNIKA\n")
        f.write("-" * 80 + "\n")
        f.write(report_data["input_text"] + "\n\n")

        f.write("3. KONTEKST RAG PRZEKAZANY DO MODELU\n")
        f.write("-" * 80 + "\n")

        retrieved_context = report_data.get("retrieved_context", [])

        if retrieved_context:
            for idx, item in enumerate(retrieved_context, start=1):
                metadata = item.get("metadata", {})
                section_title = metadata.get("section_title", "Brak tytułu")
                trait_type = metadata.get("trait_type", "Brak typu")

                f.write(f"{idx}. [{section_title}] ({trait_type})\n")
                f.write(item.get("page_content", "") + "\n\n")
        else:
            f.write("Brak kontekstu RAG.\n\n")

        f.write("4. RAPORT MODELU\n")
        f.write("-" * 80 + "\n")
        f.write(report_data.get("model_report", "") + "\n")


# ============================================================
# WALIDACJA
# ============================================================

def check_requirements() -> Optional[str]:
    if not Path(MODEL_PATH).exists():
        return f"Nie znaleziono modelu GGUF pod ścieżką: {MODEL_PATH}"

    if not RAG_FILE_PATH.exists():
        return f"Nie znaleziono pliku RAG: {RAG_FILE_PATH}"

    if not INPUT_TEXT_PATH.exists():
        return f"Nie znaleziono pliku z wypowiedzią: {INPUT_TEXT_PATH}"

    return None


# ============================================================
# START
# ============================================================

def main() -> None:
    error = check_requirements()

    if error:
        print(f"BŁĄD: {error}")
        return

    print("=== EKSPERYMENT 5 — PROSTY TEST RAG ===")
    print(f"Folder skryptu: {SCRIPT_DIR}")
    print(f"Plik RAG: {RAG_FILE_PATH}")
    print(f"Plik wejściowy: {INPUT_TEXT_PATH}")
    print()

    input_text = load_text_file(INPUT_TEXT_PATH)

    engine = SimpleRagExperimentEngine()

    retrieved_context = engine.retrieve_context(input_text)

    model_report = engine.analyze(
        input_text=input_text,
        retrieved_context=retrieved_context,
    )

    report_data = engine.create_report_payload(
        input_text=input_text,
        retrieved_context=retrieved_context,
        model_report=model_report,
    )

    txt_path = engine.save_report_file(report_data)

    print("\nZapisano raport:")
    print(f"- TXT: {txt_path}")


if __name__ == "__main__":
    main()
