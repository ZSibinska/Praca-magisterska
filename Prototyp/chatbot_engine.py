import os
import re
from pathlib import Path
from functools import lru_cache
from datetime import datetime
from typing import List, Dict, Optional, Any

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
from langchain_core.runnables import RunnableLambda


# ============================================================
# KONFIGURACJA
# ============================================================

MODEL_PATH = r"ścieżka/do/modelu.gguf"
MODEL_TYPE = "mistral"

REPORTS_DIR = "Raporty"
RAG_FILE_PATH = os.path.join("dane", "adhd_cechy_jezykowe.txt")

MODEL_DISPLAY_NAME = Path(MODEL_PATH).name
RAG_DISPLAY_NAME = Path(RAG_FILE_PATH).name

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

MIN_CONVERSATION_WORDS = 90
MAX_FOLLOWUPS = 2

# Retrieval kontekstu dla całej analizy
TOP_K_RAG = 4
RAG_FETCH_K = 10
RAG_LAMBDA_MULT = 0.75

# Dopasowania krótkich fragmentów wypowiedzi do bazy wiedzy.
TOP_K_MATCHES_PER_FRAGMENT = 3
MIN_FRAGMENT_WORDS = 20
MAX_FRAGMENT_WORDS = 70
MIN_MATCH_SCORE = 0.28

MAX_NEW_TOKENS = 1050
CONTEXT_LENGTH = 4096
GPU_LAYERS = 0
REPETITION_PENALTY = 1.18

FOLLOWUP_FALLBACKS = [
    "Co najbardziej zapamiętałeś albo zapamiętałaś z tego filmu?",
    "Który moment filmu najbardziej zwrócił Twoją uwagę?",
    "Co Ci się w tym filmie najbardziej podobało?",
    "Czy było w nim coś, co szczególnie Cię zaskoczyło?",
]


# ============================================================
# NARZĘDZIA
# ============================================================

def ensure_reports_dir() -> None:
    os.makedirs(REPORTS_DIR, exist_ok=True)


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
        r"^Podsumowanie:\s*",
    ]

    for pattern in prefixes:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

    return cleaned.strip()


def word_count(text: str) -> int:
    return len(normalize_spaces(text).split())


def truncate_text(text: str, max_chars: int) -> str:
    text = normalize_spaces(text)
    if len(text) <= max_chars:
        return text

    shortened = text[:max_chars]
    if " " in shortened:
        shortened = shortened.rsplit(" ", 1)[0]
    return shortened + " ..."


def ensure_complete_question(text: str) -> str:
    text = normalize_spaces(text)

    if not text:
        return ""

    if text.endswith("?"):
        return text

    qmark_pos = text.rfind("?")
    if qmark_pos != -1:
        return text[:qmark_pos + 1].strip()

    return ""


def split_into_fragments(
    text: str,
    min_words: int = MIN_FRAGMENT_WORDS,
    max_words: int = MAX_FRAGMENT_WORDS,
) -> List[str]:
  
    text = normalize_spaces(text)
    if not text:
        return []

    sentences = re.split(r"(?<=[\.\!\?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [text] if word_count(text) >= min_words else []

    fragments: List[str] = []
    buffer: List[str] = []

    for sentence in sentences:
        candidate = " ".join(buffer + [sentence])

        if buffer and word_count(candidate) > max_words:
            joined = " ".join(buffer).strip()
            if word_count(joined) >= min_words:
                fragments.append(joined)
            buffer = [sentence]
        else:
            buffer.append(sentence)

        joined = " ".join(buffer).strip()
        if word_count(joined) >= min_words and word_count(joined) >= max_words:
            fragments.append(joined)
            buffer = []

    if buffer:
        joined = " ".join(buffer).strip()
        if word_count(joined) >= min_words:
            fragments.append(joined)
        elif fragments:
            # Krótką końcówkę dokładamy do poprzedniego fragmentu, żeby nie
            # wymuszać osobnego, słabego dopasowania.
            fragments[-1] = normalize_spaces(f"{fragments[-1]} {joined}")

    return fragments


def classify_match_score(score: float) -> str:
    """
    Klasyfikuje siłę dopasowania dla wyników zwracanych przez InMemoryVectorStore.
    W obserwowanych raportach wartości trafniejszych kandydatów mieściły się
    zwykle w okolicach 0.28-0.30, dlatego próg akceptacji został obniżony
    z 0.35 do 0.28. Progi mają charakter eksperymentalny i powinny być
    weryfikowane na kolejnych raportach testowych.
    """
    if score >= 0.45:
        return "silne"
    if score >= MIN_MATCH_SCORE:
        return "umiarkowane"
    return "słabe"


def extract_section_metadata(text: str, base_metadata: Optional[Dict] = None) -> Dict:
    metadata = dict(base_metadata or {})
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line in lines:
        if line.startswith("## "):
            metadata["section_title"] = line.replace("## ", "").strip()
        elif line.startswith("Źródło:"):
            metadata["source_citation"] = line.replace("Źródło:", "").strip()
        elif line.startswith("Typ:"):
            metadata["trait_type"] = line.replace("Typ:", "").strip()

    metadata["source_file"] = RAG_FILE_PATH
    return metadata


def load_structured_rag_documents(file_path: str) -> List[Document]:
    text = Path(file_path).read_text(encoding="utf-8")

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
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

        if content.startswith("# ") and "## " not in content and len(content.splitlines()) <= 2:
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

    return final_docs


# ============================================================
# PROMPTY LANGCHAIN
# ============================================================

FOLLOWUP_PROMPT = PromptTemplate.from_template(
    """
Jesteś uprzejmym chatbotem prowadzącym naturalną rozmowę po polsku.

Wygeneruj jedno krótkie pytanie pogłębiające o filmie.

Zasady:
- maksymalnie 12 słów,
- tylko jedno zdanie,
- pytanie ma być proste i naturalne,
- bez diagnozy, bez ADHD, bez analizy,
- zwróć wyłącznie samo pytanie.

Wypowiedź użytkownika:
{last_answer}

Pytanie:
""".strip()
)

SYMPTOMS_SUMMARY_PROMPT = PromptTemplate.from_template(
    """
Uporządkuj poniższą wypowiedź użytkownika po polsku.

Zasady:
- nie diagnozuj,
- nie interpretuj klinicznie,
- nie dopisuj nowych informacji,
- zrób krótkie, uporządkowane podsumowanie,
- użyj dwóch sekcji:
  1. Powód zgłoszenia
  2. Zauważane objawy

Wypowiedź:
{symptoms_raw}

Podsumowanie:
""".strip()
)

FULL_ANALYSIS_PROMPT = PromptTemplate.from_template(
    """
Jesteś systemem badawczym analizującym cechy językowe wypowiedzi użytkownika.
Pracujesz wyłącznie na danych poniżej. Nie stawiaj diagnozy klinicznej i nie pisz, że użytkownik ma ADHD.

Najważniejsze zasady interpretacyjne:
- W danych wejściowych znajduje się sekcja "CECHY WSPIERANE ZAAKCEPTOWANYMI DOPASOWANIAMI".
- Sekcja 2.1 raportu ma zawierać wyłącznie cechy wymienione w tej sekcji wejściowej.
- Jeżeli w sekcji wejściowej są zaakceptowane cechy, NIE WOLNO pisać, że brak cech wspartych dopasowaniami.
- Cechy obecne tylko w ogólnym kontekście RAG, ale niewymienione jako zaakceptowane, wpisz wyłącznie w sekcji 2.2 jako niepotwierdzone.
- W sekcji 2.2 podaj tylko nazwy cech oraz krótką informację, że były dostępne jako kontekst RAG, ale nie zostały potwierdzone zaakceptowanymi dopasowaniami fragmentów.
- Dopasowania odrzucone poniżej progu nie są materiałem dowodowym dla sekcji 2.1.
- Pisz zwięźle. Każda sekcja ma mieć maksymalnie 2-4 krótkie zdania albo krótką listę punktowaną.
- W sekcji 3 podaj maksymalnie 2 krótkie fragmenty wypowiedzi, każdy do 140 znaków. Nie przepisuj długich akapitów.
- Odpowiedź musi zawierać dokładnie 4 sekcje i zakończyć się pełnym zdaniem.

Struktura odpowiedzi:
1. Krótkie podsumowanie danych
2. Cechy językowe i narracyjne możliwe do rozważenia
   2.1. Cechy wsparte zaakceptowanymi dopasowaniami fragmentów
   2.2. Cechy obecne w kontekście RAG, ale niepotwierdzone w dopasowaniach fragmentów
3. Fragmenty wypowiedzi warte uwagi, maksymalnie 2 cytaty
4. Ostrożny wniosek badawczy i zastrzeżenie metodologiczne

W sekcji 4 napisz maksymalnie 3 zdania:
- ostrożny wniosek badawczy,
- informację, że wynik nie jest diagnozą,
- ewentualną sugestię konsultacji ze specjalistą, jeśli opisywane trudności utrudniają codzienne funkcjonowanie.

Zakończ raport dokładnie takim zdaniem:
Wynik należy traktować jako pomocniczy materiał badawczy, a nie diagnozę.

Dane użytkownika - objawy:
{symptoms_text}

Skrót objawów:
{symptoms_summary}

Rozmowa swobodna:
{conversation_text}

Skrócony kontekst z RAG:
{rag_context}

Skrócone kandydackie dopasowania fragmentów do RAG:
{fragment_matches}

Pełny raport:
""".strip()
)

USER_SUMMARY_PROMPT = PromptTemplate.from_template(
    """
Na podstawie poniższej analizy przygotuj krótką odpowiedź dla użytkownika.

Zasady:
- dokładnie 3 albo 4 zdania,
- język prosty i zrozumiały,
- wskaż 2-3 najważniejsze obserwacje z wypowiedzi, ale nie przedstawiaj ich jako diagnozy,
- używaj ostrożnych sformułowań, np. "w wypowiedzi pojawiły się", "można zauważyć", "warto omówić",
- napisz ostrożnie, czy warto rozważyć kontakt ze specjalistą,
- nie diagnozuj,
- nie używaj procentów,
- nie pisz zbyt technicznie,
- nie pisz, że pojedyncze zachowania "wskazują na potrzebę kontroli uwagi"; zamiast tego opisz je jako przykłady trudności lub sygnały warte omówienia.

Analiza:
{analysis}

Krótka odpowiedź dla użytkownika:
""".strip()
)


# ============================================================
# KOMPONENTY LANGCHAIN
# ============================================================

@lru_cache(maxsize=1)
def build_llm() -> CTransformers:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Nie znaleziono modelu: {MODEL_PATH}")

    return CTransformers(
        model=MODEL_PATH,
        model_type=MODEL_TYPE,
        config={
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": 0.2,
            "top_p": 0.9,
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
    if not os.path.exists(RAG_FILE_PATH):
        raise FileNotFoundError(
            f"Nie znaleziono pliku bazy wiedzy RAG: {RAG_FILE_PATH}"
        )

    rag_docs = load_structured_rag_documents(RAG_FILE_PATH)

    embeddings = build_embeddings()
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(rag_docs)

    return vector_store


# ============================================================
# SILNIK
# ============================================================

class ADHDResearchEngine:
    def __init__(self):
        self.llm = build_llm()
        self.vector_store = build_vector_store()
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": TOP_K_RAG,
                "fetch_k": RAG_FETCH_K,
                "lambda_mult": RAG_LAMBDA_MULT,
            },
        )
        self.output_parser = StrOutputParser()

        self.followup_chain = FOLLOWUP_PROMPT | self.llm | self.output_parser
        self.symptoms_summary_chain = SYMPTOMS_SUMMARY_PROMPT | self.llm | self.output_parser
        self.user_summary_chain = USER_SUMMARY_PROMPT | self.llm | self.output_parser

        self.full_analysis_chain = (
            RunnableLambda(self._prepare_analysis_inputs)
            | FULL_ANALYSIS_PROMPT
            | self.llm
            | self.output_parser
        )

    # -------------------------
    # FOLLOW-UP
    # -------------------------

    def generate_followup_question(self, last_answer: str, followups_asked: int) -> str:
        raw = self.followup_chain.invoke(
            {"last_answer": truncate_text(last_answer, 300)}
        )
        cleaned = clean_model_output(raw)
        first_line = cleaned.splitlines()[0].strip() if cleaned else ""
        complete_question = ensure_complete_question(first_line)

        if not complete_question:
            index = min(followups_asked, len(FOLLOWUP_FALLBACKS) - 1)
            return FOLLOWUP_FALLBACKS[index]

        return complete_question

    # -------------------------
    # RAG
    # -------------------------

    def retrieve_context_for_analysis(self, symptoms_raw: str, conversation_text: str) -> List[Document]:
        query = normalize_spaces(
            f"{truncate_text(symptoms_raw, 1000)} "
            f"{truncate_text(conversation_text, 1800)}"
        )
        return self.retriever.invoke(query)

    def match_fragments_to_rag(self, conversation_text: str) -> List[Dict[str, Any]]:
        """
        Dopasowuje wyłącznie fragmenty rozmowy swobodnej do bazy wiedzy.

        """
        combined_text = normalize_spaces(truncate_text(conversation_text, 2200))
        fragments = split_into_fragments(combined_text)

        matches: List[Dict[str, Any]] = []

        for fragment in fragments:
            if word_count(fragment) < MIN_FRAGMENT_WORDS:
                continue

            docs_with_scores = self.vector_store.similarity_search_with_score(
                fragment,
                k=TOP_K_MATCHES_PER_FRAGMENT,
            )

            accepted_matches = []
            rejected_matches = []

            for doc, score in docs_with_scores:
                match_record = {
                    "document": doc,
                    "score": float(score),
                    "strength": classify_match_score(float(score)),
                }

                if float(score) >= MIN_MATCH_SCORE:
                    accepted_matches.append(match_record)
                else:
                    rejected_matches.append(match_record)

            matches.append(
                {
                    "fragment": fragment,
                    "matches": accepted_matches,
                    "rejected_matches": rejected_matches,
                    "min_score": MIN_MATCH_SCORE,
                    "note": "" if accepted_matches else "Brak wystarczająco trafnego dopasowania do bazy wiedzy.",
                }
            )

        return matches

    def format_retrieved_context(self, docs: List[Document]) -> str:
        if not docs:
            return "Brak dodatkowego kontekstu z bazy wiedzy."

        lines = []
        for idx, doc in enumerate(docs, start=1):
            title = doc.metadata.get("section_title", "Brak tytułu")
            trait_type = doc.metadata.get("trait_type", "Brak typu")
            source = doc.metadata.get("source_citation", "Brak źródła")
            lines.append(
                f"{idx}. [{title}] ({trait_type})\n"
                f"Źródło: {source}\n"
                f"{doc.page_content}"
            )
        return "\n\n".join(lines)

    def format_fragment_matches(self, matches: List[Dict[str, Any]]) -> str:
        """
        Pełniejszy format dopasowań. Używany pomocniczo, głównie do debugowania.
        Do promptu pełnej analizy przekazywana jest krótsza wersja z
        format_fragment_matches_for_prompt(), żeby nie przekroczyć limitu kontekstu.
        """
        if not matches:
            return "Brak dopasowań fragmentów do bazy wiedzy."

        blocks = []
        for item in matches:
            fragment = item["fragment"]
            docs = item.get("matches", [])
            note = item.get("note", "")

            if not docs:
                blocks.append(
                    f'Fragment wypowiedzi: "{fragment}"\n'
                    f"Dopasowania: {note}"
                )
                continue

            matched_lines = []
            for match in docs:
                doc = match["document"]
                title = doc.metadata.get("section_title", "Brak tytułu")
                trait_type = doc.metadata.get("trait_type", "Brak typu")
                score = match.get("score")
                strength = match.get("strength", "brak oceny")
                matched_lines.append(
                    f"- [{title}] ({trait_type})\n"
                    f"  Wynik podobieństwa: {score:.3f}\n"
                    f"  Siła dopasowania: {strength}\n"
                    f"  {doc.page_content}"
                )

            block = (
                f'Fragment wypowiedzi: "{fragment}"\n'
                f"Kandydackie dopasowania:\n" + "\n".join(matched_lines)
            )
            blocks.append(block)

        return "\n\n".join(blocks)

    def _compact_rag_doc_for_prompt(self, doc: Document) -> str:
        """Zwraca krótką reprezentację dokumentu RAG do promptu modelu."""
        title = doc.metadata.get("section_title", "Brak tytułu")
        trait_type = doc.metadata.get("trait_type", "Brak typu")
        text = normalize_spaces(doc.page_content)

        description = ""
        indicators = ""

        desc_match = re.search(r"Opis:\s*(.*?)(?:Wskaźniki:|Interpretacja:|---|$)", text, flags=re.DOTALL)
        ind_match = re.search(r"Wskaźniki:\s*(.*?)(?:Interpretacja:|---|$)", text, flags=re.DOTALL)

        if desc_match:
            description = truncate_text(desc_match.group(1), 220)
        else:
            description = truncate_text(text, 220)

        if ind_match:
            indicators = truncate_text(ind_match.group(1), 180)

        if indicators:
            return f"[{title}] ({trait_type}) — {description} Wskaźniki: {indicators}"
        return f"[{title}] ({trait_type}) — {description}"

    def format_retrieved_context_for_prompt(self, docs: List[Document]) -> str:
        """Skrócony kontekst RAG przekazywany do modelu."""
        if not docs:
            return "Brak dodatkowego kontekstu z bazy wiedzy."

        lines = []
        for idx, doc in enumerate(docs[:TOP_K_RAG], start=1):
            lines.append(f"{idx}. {self._compact_rag_doc_for_prompt(doc)}")
        return "\n".join(lines)

    def format_fragment_matches_for_prompt(self, matches: List[Dict[str, Any]]) -> str:
        """
        Skrócone dopasowania fragmentów do promptu modelu.

        """
        if not matches:
            return (
                "CECHY WSPIERANE ZAAKCEPTOWANYMI DOPASOWANIAMI:\n"
                "- brak\n\n"
                "Brak dopasowań fragmentów do bazy wiedzy."
            )

        accepted_by_trait: Dict[str, Dict[str, Any]] = {}
        example_blocks: List[str] = []
        fragments_without_acceptance = 0

        for item in matches:
            fragment = item.get("fragment", "")
            accepted = item.get("matches", [])

            if not accepted:
                fragments_without_acceptance += 1
                continue

            short_fragment = truncate_text(fragment, 140)
            example_trait_lines = []

            for match in accepted[:TOP_K_MATCHES_PER_FRAGMENT]:
                doc = match["document"]
                title = doc.metadata.get("section_title", "Brak tytułu")
                trait_type = doc.metadata.get("trait_type", "Brak typu")
                score = float(match.get("score", 0.0))
                strength = match.get("strength", "brak oceny")
                key = f"[{title}] ({trait_type})"

                if key not in accepted_by_trait:
                    accepted_by_trait[key] = {
                        "scores": [],
                        "strengths": set(),
                        "fragments": 0,
                    }

                accepted_by_trait[key]["scores"].append(score)
                accepted_by_trait[key]["strengths"].add(strength)
                accepted_by_trait[key]["fragments"] += 1

                example_trait_lines.append(
                    f"{key}, wynik={score:.3f}, siła={strength}"
                )

            if example_trait_lines and len(example_blocks) < 2:
                example_blocks.append(
                    f'Fragment: "{short_fragment}"\n'
                    f"Zaakceptowane dopasowania:\n- " + "\n- ".join(example_trait_lines)
                )

        lines: List[str] = ["CECHY WSPIERANE ZAAKCEPTOWANYMI DOPASOWANIAMI:"]

        if accepted_by_trait:
            sorted_traits = sorted(
                accepted_by_trait.items(),
                key=lambda kv: (kv[1]["fragments"], max(kv[1]["scores"])),
                reverse=True,
            )

            for trait_label, data in sorted_traits:
                scores = sorted(data["scores"], reverse=True)
                score_text = ", ".join(f"{score:.3f}" for score in scores[:4])
                strengths = ", ".join(sorted(data["strengths"]))
                lines.append(
                    f"- {trait_label}: {data['fragments']} fragment(y), "
                    f"wyniki podobieństwa: {score_text}, siła: {strengths}"
                )
        else:
            lines.append("- brak")

        lines.append("")
        lines.append(
            f"Fragmenty bez zaakceptowanego dopasowania powyżej progu: "
            f"{fragments_without_acceptance}."
        )
        lines.append(
            "Uwaga: odrzucone dopasowania poniżej progu nie są materiałem "
            "dowodowym dla sekcji 2.1."
        )

        if example_blocks:
            lines.append("")
            lines.append("KRÓTKIE PRZYKŁADY ZAAKCEPTOWANYCH DOPASOWAŃ, MAKSYMALNIE 2:")
            lines.extend(example_blocks)

        return "\n".join(lines)

    # -------------------------
    # ANALIZA
    # -------------------------

    def summarize_symptoms(self, symptoms_raw: str) -> str:
        if not symptoms_raw.strip():
            return (
                "1. Powód zgłoszenia: użytkownik nie podał wyraźnego powodu.\n"
                "2. Zauważane objawy: brak danych."
            )

        raw = self.symptoms_summary_chain.invoke(
            {"symptoms_raw": truncate_text(symptoms_raw, 1200)}
        )
        cleaned = clean_model_output(raw)
        return cleaned if cleaned else "Nie udało się wygenerować podsumowania objawów."

    def _prepare_analysis_inputs(self, values: Dict) -> Dict:
        """
        Przygotowuje skrócone wejście do modelu.

        Pełne dane nadal są zapisywane w raporcie TXT, ale do promptu trafiają
        tylko skondensowane informacje. Zapobiega to przekroczeniu limitu
        context_length=4096 i ogranicza ryzyko degeneracji generowanego tekstu.
        """
        return {
            "symptoms_text": truncate_text(values["symptoms_raw"], 600),
            "symptoms_summary": truncate_text(values["symptoms_summary"], 500),
            "conversation_text": truncate_text(values["conversation_text"], 750),
            "rag_context": truncate_text(
                self.format_retrieved_context_for_prompt(values["retrieved_context"]), 750
            ),
            # Dopasowania do promptu są już agregowane i zaczynają się od cech
            # zaakceptowanych, dlatego można zachować nieco większy limit bez
            # ryzyka, że model zobaczy głównie odrzucone kandydaty.
            "fragment_matches": truncate_text(
                self.format_fragment_matches_for_prompt(values["fragment_matches"]), 1200
            ),
        }

    def _looks_degenerate(self, text: str) -> bool:
        """Wykrywa typowe objawy zapętlenia lub zdegenerowanej generacji."""
        if not text:
            return True

        normalized = normalize_spaces(text.lower())

        # Przykład zaobserwowanej degeneracji lokalnego modelu.
        if "mogłat" in normalized or "mógłat" in normalized:
            return True

        words = normalized.split()
        if len(words) < 40:
            return False

        # Zbyt częste powtórzenie jednego tokenu.
        most_common_count = max(words.count(w) for w in set(words))
        if most_common_count / max(len(words), 1) > 0.18:
            return True

        # Długie powtórzenia identycznych fraz.
        repeated_phrase_pattern = r"(\b[\wąćęłńóśźż-]+\s+[\wąćęłńóśźż-]+\b)(?:\s+\1){3,}"
        if re.search(repeated_phrase_pattern, normalized):
            return True

        return False

    def _fallback_full_analysis(
        self,
        symptoms_summary: str,
        fragment_matches: List[Dict[str, Any]],
    ) -> str:
        """Bezpieczna analiza awaryjna, gdy model zwróci pusty lub zdegenerowany tekst."""
        supported_traits = []
        context_only_traits = []

        for item in fragment_matches:
            for match in item.get("matches", []):
                doc = match["document"]
                title = doc.metadata.get("section_title", "Brak tytułu")
                trait_type = doc.metadata.get("trait_type", "Brak typu")
                score = match.get("score", 0.0)
                strength = match.get("strength", "brak oceny")
                label = f"[{title}] ({trait_type}), wynik={score:.3f}, siła={strength}"
                if label not in supported_traits:
                    supported_traits.append(label)

            for match in item.get("rejected_matches", []):
                doc = match["document"]
                title = doc.metadata.get("section_title", "Brak tytułu")
                trait_type = doc.metadata.get("trait_type", "Brak typu")
                label = f"[{title}] ({trait_type})"
                if label not in context_only_traits and all(label not in s for s in supported_traits):
                    context_only_traits.append(label)

        if not supported_traits:
            supported_text = "Brak cech wspartych zaakceptowanymi dopasowaniami fragmentów."
        else:
            supported_text = "\n".join(f"- {trait}" for trait in supported_traits[:5])

        if not context_only_traits:
            context_text = "Brak dodatkowych cech do odnotowania jako niepotwierdzone."
        else:
            context_text = "\n".join(f"- {trait}" for trait in context_only_traits[:5])

        return (
            "1. Krótkie podsumowanie danych\n"
            f"{truncate_text(symptoms_summary, 500)}\n\n"
            "2. Cechy językowe i narracyjne możliwe do rozważenia\n"
            "2.1. Cechy wsparte zaakceptowanymi dopasowaniami fragmentów\n"
            f"{supported_text}\n"
            "2.2. Cechy obecne w kontekście RAG, ale niepotwierdzone w dopasowaniach fragmentów\n"
            f"{context_text}\n\n"
            "3. Fragmenty wypowiedzi warte uwagi\n"
            "Fragmenty należy ocenić na podstawie sekcji dopasowań w raporcie, ponieważ generacja modelu została zastąpiona analizą awaryjną.\n\n"
            "4. Dominujące sygnały widoczne w wypowiedzi\n"
            "Wypowiedź zawiera przykłady trudności z utrzymaniem uwagi i porządkowaniem treści, ale raport nie stanowi diagnozy.\n\n"
            "5. Ostrożny wniosek badawczy i zastrzeżenie metodologiczne\n"
            "Na podstawie dostępnych danych można traktować wynik jako pomocniczy materiał badawczy do oceny działania prototypu. "
            "Analiza nie zastępuje diagnozy klinicznej ani konsultacji ze specjalistą. "
            "Jeżeli opisywane trudności realnie utrudniają codzienne funkcjonowanie, warto omówić je ze specjalistą."
        )


    def _postprocess_full_analysis(self, text: str) -> str:
        """Porządkuje pełną analizę i zabezpiecza raport przed uciętym zakończeniem."""
        if not text:
            return ""

        text = text.strip()

        match = re.search(r"\n\s*5\.\s+", text)
        if match:
            text = text[:match.start()].rstrip()

        if not re.search(r"(?m)^\s*4\.\s+", text):
            text = text.rstrip() + (
                "\n\n4. Ostrożny wniosek badawczy i zastrzeżenie metodologiczne\n"
                "Analiza wskazuje wyłącznie cechy możliwe do rozważenia na podstawie dostępnej wypowiedzi i kontekstu RAG. "
                "Jeżeli opisywane trudności utrudniają codzienne funkcjonowanie, warto omówić je ze specjalistą. "
                "Wynik należy traktować jako pomocniczy materiał badawczy, a nie diagnozę."
            )

        # Jeśli tekst wygląda na urwany, usuwamy ostatnie niedokończone zdanie
        # i dodajemy stałe domknięcie. To zabezpiecza raport przed końcówkami
        # typu „należy traktować te ob”.
        final_sentence = "Wynik należy traktować jako pomocniczy materiał badawczy, a nie diagnozę."
        if final_sentence not in text:
            stripped = text.rstrip()
            if stripped and stripped[-1] not in ".!?":
                last_punct = max(stripped.rfind("."), stripped.rfind("!"), stripped.rfind("?"))
                if last_punct > 0:
                    stripped = stripped[:last_punct + 1]
                text = stripped.rstrip() + " " + final_sentence
            else:
                text = stripped + " " + final_sentence

        return text.strip()

    def generate_full_analysis(
        self,
        symptoms_raw: str,
        symptoms_summary: str,
        conversation_text: str,
        retrieved_context: List[Document],
        fragment_matches: List[Dict[str, Any]],
    ) -> str:
        raw = self.full_analysis_chain.invoke(
            {
                "symptoms_raw": symptoms_raw,
                "symptoms_summary": symptoms_summary,
                "conversation_text": conversation_text,
                "retrieved_context": retrieved_context,
                "fragment_matches": fragment_matches,
            }
        )
        cleaned = clean_model_output(raw)

        if cleaned and not self._looks_degenerate(cleaned):
            return self._postprocess_full_analysis(cleaned)

        return self._fallback_full_analysis(
            symptoms_summary=symptoms_summary,
            fragment_matches=fragment_matches,
        )

    def generate_user_summary(self, full_analysis: str) -> str:
        raw = self.user_summary_chain.invoke(
            {"analysis": truncate_text(full_analysis, 2200)}
        )
        cleaned = clean_model_output(raw)

        if not cleaned:
            return (
                "W Twojej wypowiedzi widać kilka cech, które mogą sugerować trudności z organizacją uwagi, "
                "utrzymaniem wątku i regulacją napięcia. Taki wynik nie stanowi diagnozy, ale może być "
                "powodem do rozważenia konsultacji ze specjalistą, jeśli te trudności realnie wpływają na Twoje życie."
            )

        sentences = re.split(r"(?<=[\.\!\?])\s+", cleaned)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) > 4:
            cleaned = " ".join(sentences[:4])

        return cleaned

    # -------------------------
    # RAPORT
    # -------------------------

    def create_report_payload(
        self,
        symptoms_raw: str,
        symptoms_summary: str,
        conversation_turns: List[Dict[str, str]],
        retrieved_context: List[Document],
        fragment_matches: List[Dict[str, Any]],
        full_analysis: str,
        user_summary: str,
    ) -> Dict:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        serialized_retrieved_context = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in retrieved_context
        ]

        serialized_fragment_matches = []
        for item in fragment_matches:
            serialized_fragment_matches.append(
                {
                    "fragment": item["fragment"],
                    "min_score": item.get("min_score", MIN_MATCH_SCORE),
                    "note": item.get("note", ""),
                    "matches": [
                        {
                            "page_content": match["document"].page_content,
                            "metadata": match["document"].metadata,
                            "score": match["score"],
                            "strength": match["strength"],
                        }
                        for match in item.get("matches", [])
                    ],
                    "rejected_matches": [
                        {
                            "page_content": match["document"].page_content,
                            "metadata": match["document"].metadata,
                            "score": match["score"],
                            "strength": match["strength"],
                        }
                        for match in item.get("rejected_matches", [])
                    ],
                }
            )

        return {
            "timestamp": timestamp,
            "local_model_name": MODEL_DISPLAY_NAME,
            "local_model_path": MODEL_PATH,
            "model_type": MODEL_TYPE,
            "rag_file_name": RAG_DISPLAY_NAME,
            "rag_file_path": RAG_FILE_PATH,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "llm_parameters": {
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": 0.2,
                "top_p": 0.9,
                "repetition_penalty": REPETITION_PENALTY,
                "context_length": CONTEXT_LENGTH,
                "gpu_layers": GPU_LAYERS,
            },
            "rag_parameters": {
                "top_k_rag": TOP_K_RAG,
                "fetch_k": RAG_FETCH_K,
                "lambda_mult": RAG_LAMBDA_MULT,
                "top_k_matches_per_fragment": TOP_K_MATCHES_PER_FRAGMENT,
                "min_fragment_words": MIN_FRAGMENT_WORDS,
                "max_fragment_words": MAX_FRAGMENT_WORDS,
                "min_match_score": MIN_MATCH_SCORE,
            },
            "symptoms_raw": symptoms_raw,
            "symptoms_summary": symptoms_summary,
            "conversation_turns": conversation_turns,
            "retrieved_context": serialized_retrieved_context,
            "fragment_matches": serialized_fragment_matches,
            "full_analysis": full_analysis,
            "user_summary": user_summary,
            "langchain_enabled": True,
        }

    def save_report_files(self, report_data: Dict) -> Dict[str, str]:
        """
        Zapisuje wyłącznie raport TXT.

        Wersja JSON została usunięta, aby implementacja była zgodna z założeniem,
        że finalnym artefaktem badawczym prototypu jest czytelny raport tekstowy.
        """
        ensure_reports_dir()
        timestamp = report_data["timestamp"]
        txt_path = os.path.join(REPORTS_DIR, f"report_{timestamp}.txt")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=== RAPORT BADAWCZY ===\n\n")
            f.write(f"Model lokalny: {report_data.get('local_model_name', MODEL_DISPLAY_NAME)}\n")
            f.write(f"Typ modelu: {report_data.get('model_type', MODEL_TYPE)}\n")
            f.write(f"Ścieżka modelu: {report_data.get('local_model_path', MODEL_PATH)}\n")
            f.write(f"Baza wiedzy RAG: {report_data.get('rag_file_name', RAG_DISPLAY_NAME)}\n")
            f.write(f"Ścieżka bazy wiedzy RAG: {report_data.get('rag_file_path', RAG_FILE_PATH)}\n")
            f.write(f"Model osadzeń: {report_data.get('embedding_model', EMBEDDING_MODEL_NAME)}\n")

            llm_params = report_data.get("llm_parameters", {})
            rag_params = report_data.get("rag_parameters", {})

            f.write("\nParametry modelu językowego:\n")
            for key, value in llm_params.items():
                f.write(f"- {key}: {value}\n")

            f.write("\nParametry RAG i dopasowań fragmentów:\n")
            for key, value in rag_params.items():
                f.write(f"- {key}: {value}\n")

            f.write("\n1. Surowa wypowiedź użytkownika o objawach\n")
            f.write(report_data["symptoms_raw"] + "\n\n")

            f.write("2. Podsumowanie objawów\n")
            f.write(report_data["symptoms_summary"] + "\n\n")

            f.write("3. Rozmowa swobodna z wypowiedziami użytkownika i bota\n")
            role_labels = {
                "user": "Użytkownik",
                "assistant": "Bot",
            }
            for turn in report_data["conversation_turns"]:
                role = role_labels.get(turn.get("role", ""), turn.get("role", "Nieznana rola"))
                text = turn.get("text", "")
                f.write(f"- {role}: {text}\n")

            f.write("\n4. Retrieval z RAG\n")
            if report_data["retrieved_context"]:
                for idx, item in enumerate(report_data["retrieved_context"], start=1):
                    metadata = item.get("metadata", {})
                    section_title = metadata.get("section_title", "Brak tytułu")
                    trait_type = metadata.get("trait_type", "Brak typu")
                    source = metadata.get("source_citation", "Brak źródła")

                    f.write(f"{idx}. [{section_title}] ({trait_type})\n")
                    f.write(f"Źródło: {source}\n")
                    f.write(f"{item['page_content']}\n\n")
            else:
                f.write("Brak dodatkowego kontekstu z bazy wiedzy.\n\n")

            f.write("5. Wstepne dopasowania fragmentów rozmowy swobodnej do RAG\n")
            if report_data["fragment_matches"]:
                for item in report_data["fragment_matches"]:
                    f.write(f'Fragment wypowiedzi: "{item["fragment"]}"\n')
                    f.write(f"Minimalny próg podobieństwa: {item.get('min_score', MIN_MATCH_SCORE)}\n")

                    if item.get("matches"):
                        f.write("Dopasowania zaakceptowane:\n")
                        for match in item["matches"]:
                            metadata = match.get("metadata", {})
                            section_title = metadata.get("section_title", "Brak tytułu")
                            trait_type = metadata.get("trait_type", "Brak typu")
                            f.write(f"- [{section_title}] ({trait_type})\n")
                            f.write(f"  Wynik podobieństwa: {match.get('score', 0):.3f}\n")
                            f.write(f"  Siła dopasowania: {match.get('strength', 'brak oceny')}\n")
                            f.write(f"  {match['page_content']}\n")
                    else:
                        f.write(f"Dopasowania zaakceptowane: {item.get('note', 'brak')}\n")

                    if item.get("rejected_matches"):
                        f.write("Dopasowania odrzucone poniżej progu:\n")
                        for match in item["rejected_matches"]:
                            metadata = match.get("metadata", {})
                            section_title = metadata.get("section_title", "Brak tytułu")
                            trait_type = metadata.get("trait_type", "Brak typu")
                            f.write(
                                f"- [{section_title}] ({trait_type}), "
                                f"wynik podobieństwa={match.get('score', 0):.3f}\n"
                            )
                    f.write("\n")
            else:
                f.write("Brak dopasowań fragmentów do bazy wiedzy.\n\n")

            f.write("6. Pełna analiza\n")
            f.write(report_data["full_analysis"] + "\n\n")

            f.write("7. Krótkie podsumowanie dla użytkownika\n")
            f.write(report_data["user_summary"] + "\n")

        return {"txt_path": txt_path}


# ============================================================
# WALIDACJA
# ============================================================

def check_requirements() -> str | None:
    if not os.path.exists(MODEL_PATH):
        return f"Nie znaleziono modelu GGUF pod ścieżką: {MODEL_PATH}"
    if not os.path.exists(RAG_FILE_PATH):
        return f"Nie znaleziono pliku wiedzy RAG: {RAG_FILE_PATH}"
    return None
