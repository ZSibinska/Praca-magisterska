import os
import json
import time
from pathlib import Path
from typing import List, Dict

from langchain_community.llms import CTransformers


# =========================
# Konfiguracja - dopasowana do lokalnego uruchomienia
# =========================

MODEL_PATH = r"ścieżka/do/modelu.gguf"

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

WYPOWIEDZI_DIR = BASE_DIR / "Wypowiedzi"
OUTPUT_DIR = BASE_DIR / "Wyniki"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


PROMPT_TEMPLATE_WITH_DEFINITIONS = """Przeanalizuj wypowiedź użytkownika.

Definicje:
- Powtórzenia: dosłowne ponowne użycie tego samego słowa lub tego samego wyrażenia w jednej wypowiedzi, np. „bardzo, bardzo”, „ten raport… ten raport”, „dobre, naprawdę dobre”. Nie oznaczaj powtórzeń, jeśli w wypowiedzi występują różne słowa o podobnym znaczeniu, różne określenia wzmacniające albo podobny ton wypowiedzi.
- Emocjonalne wzmocnienia: słowa lub wyrażenia wyraźnie wzmacniające emocję, ocenę albo subiektywne przeżycie mówiącego, np. „mega”, „strasznie”, „okropnie”, „totalnie”. Nie oznaczaj tej cechy tylko dlatego, że pojawia się słowo „bardzo” albo „naprawdę”; oceń, czy służy ono wyraźnemu wzmocnieniu emocji lub oceny.
- Przeskoki tematyczne: przechodzenie między różnymi, słabo powiązanymi wątkami, czynnościami lub tematami w jednej wypowiedzi. Nie oznaczaj przeskoku tematycznego, jeśli wypowiedź opisuje kilka czynności, ale są one logicznie połączone jednym kontekstem, np. jednym projektem, jednym zadaniem albo chronologicznym opisem dnia.
Jeżeli nie masz pewności, czy cecha występuje, wybierz NIE.

Zasady:
- Odpowiedz dokładnie trzema liniami.
- Nie dodawaj wyjaśnień.
- Nie powtarzaj wypowiedzi.
- Nie dopisuj żadnego innego tekstu.

Format odpowiedzi:
powtórzenia: TAK/NIE
emocjonalne wzmocnienia: TAK/NIE
przeskoki tematyczne: TAK/NIE

Wypowiedź:
{tekst}
"""

# =========================
# Funkcje pomocnicze
# =========================

def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_text_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Nie znaleziono pliku: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        return file.read().strip()


def get_ordered_text_files(folder_path: str) -> List[Path]:
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Nie znaleziono folderu: {folder_path}")

    txt_files = sorted(
        folder.glob("*.txt"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem
    )

    if not txt_files:
        raise FileNotFoundError(f"W folderze nie znaleziono plików .txt: {folder_path}")

    return txt_files


# =========================
# Model językowy
# =========================

def build_llm(model_path: str) -> CTransformers:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Nie znaleziono modelu: {model_path}")

    llm = CTransformers(
        model=model_path,
        model_type="llama",
        config={
            "max_new_tokens": 64,
            "temperature": 0.0,
            "context_length": 4096,
            "gpu_layers": 0,
            "threads": 8,
        },
    )
    return llm


# =========================
# Testowanie modelu
# =========================

def test_model_on_texts_with_definitions(
    llm: CTransformers,
    folder_path: str,
    prompt_template: str,
) -> List[Dict]:
    results = []
    text_files = get_ordered_text_files(folder_path)

    for idx, file_path in enumerate(text_files, start=1):
        text = load_text_file(str(file_path))
        prompt = prompt_template.format(tekst=text)

        print(f"\n--- Przetwarzanie pliku {file_path.name} ({idx}/{len(text_files)}) ---")

        start_time = time.time()
        response = llm.invoke(prompt)
        elapsed_time = round(time.time() - start_time, 2)

        result = {
            "numer": idx,
            "plik": file_path.name,
            "wypowiedz": text,
            "prompt": prompt,
            "odpowiedz_modelu": response.strip(),
            "czas_s": elapsed_time,
        }

        results.append(result)

        print("Odpowiedź modelu:")
        print(response.strip())
        print(f"Czas: {elapsed_time} s")

    return results


# =========================
# Zapis wyników
# =========================
def save_results_to_txt(results: List[Dict], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as file:
        for item in results:
            file.write("=" * 80 + "\n")
            file.write(f"Numer: {item['numer']}\n")
            file.write(f"Plik: {item['plik']}\n")
            file.write(f"Czas odpowiedzi: {item['czas_s']} s\n\n")
            file.write("Wypowiedź:\n")
            file.write(item["wypowiedz"] + "\n\n")
            file.write("Odpowiedź modelu:\n")
            file.write(item["odpowiedz_modelu"] + "\n\n")


# =========================
# Uruchomienie
# =========================

if __name__ == "__main__":
    ensure_directory(OUTPUT_DIR)

    print("Ładowanie modelu językowego...")
    llm = build_llm(MODEL_PATH)

    print("Uruchamianie testu z definicjami w prompcie...")
    results = test_model_on_texts_with_definitions(
        llm=llm,
        folder_path=WYPOWIEDZI_DIR,
        prompt_template=PROMPT_TEMPLATE_WITH_DEFINITIONS,
    )

    json_output = os.path.join(OUTPUT_DIR, "wyniki_badanie2.json")
    txt_output = os.path.join(OUTPUT_DIR, "wyniki_badanie2.txt")

    save_results_to_txt(results, txt_output)

    print("\nZakończono test.")
    print(f"Wyniki TXT:  {txt_output}")