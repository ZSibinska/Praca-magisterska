from ctransformers import AutoModelForCausalLM
import os
from typing import Optional

# Folder z modelami
MODEL_FOLDER = r"ścieżka/do/modelu.gguf"

# Folder bazowy = folder, w którym znajduje się ten plik
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_text_file(filename: str, base_dir: Optional[str] = None) -> str:
    """
    Wczytuje zawartość pliku tekstowego z tego samego folderu co skrypt
    lub z podanej ścieżki bazowej.
    """
    if base_dir is None:
        base_dir = BASE_DIR

    file_path = os.path.join(base_dir, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Nie znaleziono pliku: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_full_prompt(system_prompt: str, user_prompt: str) -> str:
    """
    Łączy prompt systemowy i prompt użytkownika w jeden tekst wejściowy
    przekazywany do modelu.
    
    Forma może być później łatwo zmieniona, jeśli dany model lepiej działa
    z innym formatowaniem.
    """
    return f"""### Instrukcja systemowa:
{system_prompt}

### Polecenie użytkownika:
{user_prompt}

### Odpowiedź:
"""


def run_local_model(
    model_file: str,
    model_type: str,
    system_prompt: str,
    user_prompt: str,
    gpu_layers: int = 0,
    context_length: int = 4096,
    max_new_tokens: int = 2000
) -> str:
    """
    Uruchamia lokalny model na podstawie:
    - pliku modelu
    - typu modelu
    - promptu systemowego
    - promptu użytkownika
    """
    model_path = os.path.join(MODEL_FOLDER, model_file)
    print("Ładuję model z:", model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        model_file=model_file,
        model_type=model_type,
        gpu_layers=gpu_layers,
        context_length=context_length
    )

    full_prompt = build_full_prompt(system_prompt, user_prompt)
    output = model(full_prompt, max_new_tokens=max_new_tokens)
    return output


if __name__ == "__main__":
    # Parametry modelu
    # Konieczne do zmodyfikowania w ramach kolejnych testów
    model_file = "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
    model_type = "qwen"

    # Nazwy plików z promptami
    system_prompt_file = "system_prompt.txt"
    user_prompt_file = "user_prompt.txt"

    # Wczytanie promptów z plików
    system_prompt = load_text_file(system_prompt_file)
    user_prompt = load_text_file(user_prompt_file)

    # Uruchomienie modelu
    result = run_local_model(
        model_file=model_file,
        model_type=model_type,
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )

    print("\n=== ODPOWIEDŹ MODELU ===\n")
    print(result)