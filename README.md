"# Praca-magisterska" 
# Prototyp systemu analizy wypowiedzi tekstowych z wykorzystaniem LLM i RAG

Repozytorium zawiera materiały badawcze oraz prototyp systemu przygotowany w ramach pracy magisterskiej. Celem projektu jest eksperymentalne sprawdzenie możliwości wykorzystania dużego modelu językowego oraz podejścia RAG do analizy swobodnych wypowiedzi tekstowych użytkownika.

Projekt koncentruje się na identyfikacji wybranych cech wypowiedzi, takich jak powtórzenia, emocjonalne wzmocnienia oraz przeskoki tematyczne. Rozwiązanie ma charakter badawczy i nie stanowi narzędzia diagnostycznego.

## Struktura repozytorium

```text
.
├── badanie_1/
├── badanie_2/
├── badanie_3/
├── badanie_4/
├── badanie_5/
├── prototyp/
└── README.md
```

## Opis folderów

### `badanie_1/`

Folder zawiera pliki związane z pierwszym badaniem eksperymentalnym. Badanie dotyczyło podstawowej klasyfikacji wypowiedzi pod kątem występowania wybranych cech językowych.

### `badanie_2/`

Folder zawiera pliki związane z drugim badaniem eksperymentalnym. Badanie służyło dalszej weryfikacji działania przygotowanego podejścia oraz analizie jakości uzyskiwanych odpowiedzi.

### `badanie_3/`

Folder zawiera pliki związane z trzecim badaniem eksperymentalnym, obejmującym porównanie działania systemu z wykorzystaniem mechanizmu RAG oraz bez jego użycia.

### `badanie_4/`

Folder zawiera pliki związane z czwartym badaniem eksperymentalnym. Badanie dotyczyło stabilności działania systemu oraz powtarzalności generowanych wyników.

### `badanie_5/`

Folder zawiera pliki związane z piątym badaniem eksperymentalnym, obejmującym analizę przypadków granicznych i nietypowych danych wejściowych.

### `prototyp/`

Folder zawiera kod źródłowy prototypu systemu, szablony poleceń, elementy bazy wiedzy RAG oraz pliki potrzebne do uruchomienia eksperymentalnego potoku analizy wypowiedzi tekstowych.

## Zakres analizy

System analizuje wypowiedzi tekstowe pod kątem wybranych cech jezykowych.

Wyniki działania systemu mają charakter eksperymentalny i służą ocenie zaproponowanego podejścia w kontekście pracy magisterskiej.

## Wykorzystany model językowy

Repozytorium nie zawiera plików dużego modelu językowego. Modele tego typu mają duży rozmiar i są udostępniane oddzielnie, między innymi na platformie Hugging Face.

W ramach eksperymentów wykorzystano instrukcyjną wersję modelu Bielik 11B. Model ten nie jest dołączony do repozytorium i powinien zostać pobrany lub skonfigurowany oddzielnie zgodnie z dokumentacją dostawcy modelu oraz wykorzystywanego środowiska uruchomieniowego.

Przykładowo, w środowisku lokalnym należy zapewnić dostęp do odpowiedniego modelu językowego oraz dostosować ścieżki lub parametry konfiguracyjne w kodzie prototypu.

## Uruchomienie prototypu

Aby uruchomić prototyp, należy przejść do folderu `prototyp/`:

```bash
cd prototyp
```

Następnie zainstalować wymagane zależności:

```bash
pip install -r requirements.txt
```

Uruchomienie programu:

```bash
python app.py
```

Jeżeli nazwa głównego pliku jest inna, należy dostosować powyższe polecenie do faktycznej struktury projektu.

## Wymagania

Projekt został przygotowany z wykorzystaniem języka Python. Wymagane biblioteki znajdują się w pliku:

```text
prototyp/requirements.txt
```

Do odtworzenia działania prototypu potrzebne jest również środowisko umożliwiające uruchomienie lokalnego dużego modelu językowego lub połączenie z odpowiednim mechanizmem jego obsługi.

## Uwagi dotyczące danych i wyników

Materiały umieszczone w folderach `badanie_1–5` służą odtworzeniu przebiegu eksperymentów opisanych w pracy magisterskiej. Repozytorium może zawierać przykładowe dane wejściowe, wyniki eksperymentów, szablony poleceń oraz pliki pomocnicze.

W repozytorium nie należy umieszczać danych wrażliwych, prywatnych wypowiedzi użytkowników ani plików zawierających klucze dostępowe lub lokalne ustawienia środowiska.

## Charakter projektu

Repozytorium ma charakter badawczy i edukacyjny. Zawarte w nim rozwiązanie nie służy do diagnozowania zaburzeń ani do podejmowania decyzji medycznych. System analizuje wyłącznie wybrane cechy językowe wypowiedzi tekstowych na potrzeby eksperymentów opisanych w pracy magisterskiej.

## Autor

Projekt przygotowany w ramach pracy magisterskiej.
