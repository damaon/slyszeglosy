---
language: pl
task_categories:
- text-to-speech
license: cc-by-nc-4.0
pretty_name: słyszę głosy
---
# Zbiór danych słyszę głosy

Syntetyczny korpus polskiej mowy z dwoma głosami, `F` (kobiecy) i `M`
(męski), przygotowany do treningu modelu słyszę głosy. Każde zdanie czytają
oba głosy.

| | |
|---|---|
| Nagrania | 19 796 (po odsiewie Whisperem) |
| Czas | 19,2 h |
| Głosy | `F`, `M` |
| Dźwięk | 24 kHz mono |
| Język | polski |
| Licencja | CC-BY-NC 4.0 |

## Zawartość

| Podział | Zdania | Nagrania | Godziny |
|---|---:|---:|---:|
| train | 9 420 | 18 622 | 18,2 |
| validation | 300 | 586 | 0,5 |
| test | 300 | 588 | 0,5 |

Podział jest według zdań: oba odczyty zdania są zawsze w tym samym podziale.
Po kontroli jakości wymuszonego dopasowania do treningu weszło 18 268 nagrań
(walidacja 576, test 566).

## Jak powstał

1. **Głosy** zaprojektowano z opisu tekstowego w OmniVoice („female, young
   adult, moderate pitch” i „male, middle-aged, moderate pitch”). To nie są
   nagrania ani naśladowanie prawdziwych osób. Nagrania referencyjne i opisy są
   w `voices/`.
2. **Nagrania**: każde zdanie oboma głosami, OmniVoice 0.6B (`language="pl"`)
   z klonowaniem nagrań referencyjnych, partie po 4 zdania
   (`scripts/generate_audio_pl.py`).
3. **Odsiew**: transkrypcja Whisper large-v3-turbo. Nagrania z CER powyżej 5%
   (12% gdy zdanie ma cyfry) generowano drugi raz z innym ziarnem. Po dwóch
   próbach odrzucono 244 z 20 040 (`scripts/screen_audio_pl.py`).

Automatyczny odsiew łapie błędne i pominięte słowa. Nie mierzy naturalności,
a korpusu nie przesłuchano nagranie po nagraniu.

## Źródła tekstów

| Źródło | Zdania | Licencja tekstu |
|---|---:|---|
| Mozilla Common Voice PL, sentence collector | 7 101 | CC0 |
| Mozilla Common Voice PL, Europarl | 1 711 | CC0 |
| Szablony projektu: liczby, daty, godziny, kwoty, skróty, nazwiska | 900 | CC-BY-NC 4.0 |
| Zdania napisane dla projektu | 308 | CC-BY-NC 4.0 |

Zdania z Common Voice przefiltrowano pod kątem długości, kształtu,
interpunkcji, obcych słów i duplikatów (`scripts/build_corpus_pl.py`). 17%
zdań to pytania, 9% wykrzyknienia.

## Adnotacje

Długości głosek (w ramkach 10,7 ms) pochodzą z wymuszonego dopasowania
`facebook/wav2vec2-lv-60-espeak-cv-ft`. Głoski daje front-end `wymowa`. Mapę
etykiet alignera skalibrowano na danych
(`scripts/calibrate_aligner_labels.py`), a przesunięcie granic głosek
(−10,3 ms) na zbiorze walidacyjnym (`scripts/calibrate_aligner_timing.py`).
Progi kontroli jakości wyznaczono z rozkładu na pełnych danych:
średnia pewność tokenu ≥ 0,45, odsetek słabych głosek ≤ 0,35, dźwięczność
samogłosek ≥ 0,45.

## Ograniczenia

- Mowa jest syntetyczna i dziedziczy nawyki oraz błędy wymowy nauczyciela.
- Dwa głosy, jedna odmiana (standard warszawski), tylko mowa czytana.

## Licencja

CC-BY-NC 4.0, tylko użytek niekomercyjny, bo nagrania wygenerował OmniVoice,
którego wagi są na licencji CC-BY-NC. Teksty z Common Voice: CC0
(https://github.com/common-voice/common-voice).
