---
license: apache-2.0
language:
- pl
pipeline_tag: text-to-speech
tags:
- text-to-speech
- onnx
- polish
datasets:
- czyzi0/the-mc-speech-dataset
---
# słyszę głosy V2

Mały polski model syntezy mowy (TTS) z jednym głosem: `MC`, prawdziwym głosem
lektora ze zbioru MC Speech. Cały model, razem z generatorem fali, ma 4,5 mln
parametrów i został wytrenowany od zera na 20,7 h mowy. Na czterech wątkach
CPU działa ok. 150× szybciej niż w czasie rzeczywistym.

**Wagi są na licencji Apache-2.0**, tak jak kod: można ich używać w dowolnym
celu, także komercyjnym, z zachowaniem informacji o licencji i autorach.

<audio controls src="https://huggingface.co/Damaon/slyszeglosy-v2/resolve/main/samples/lokomotywa-MC.wav"></audio>

- Kod: https://github.com/damaon/slyszeglosy
- Poprzednia wersja (dwa syntetyczne głosy, CC-BY-NC): https://huggingface.co/Damaon/slyszeglosy

| | |
|---|---|
| Parametry | 4 533 067 |
| Głos | `MC` (polski, męski, prawdziwy lektor) |
| Wyjście | 24 kHz mono |
| Wejście | Tekst zamieniany na głoski przez `wymowa` (reguły, bez słownika zewnętrznego) |
| Wagi wstępnie wytrenowane w modelu | Brak |
| Licencja | Apache-2.0 (wagi i kod) |

## Pliki

| Plik | Rozmiar | Zawartość |
|---|---:|---|
| `slyszeglosy-v2.pt` | 18,2 MB | Wagi PyTorch, float32 |
| `onnx/slyszeglosy-v2.onnx` | 22,4 MB | Eksport ONNX, float32, opset 17 |
| `onnx/slyszeglosy-v2.json` | | Metadane eksportu: słownik głosek, głosy, wejścia i wyjścia |
| `config.json` | | Ustawienia modelu |
| `LICENSE`, `NOTICE` | | Tekst licencji Apache-2.0 i informacja o autorach |
| `samples/` | | Próbki z tej strony |

SHA256:

- `slyszeglosy-v2.pt`: `a135a55ef5bccb2972d95fbe0bc25854d587b7a67e0e2db47ce24dd8905273af`
- `onnx/slyszeglosy-v2.onnx`: `90eb68a6c2db85d42bf86c1ffe44439e71ce3ba88d6c3c79b392c704fcef32d0`

## Próbki

| Tekst | `MC` |
|---|---|
| Dzień dobry! Spotkanie zaczyna się o 15:30 w sali nr 4. | <audio controls src="https://huggingface.co/Damaon/slyszeglosy-v2/resolve/main/samples/MC-1.wav"></audio> |
| Czy możesz kupić 2 kg jabłek i 1,5 l mleka? | <audio controls src="https://huggingface.co/Damaon/slyszeglosy-v2/resolve/main/samples/MC-2.wav"></audio> |
| Zadzwoń pod numer 600 123 456 albo napisz maila. | <audio controls src="https://huggingface.co/Damaon/slyszeglosy-v2/resolve/main/samples/MC-3.wav"></audio> |
| Tuwim, „Lokomotywa” (początek) | <audio controls src="https://huggingface.co/Damaon/slyszeglosy-v2/resolve/main/samples/lokomotywa-MC.wav"></audio> |
| Mickiewicz, „Pan Tadeusz”, Inwokacja (początek) | <audio controls src="https://huggingface.co/Damaon/slyszeglosy-v2/resolve/main/samples/inwokacja-MC.wav"></audio> |

## Użycie

### PyTorch

```bash
pip install git+https://github.com/damaon/slyszeglosy
hf download Damaon/slyszeglosy-v2 --local-dir slyszeglosy-v2

python -m tiny_voca.infer --checkpoint slyszeglosy-v2/slyszeglosy-v2.pt \
  --text "Dzień dobry! Spotkanie zaczyna się o 15:30." --output dzien-dobry.wav
```

| Opcja | Domyślnie | Znaczenie |
|---|---|---|
| `--text-file` | | Tekst z pliku zamiast `--text` |
| `--voice` | `MC` | Jedyny głos tego modelu |
| `--speed` | 1.0 | Tempo mowy |
| `--pitch-shift` | 0 | Dodawane do log F0; 0,058 to ok. pół tonu |
| `--pitch-range` | 1.0 | Skaluje ruch melodii wokół średniej |

Tekst może mieć dowolną długość: jest czytany zdanie po zdaniu, a nagrania są
sklejane. Liczby, daty, godziny, kwoty, procenty, jednostki, numery telefonów,
skróty i skrótowce rozwija `wymowa.normalize`.

### ONNX

Graf przyjmuje numery głosek i akcent, a zwraca falę. Tekst zamienia się na
numery poza grafem, przez `tiny_voca.text.phonemize`.

```python
import numpy as np, onnxruntime as ort, soundfile as sf
from tiny_voca.text import phonemize

phones = phonemize("Dzień dobry! Tu słyszę głosy.")
one = lambda value: np.array([value], dtype=np.float32)
session = ort.InferenceSession("slyszeglosy-v2/onnx/slyszeglosy-v2.onnx")
audio, durations = session.run(None, {
    "tokens": np.array([phones.ids], dtype=np.int64),
    "stress": np.array([phones.stress], dtype=np.int64),
    "voice": np.array([0], dtype=np.int64),      # 0 = MC
    "speed": one(1.0), "pitch_shift": one(0.0), "pitch_range": one(1.0),
})
audio = audio[0] * min(1.0, 0.99 / np.abs(audio[0]).max())
sf.write("dzien-dobry.wav", audio, 24000)
```

Jedno wywołanie czyta jedną wypowiedź (do 512 głosek i 32 s). Wypowiedź
powinna kończyć się znakiem interpunkcyjnym. Ten sam plik działa w
onnxruntime-web; demo jest w katalogu `web/` repozytorium.

## Model

Architektura tiny-voca: enkoder transformerowy (4 bloki) czyta głoski z
akcentem, predyktor długości daje każdej głosce liczbę ramek, dwa transformery
ramkowe (2 i 5 bloków) przewidują wysokość, dźwięczność i energię, a potem
parametry syntezatora źródło-filtr (ciąg impulsów i szum kształtowane filtrami
cepstralnymi), bez neuronowego wokodera. 192 kanały, 4 głowice, RoPE, ramka
256 próbek (10,7 ms).

Nowość w V2: **przydech** (`aspiration` = 0,35 w `config.json`). Powyżej
1 kHz, narastająco do 4 kHz, 35% dźwięcznego źródła zastępuje szum o tej samej
obwiedni. Harmoniczne zgrane w fazie brzmią bez tego „brzęcząco”. Z
przydechem aperiodyczność mowy (WORLD D4C) zgadza się z nagraniami lektora we
wszystkich pasmach. To ustawienie syntezy, nie wymaga treningu.

Słownik ma 56 symboli: 39 polskich głosek (8 samogłosek z nosowymi
`ɛ̃ ɔ̃`, 31 spółgłosek), znaki specjalne i interpunkcję.

## Trening

| | |
|---|---|
| Dane | 22 914 nagrań, 20,7 h (walidacja 328, test 264) |
| Kroki | 104 000, od zera |
| Sprzęt | Jedna karta GPU (24 GB), 9 h, z termostatem GPU (`--max-temp 76`) |
| Straty | Wielorozdzielcze straty widmowe; od kroku 8000 strata adwersarialna (dyskryminatory wielookresowe i wielorozdzielcze) |

Dane:

1. **MC Speech** (Mateusz Czyżnikiewicz, CC0): 20 541 prawdziwych nagrań
   jednego lektora, 18,2 h. Nagrania sprowadzono do 24 kHz, przycięto ciszę
   i wyrównano głośność.
2. **Trudne przypadki**: 2373 nagrania, 2,5 h, w których MC Speech jest
   ubogi: liczby, daty, godziny, kwoty, jednostki, skróty, numery telefonów,
   pytania i wykrzyknienia. Teksty to szablony napisane dla projektu i zdania z
   Common Voice PL (CC0). Przeczytał je VoxCPM2 (OpenBMB, Apache-2.0),
   klonując głos lektora z jednego nagrania MC Speech. Zdania, które MC Speech
   już ma, usunięto.
3. **Odsiew**: każde nagranie przepisały dwa ASR, Whisper large-v3-turbo i
   NVIDIA Parakeet-TDT-0.6B-v3. Odrzucano nagranie, jeśli oba usłyszały coś
   innego niż tekst. Odpadło 1176 z 24 002 nagrań MC Speech i 76 z 3050
   trudnych przypadków (po jednej ponownej próbie).
4. **Adnotacja**: długości głosek z wymuszonego dopasowania
   `facebook/wav2vec2-lv-60-espeak-cv-ft`, z przesunięciem granic
   skalibrowanym na tym głosie. Kontrola jakości odrzuciła dalsze 1796
   nagrań, głównie z niewykrytą dźwięcznością samogłosek. Aligner służy tylko
   do adnotacji i nie jest częścią modelu.

## Ocena

CER i WER mowy z modelu w transkrypcji Whisper large-v3-turbo (język `pl`),
obok prawdziwych nagrań tych samych zdań.

| Zdania | Nagrania | CER model | CER lektor | WER model | WER lektor |
|---|---:|---:|---:|---:|---:|
| Walidacja | 150 | 0,56% | 0,40% | 2,9% | 2,2% |
| Test | 150 | 0,31% | 0,42% | 1,7% | 2,6% |
| Nowe zdania spoza danych | 9 | 0,0% | | 0,0% | |

Trudne przypadki, 150 zdań odłożonych przed treningiem, czytanych z zapisu z
cyframi i skrótami (CER, Parakeet / Whisper):

| Model | Parakeet | Whisper |
|---|---:|---:|
| V2 (`MC`) | 1,67% | 1,82% |
| V1 (`F`) | 3,23% | 1,86% |
| V1 (`M`) | 3,43% | 1,91% |
| VoxCPM2 (nauczyciel trudnych przypadków) | 1,53% | 1,83% |

Podobieństwo barwy do lektora (WavLM-SV, kosinus do średniej z nagrań MC
Speech): model 0,967 na teście, prawdziwe nagrania testowe 0,971, górna
granica (inne nagrania lektora) 0,976.

Ocena powstała przed włączeniem przydechu. Na 100 zdaniach testowych przydech
nie zmienił ani CER, ani podobieństwa barwy. To małe zbiory: wynik jest
kontrolą poprawności, nie benchmarkiem. Nie przeprowadzono formalnego testu
odsłuchowego.

## Ograniczenia

- Tylko polski. Jeden głos, bez klonowania.
- Ten sam tekst jest zawsze czytany tak samo.
- Styl lektorski z czytania książek; emocje i potoczna mowa są słabiej
  reprezentowane.
- Słychać, że głos jest syntetyczny (generator źródło-filtr).
- Liczba po przyimku jest odmieniana („z 5 osobami” → „z pięcioma osobami”), ale
  bez samotnej jedynki („z 1 osobą”) i liczb od tysiąca w górę.
- Obce nazwy i nietypowa pisownia bywają czytane źle; wymowa zależy od
  reguł `wymowa` i słownika wyjątków `src/wymowa/lexicon.py`.
- Bez strumieniowania.

## Głos i odpowiedzialne użycie

Głos należy do prawdziwej osoby, autora zbioru MC Speech, który udostępnił
swoje nagrania w domenie publicznej. Prosimy nie używać modelu do podszywania
się pod niego ani do wprowadzania w błąd, że mówi on słowa wygenerowane przez
model.

## Licencja

Wagi i kod: **Apache-2.0** (https://www.apache.org/licenses/LICENSE-2.0).
Można ich używać, zmieniać i rozpowszechniać w dowolnym celu, także
komercyjnym. Kto rozpowszechnia wagi lub ich zmienioną wersję, dołącza tekst
licencji i plik `NOTICE` oraz zaznacza swoje zmiany. Mowa wygenerowana
modelem nie wymaga podpisu.

Licencja wynika ze źródeł: MC Speech to domena publiczna (CC0), teksty Common
Voice też (CC0), szablony napisano dla projektu, a VoxCPM2 jest na licencji
Apache-2.0, która nie ogranicza użycia wygenerowanego dźwięku. Model nie
korzysta z wag V1 ani z nagrań OmniVoice (CC-BY-NC). Aligner, Whisper i
Parakeet służyły tylko do przygotowania danych i oceny.

## Podziękowania

Głos: MC Speech, Mateusz Czyżnikiewicz (CC0):

```
@masterthesis{mcspeech,
  title={Analiza porównawcza korpusów nagrań mowy dla celów syntezy mowy w języku polskim},
  author={Czyżnikiewicz, Mateusz},
  year={2022},
  school={Warsaw University of Technology},
  type={Master's thesis},
  doi={10.13140/RG.2.2.26293.24800}
}
```

Architektura, przepis treningu i kod bazowy: tiny-voca (Muhammed Nazeem,
Apache-2.0). Trudne przypadki: VoxCPM2 (OpenBMB, Apache-2.0). Teksty: Mozilla
Common Voice (CC0). Aligner: `facebook/wav2vec2-lv-60-espeak-cv-ft`
(Apache-2.0). Odsiew i ocena: Whisper (OpenAI, MIT), Parakeet-TDT-0.6B-v3
(NVIDIA, CC-BY-4.0), WavLM (Microsoft). Projekt opiera się na FastSpeech 2,
DDSP, neuronowym wokoderze homomorficznym oraz dyskryminatorach HiFi-GAN i
UnivNet, bez użycia ich kodu ani wag.
