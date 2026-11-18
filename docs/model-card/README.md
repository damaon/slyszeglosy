---
license: cc-by-nc-4.0
language:
- pl
pipeline_tag: text-to-speech
tags:
- text-to-speech
- onnx
- polish
---
# słyszę głosy

Mały polski model syntezy mowy (TTS) z dwoma głosami: `F` (kobiecy) i `M`
(męski). Cały model, razem z generatorem fali, ma 4,5 mln parametrów i został
wytrenowany od zera na 18 godzinach syntetycznej mowy. Na czterech wątkach CPU
działa ok. 150× szybciej niż w czasie rzeczywistym, w przeglądarce
(ONNX Runtime Web) ok. 25×.

<audio controls src="https://huggingface.co/Damaon/slyszeglosy/resolve/main/samples/lokomotywa-F.wav"></audio>

- Kod: https://github.com/damaon/slyszeglosy
- Demo w przeglądarce: https://slyszeglosy-demo.damaon.workers.dev

Inspiracją i bazą jest [tiny-voca](https://github.com/n4ze3m/tiny-voca)
(Muhammed Nazeem, Apache-2.0): ta sama architektura i ten sam przepis treningu,
z nowym front-endem fonetycznym `wymowa`, nowym korpusem i nowymi głosami.

| | |
|---|---|
| Parametry | 4 533 451 |
| Głosy | `F`, `M` (polski, syntetyczne) |
| Wyjście | 24 kHz mono |
| Wejście | Tekst zamieniany na głoski przez `wymowa` (reguły, bez słownika zewnętrznego) |
| Wagi wstępnie wytrenowane w modelu | Brak |
| Licencja wag | CC-BY-NC 4.0 (kod: Apache-2.0) |

## Pliki

| Plik | Rozmiar | Zawartość |
|---|---:|---|
| `slyszeglosy.pt` | 18,2 MB | Wagi PyTorch, float32 |
| `onnx/model.onnx` | 21,4 MB | Eksport ONNX, float32, opset 17 |
| `onnx/model.json` | | Metadane eksportu: słownik głosek, głosy, wejścia i wyjścia |
| `config.json` | | Ustawienia modelu, słownik głosek, głosy |
| `samples/` | | Próbki z tej strony |

SHA256:

- `slyszeglosy.pt`: `edfba942687504b0b5933a8718773f3956780d1690a71af556be84be9abb4841`
- `onnx/model.onnx`: `6d2a81bcc47589f9235452011979f641395749015a3903aaadff4558784ab087`

## Próbki

| Tekst | `F` | `M` |
|---|---|---|
| Dzień dobry! Spotkanie zaczyna się o 15:30 w sali nr 4. | <audio controls src="https://huggingface.co/Damaon/slyszeglosy/resolve/main/samples/F-1.wav"></audio> | <audio controls src="https://huggingface.co/Damaon/slyszeglosy/resolve/main/samples/M-1.wav"></audio> |
| W Szczebrzeszynie chrząszcz brzmi w trzcinie. | <audio controls src="https://huggingface.co/Damaon/slyszeglosy/resolve/main/samples/F-2.wav"></audio> | <audio controls src="https://huggingface.co/Damaon/slyszeglosy/resolve/main/samples/M-2.wav"></audio> |
| Czy możesz kupić 2 kg jabłek i 1,5 l mleka? | <audio controls src="https://huggingface.co/Damaon/slyszeglosy/resolve/main/samples/F-3.wav"></audio> | <audio controls src="https://huggingface.co/Damaon/slyszeglosy/resolve/main/samples/M-3.wav"></audio> |
| Tuwim, „Lokomotywa” (początek) | <audio controls src="https://huggingface.co/Damaon/slyszeglosy/resolve/main/samples/lokomotywa-F.wav"></audio> | <audio controls src="https://huggingface.co/Damaon/slyszeglosy/resolve/main/samples/lokomotywa-M.wav"></audio> |
| Mickiewicz, „Pan Tadeusz”, Inwokacja | <audio controls src="https://huggingface.co/Damaon/slyszeglosy/resolve/main/samples/inwokacja-F.wav"></audio> | <audio controls src="https://huggingface.co/Damaon/slyszeglosy/resolve/main/samples/inwokacja-M.wav"></audio> |

## Użycie

### PyTorch

```bash
pip install git+https://github.com/damaon/slyszeglosy
hf download Damaon/slyszeglosy --local-dir slyszeglosy

python -m tiny_voca.infer --checkpoint slyszeglosy/slyszeglosy.pt --voice F \
  --text "Dzień dobry! Spotkanie zaczyna się o 15:30." --output dzien-dobry.wav
```

| Opcja | Domyślnie | Znaczenie |
|---|---|---|
| `--text-file` | | Tekst z pliku zamiast `--text` |
| `--voice` | `F` | `F` albo `M` |
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
session = ort.InferenceSession("slyszeglosy/onnx/model.onnx")
audio, durations = session.run(None, {
    "tokens": np.array([phones.ids], dtype=np.int64),
    "stress": np.array([phones.stress], dtype=np.int64),
    "voice": np.array([0], dtype=np.int64),      # 0 = F, 1 = M
    "speed": one(1.0), "pitch_shift": one(0.0), "pitch_range": one(1.0),
})
audio = audio[0] * min(1.0, 0.99 / np.abs(audio[0]).max())
sf.write("dzien-dobry.wav", audio, 24000)
```

Jedno wywołanie czyta jedną wypowiedź (do 512 głosek i 32 s). Wypowiedź
powinna kończyć się znakiem interpunkcyjnym. Ten sam plik działa w
onnxruntime-web; demo jest w katalogu `web/` repozytorium.

## Model

Architektura tiny-voca bez zmian: enkoder transformerowy (4 bloki) czyta
głoski z akcentem, predyktor długości daje każdej głosce liczbę ramek, dwa
transformery ramkowe (2 i 5 bloków) przewidują wysokość, dźwięczność i
energię, a potem parametry syntezatora źródło-filtr (ciąg impulsów i szum
kształtowane filtrami cepstralnymi), bez neuronowego wokodera. 192 kanały,
4 głowice, RoPE, ramka 256 próbek (10,7 ms).

| Komponent | Parametry |
|---|---:|
| Osadzenia (głoska, akcent, głos, pozycja) | 12 864 |
| Enkoder tekstu | 1 588 864 |
| Predyktor długości i pauz | 123 906 |
| Transformer ramkowy A, głowica prozodii | 795 971 |
| Transformer ramkowy B | 1 985 984 |
| Syntezator źródło-filtr | 25 862 |
| Razem | 4 533 451 |

Słownik ma 56 symboli: 39 polskich głosek (8 samogłosek z nosowymi
`ɛ̃ ɔ̃`, 31 spółgłosek), znaki specjalne i interpunkcję.

## Trening

| | |
|---|---|
| Dane | 18 268 nagrań, 18,0 h, po połowie na głos (walidacja 576, test 566) |
| Kroki | 104 000 |
| Sprzęt | Jedna karta GPU (24 GB), 8,9 h, z termostatem GPU (`--max-temp 76`) |
| Straty | Wielorozdzielcze straty widmowe; od kroku 8000 strata adwersarialna (dyskryminatory wielookresowe i wielorozdzielcze) |

Nagrania treningowe są syntetyczne:

1. **Teksty**: 10 020 zdań: Common Voice PL (sentence collector i Europarl,
   CC0) oraz zdania i szablony napisane dla projektu (liczby, daty, godziny,
   kwoty, skróty).
2. **Głosy** zaprojektowano z samego opisu tekstowego w OmniVoice
   (k2-fsa), bez nagrania prawdziwej osoby.
3. **Nagrania**: każde zdanie oboma głosami, OmniVoice 0.6B z klonowaniem
   zaprojektowanych głosów.
4. **Odsiew**: Whisper large-v3-turbo; nagrania z CER powyżej 5% generowano
   drugi raz, potem odrzucano (244 z 20 040).
5. **Adnotacja**: długości głosek z wymuszonego dopasowania
   `facebook/wav2vec2-lv-60-espeak-cv-ft`, z mapą etykiet i przesunięciem
   granic skalibrowanymi na polskich nagraniach. Kontrola jakości dopasowania
   odrzuciła dalsze 386 nagrań. Aligner służy tylko do adnotacji i nie jest
   częścią modelu.

## Ocena

CER i WER mowy z modelu w transkrypcji Whisper large-v3-turbo (język `pl`).
Oryginalne nagrania nauczyciela tych samych zdań oceniono tak samo.

| Zdania | Nagrania | CER model | CER nauczyciel | WER model | WER nauczyciel |
|---|---:|---:|---:|---:|---:|
| Walidacja | 50 | 0,37% | 0,28% | 2,7% | 2,5% |
| Test | 50 | 0,48% | 0,52% | 3,1% | 3,6% |
| Nowe zdania spoza korpusu (z liczbami i skrótami) | 20 | 0,0% | | 0,0% | |

Model jest tak samo zrozumiały jak jego nauczyciel: różnice mieszczą się w
szumie pomiaru. Długość wypowiedzi wynosi 98–99% długości nagrań nauczyciela.

To małe zbiory: wynik jest kontrolą poprawności, nie benchmarkiem. Nie
przeprowadzono formalnego testu odsłuchowego.

## Ograniczenia

- Tylko polski. Dwa stałe głosy, bez klonowania.
- Ten sam tekst jest zawsze czytany tak samo.
- Głos brzmi czysto, ale słychać, że jest syntetyczny (generator źródło-filtr).
- Liczba po przyimku jest odmieniana („z 5 osobami” → „z pięcioma osobami”), ale
  bez samotnej jedynki („z 1 osobą”) i liczb od tysiąca w górę.
- Obce nazwy i nietypowa pisownia bywają czytane źle; wymowa zależy od
  reguł `wymowa` i słownika wyjątków `src/wymowa/lexicon.py`.
- Bez strumieniowania.
- Dane są syntetyczne, więc model nie brzmi naturalniej niż jego nauczyciel i
  dziedziczy jego wymowę.

## Licencja

Wagi: **CC-BY-NC 4.0**, tylko użytek niekomercyjny. Nagrania treningowe
wygenerował OmniVoice, którego wagi są udostępnione na licencji CC-BY-NC.
Kod (tiny-voca i `wymowa`): Apache-2.0.

## Podziękowania

Architektura, przepis treningu i kod bazowy: tiny-voca (Muhammed Nazeem,
Apache-2.0). Nauczyciel: OmniVoice (k2-fsa). Teksty: Mozilla Common Voice
(CC0). Aligner: `facebook/wav2vec2-lv-60-espeak-cv-ft` (Apache-2.0). Ocena:
Whisper (MIT). Projekt opiera się na FastSpeech 2, DDSP, neuronowym wokoderze
homomorficznym oraz dyskryminatorach HiFi-GAN i UnivNet, bez użycia ich kodu
ani wag.
