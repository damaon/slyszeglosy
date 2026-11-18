# Zmiana głosu bez treningu

Ten przepis upodabnia głos gotowego modelu do innego głosu, na przykład
własnego albo lektora z nagrań. Robi to bez ani jednego kroku treningu. Mierzy,
czym głos docelowy różni się od głosu modelu, i wpisuje tę różnicę na stałe w
cztery warstwy liniowe. Całość zajmuje kilka minut na GPU. Wynikiem jest
zwykły checkpoint tego samego rozmiaru, który działa z `tiny_voca.infer`,
eksportem do ONNX i demem.

Kod: [`scripts/voicemorph/morph.py`](../scripts/voicemorph/morph.py).

## Czego się spodziewać

- Model mówi **wysokością, intonacją i tempem** głosu docelowego, a jego
  **barwa** przesuwa się w stronę celu.
- Sposób wymawiania, akcent, rytm zdań i „charakter” głosu zostają z modelu.
  Wynik brzmi jak głos modelu przestrojony na cel, a nie jak klon. Prawdziwy
  klon wymaga treningu na nagraniach docelowego głosu.
- W pomiarze na lektorze MC Speech podobieństwo głosu wzrosło z 0,79 do
  0,86–0,87. Sufit, czyli inne nagrania tego samego lektora, to 0,96. Wyniki
  są w tabeli niżej.
- Zmienione warstwy są **wspólne dla obu głosów** (`F` i `M`). Wynikowego
  checkpointu używaj tylko z głosem, który przerabiałeś (`--voice`). Drugi
  głos też się zmieni, ale nie w sensowną stronę.

## Dlaczego to działa

Dźwięk powstaje w stałym syntezatorze typu źródło–filtr:

- **źródło** to ciąg impulsów o wysokości F0, czyli struny głosowe;
- **filtr** to kształt widma, czyli gardło i usta, opisany 80 współczynnikami
  cepstrum (plus 16 dla kształtu impulsu krtaniowego i 32 dla szumu).

Wysokość (log F0) i cepstrum to wyjścia **pojedynczych warstw liniowych**
`y = W x + b`. Jeśli chcemy, żeby wyjście było `T y + t`, wystarczy podmienić
wagi: `W' = T W`, `b' = T b + t`. Model dalej ma ten sam rozmiar i działa tak
samo szybko.

Zmieniane warstwy:

| Warstwa | Zmiana | Skąd liczby |
|---|---|---|
| `prosody_head` (wiersz log F0) | `log F0' = k · log F0 + d` | średnia i rozrzut log F0 obu głosów |
| `prosody_projection` | kompensacja, żeby dalsza część modelu widziała starą wysokość | wynika z `k`, `d` |
| `duration.out` (bias) | tempo × `r` | łączny czas tych samych zdań |
| `vocoder.harmonic` | mapa `T`, `t` na cepstrum filtra | dopasowanie na parach ramek |
| `vocoder.noise` | tylko w wariancie `warp`: to samo przesunięcie formantów | `α` z wariantu `warp` |

Kompensacja w `prosody_projection` sprawia, że część modelu przewidująca
filtr dostaje dokładnie to samo co przed zmianą. Filtr nie „rozjeżdża się”
przez nową wysokość, a mapa barwy działa na tych samych danych, na których ją
policzono.

### Skąd się bierze mapa barwy

1. Model czyta **te same zdania**, które są w nagraniach docelowych, czyli
   powstają dane równoległe.
2. Oba zestawy przechodzą tę samą analizę ([pyworld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder)):
   F0 i obwiednia widma co ramkę modelu (256 próbek, ok. 10,7 ms).
3. DTW wyrównuje oba nagrania w czasie, więc porównujemy „to samo »a« u obu”.
   Zostają pary ramek dźwięcznych.
4. Cepstrum filtra modelu jest odczytywane wprost z warstwy `vocoder.harmonic`
   (hook). Stała różnica między analizą a filtrem modelu (`e`) jest
   odejmowana, żeby obie strony były w tej samej skali.
5. Mapa powstaje w jednym kroku. W wariancie `warp` to przeszukanie jednego
   parametru. W wariantach `affine` to jedno rozwiązanie równań najmniejszych
   kwadratów z lekką regularyzacją. Nie ma iteracji ani gradientów.

Współczynnik 0, czyli głośność, nie jest zmieniany.

### Warianty

Skrypt zapisuje wszystkie warianty, żeby można było je porównać.

| Wariant | Co zmienia |
|---|---|
| `pitch` | tylko wysokość, intonacja i tempo |
| `warp` | + przesunięcie formantów (jeden parametr α, długość toru głosowego) i średnia różnica widma |
| `diag` | + dopasowanie średniej i rozrzutu każdego współczynnika osobno |
| `affine20` | + pełna mapa liniowa na zgrubnej barwie (współczynniki 1–19) |
| `affine40` | + pełna mapa liniowa na współczynnikach 1–39 |

Domyślnie polecam **`affine20`**. Jest najbezpieczniejszy:
- `affine40` bywa minimalnie bardziej podobny do celu, ale mocniej wygładza
  widmo;
- na trudniejszym zestawie zdań `affine40` trzykrotnie podniósł CER (z 0,96%
  do 3,04%), a `affine20` nie (1,06%).

## Przepis

### 1. Środowisko

To samo co do treningu (`bash scripts/setup_env.sh`), plus analiza pyworld.
pyworld potrzebuje `pkg_resources`, a tego nowe setuptools już nie mają:

```bash
uv pip install --python ~/.venvs/sg/bin/python pyworld "setuptools<81"
```

Ocena zrozumiałości (opcjonalna) korzysta ze środowiska Parakeet:
`bash scripts/asrtest/setup_parakeet_gpu.sh`.

### 2. Nagrania docelowego głosu

Potrzebny jest folder z nagraniami i plikiem `metadata.csv`. Każda linia
wygląda tak: `plik.wav|dokładny tekst nagrania`.

```
moje-nagrania/
  metadata.csv
  001.wav
  002.wav
  ...
```

```
001.wav|Dzień dobry, nazywam się Jan Kowalski.
002.wav|Hrabia napełnił dwie szklanki.
```

- **Ile:** minimum 60 nagrań, a dobrze ok. 300–450 zdań, czyli 20–30 minut
  czytania. Skrypt dzieli je losowo (ziarno 0):
  - 1/6, najwyżej 60, na „odcisk” głosu do oceny;
  - 1/5, najwyżej 100, na zdania testowe;
  - resztę na dopasowanie.
- **Tekst musi się zgadzać z nagraniem**, bo model czyta te same zdania, a
  pary ramek powstają z tych samych słów.
- **Jakość:** jeden mówca, cicho, bez pogłosu i muzyki. Czytane spokojnie,
  pojedyncze zdania po kilka do kilkunastu sekund.
- Dowolna częstotliwość próbkowania i stereo są w porządku. Skrypt zamienia
  wszystko na 24 kHz mono.
- Zamiast folderu można podać plik parquet z Hugging Face z kolumnami
  `audio` i `transcript` (`--parquet`), tak jak w MC Speech.

### 3. Uruchomienie

Z katalogu repozytorium, w środowisku `sg`:

```bash
export PYTHONPATH=src:scripts
M="python scripts/voicemorph/morph.py --checkpoint weights/slyszeglosy.pt --work data/voicemorph --voice M"

$M data --folder moje-nagrania   # nagrania -> data/voicemorph/target, podział na zbiory
$M fit                           # dopasowanie, zapis data/voicemorph/inference-*.pt
$M render                        # zdania testowe każdym wariantem
$M score                         # podobieństwo głosu, MCD, F0, tempo (+ CER, jeśli jest)
$M samples                       # pliki do odsłuchu: data/voicemorph/samples/
```

- `--voice`: wybierz głos modelu bliższy celowi, zwykle tej samej płci.
- `--checkpoint`: plik `inference.pt` albo `slyszeglosy.pt` z Hugging Face.
  Punkty kontrolne z treningu (`step-*.pt`) się nie nadają.

Opcjonalnie zrozumiałość, czyli CER. Uruchom przed `score`, żeby trafiła do
tabeli:

```bash
~/.venvs/parakeet/bin/python scripts/asrtest/run_asr.py --engine parakeet --sets data/voicemorph/eval/sets.jsonl
python scripts/voicemorph/cer.py --work data/voicemorph
```

`fit` wypisuje zmierzone wielkości, na przykład:

```
pairs 54641; F0 source 148 Hz sd 0.190, target 113 Hz sd 0.143 -> k 0.756 d -0.341; tempo x0.972
err_none 5.0810 err_warp 2.3393 err_affine40 1.0355 err_affine20 1.2631 err_diag 2.0955 (warp alpha -0.01)
```

- `k` i `d`: skala i przesunięcie log F0;
- `tempo`: mnożnik czasu (poniżej 1 oznacza szybciej);
- `err_*`: błąd dopasowania barwy na parach ramek (mniej znaczy lepiej, ale to
  dopasowanie, a nie ocena);
- `alpha`: przesunięcie formantów (0 oznacza, że formanty są w tym samym
  miejscu).

### 4. Ocena i wybór wariantu

`score` liczy na zdaniach testowych, których nie było w dopasowaniu:

- **SIM**: podobieństwo do „odcisku” głosu docelowego, czyli cosinus
  embeddingów [WavLM-SV](https://huggingface.co/microsoft/wavlm-base-plus-sv).
  Skala jest ściśnięta: nawet głos drugiej płci dostaje ok. 0,75. Liczy się
  pozycja między `base_*` a `target`.
- **MCD**: różnica widma względem prawdziwego nagrania tego samego zdania w
  dB, po wyrównaniu DTW. Mniej znaczy lepiej.
- **F0_Hz**, **F0_sd**: średnia wysokość i rozrzut intonacji. Powinny być
  bliskie `target`.
- **seconds**: łączny czas mowy (tempo).
- **CER_%**: zrozumiałość. Nie powinna być wyraźnie gorsza od `base_*`.

Zawsze trzeba posłuchać `samples/`. Liczby nie wyłapią metalicznego brzmienia
ani przytłumienia.

### 5. Użycie wyniku

```bash
python -m tiny_voca.infer --checkpoint data/voicemorph/inference-affine20.pt --voice M \
  --text "Dzień dobry! Spotkanie zaczyna się o 15:30." --output test.wav

python -m tiny_voca.export --checkpoint data/voicemorph/inference-affine20.pt --output model.onnx
bash web/build_assets.sh model.onnx     # demo z nowym głosem
```

## Wyniki referencyjne

Głos `M` przerobiony na lektora MC Speech (CC0), pierwszy plik parquet zbioru:

- 460 nagrań: 308 do dopasowania, 92 testowe i 60 na „odcisk” głosu;
- 54,6 tys. par ramek.

| System | SIM | MCD dB | F0 | CER |
|---|---|---|---|---|
| `target` (prawdziwe nagrania, sufit) | 0,964 | 0 | 113 Hz | 0,41% |
| `base_M` (bez zmian) | 0,787 | 5,75 | 149 Hz | 0,20% |
| `base_F` (dla skali) | 0,747 | 4,73 | 176 Hz | 0,29% |
| `pitch` | 0,846 | 5,77 | 114 Hz | 0,24% |
| `warp` | 0,832 | 4,73 | 115 Hz | 0,23% |
| `diag` | 0,830 | 4,52 | 114 Hz | 0,22% |
| `affine20` | 0,861 | 4,06 | 116 Hz | 0,23% |
| `affine40` | 0,868 | 3,98 | 116 Hz | 0,23% |

Wnioski:

- Większość zysku daje wysokość głosu (`pitch`).
- Mapa barwy dokłada wyraźnie mniej, ale zmniejsza różnicę widma (MCD z 5,8
  do 4,0 dB).
- Formanty głosu `M` i lektora leżą prawie w tym samym miejscu (α = −0,01),
  więc samo przesunięcie formantów niczego tu nie daje. Przy głosie innej
  płci lub dziecka α będzie większe.

## Ograniczenia

- **To nie klon.** Zmieniamy tylko to, co da się opisać liniowo na samym
  końcu modelu: wysokość, tempo i średnią barwę. Wszystko, co siedzi w
  transformerach (wymowa, akcent, rytm, styl), zostaje z modelu.
- **Ten sam „kierunek” dla wszystkich głosek.** Mapa barwy jest jedna dla
  całego widma. Różnice, które zależą od głoski, uśredniają się.
- **Duże różnice gorzej się przenoszą**, na przykład głos męski na dziecięcy
  albo szept. Wtedy warto zacząć od głosu modelu bliższego celowi.
- **Licencja wyniku = licencja wag wejściowych.** Wagi V1 są CC-BY-NC 4.0,
  więc przerobiony model też. Głos prawdziwej osoby upodabniaj tylko za jej
  zgodą.
