# Głosy referencyjne

Dwa syntetyczne głosy zaprojektowane przez OmniVoice z samych atrybutów
(bez nagrania prawdziwej osoby), wybrane przez porównanie nauczycieli (`scripts/bakeoff/`).
Nauczyciel klonuje je przy generacji danych treningowych:

    python scripts/bakeoff/generate.py --engine omnivoice --voices voices/voices.json

Licencja: CC-BY-NC 4.0 (wytworzone modelem OmniVoice, którego wagi są CC-BY-NC).
