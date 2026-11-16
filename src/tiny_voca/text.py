"""Polish front end: text -> phoneme tokens with a stress feature.

The same code runs during annotation, training-set preparation and inference,
so tokens can never drift between them. The phonemes come from ``wymowa``
(rules and a small exception list, Apache-2.0, in ``src/wymowa``).

* A token is one phone, e.g. ``t``, ``ʂ``, ``tɕ`` or ``ɔ̃`` (see
  ``wymowa.symbols``). The forced aligner maps each to one of its labels
  (``tiny_voca.align.ALIGNER_LABEL_FALLBACK``).
* Lexical stress is a parallel feature (0 none, 1 primary, 2 secondary), not a
  token, so it never has to be given audio frames.
* Punctuation, word boundaries and the utterance edges are explicit tokens that
  may last zero frames. They carry pauses when the recording has them.

eSpeak NG (GPL) is not used here; ``scripts/compare_espeak_pl.py`` compares
the two for reference only.
"""

from dataclasses import dataclass
import hashlib
import json
import os

import wymowa
from wymowa.symbols import (BOS, CONSONANTS, EOS, PAD, PHONE_SET, PHONES, PUNCTUATION, SP, SPECIALS, SYMBOLS, UNK,
                            VOICELESS, VOWEL_SET, VOWELS)

FRONTEND_VERSION = wymowa.VERSION
FRONTENDS = ("wymowa",)
FRONTEND_VARIABLE = "TINY_VOCA_FRONTEND"
_frontend = os.environ.get(FRONTEND_VARIABLE, FRONTENDS[0])
if _frontend not in FRONTENDS:
    raise ValueError(f"{FRONTEND_VARIABLE} must be one of {FRONTENDS}, not {_frontend!r}")

SYMBOL_TO_ID = {symbol: index for index, symbol in enumerate(SYMBOLS)}
PHONE_IDS = frozenset(SYMBOL_TO_ID[p] for p in PHONES) | {SYMBOL_TO_ID[UNK]}

__all__ = ["BOS", "EOS", "PAD", "SP", "UNK", "SPECIALS", "PUNCTUATION", "CONSONANTS", "VOWELS", "PHONES", "SYMBOLS",
           "SYMBOL_TO_ID", "PHONE_SET", "PHONE_IDS", "VOICELESS", "VOWEL_SET", "Phonemized", "NothingToPronounce",
           "phonemize", "tokenize", "frontend", "set_frontend", "frontend_identity", "frontend_runtime",
           "frontend_fingerprint", "is_phone"]


class NothingToPronounce(ValueError):
    """The text has no word, number or letter to say: only marks, emoji and the like."""


@dataclass(frozen=True)
class Phonemized:
    symbols: tuple
    ids: tuple
    stress: tuple
    unknown: tuple  # phones outside the inventory (should never happen with wymowa)

    def __len__(self):
        return len(self.ids)


def frontend():
    return _frontend


def set_frontend(name):
    """Kept for the command-line interfaces; there is one front end."""
    global _frontend
    if name not in FRONTENDS:
        raise ValueError(f"front end must be one of {FRONTENDS}, not {name!r}")
    _frontend = name


def frontend_identity():
    return {"frontend_version": FRONTEND_VERSION, "phonemizer": "wymowa", "language": "pl", "symbols": list(SYMBOLS)}


def frontend_runtime():
    """What an annotation run records about the exact rules behind its tokens."""
    return {"phonemizer": wymowa.identity()["fingerprint"]}


def frontend_fingerprint():
    return hashlib.sha256(json.dumps(frontend_identity(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def tokenize(pieces):
    """Token sequence from ``(is_punctuation, value)`` pieces, where a text
    piece's value is a list of words, each a list of ``(phone, stress)``.
    Pure function (testable without the phonemizer)."""
    symbols, stress, unknown = [BOS], [0], []
    for punctuation, value in pieces:
        if punctuation:
            if symbols[-1] != value:  # collapse repeats such as "!!"
                symbols.append(value)
                stress.append(0)
            continue
        for word in value:
            if symbols[-1] in PHONE_SET or symbols[-1] == UNK:
                symbols.append(SP)
                stress.append(0)
            for phone, level in word:
                if phone not in PHONE_SET:
                    unknown.append(phone)
                    phone = UNK
                symbols.append(phone)
                stress.append(level)
    symbols.append(EOS)
    stress.append(0)
    if not any(s in PHONE_SET or s == UNK for s in symbols):
        raise NothingToPronounce("the text has nothing to pronounce")
    return Phonemized(tuple(symbols), tuple(SYMBOL_TO_ID[s] for s in symbols), tuple(stress), tuple(unknown))


def phonemize(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must not be empty")
    return tokenize(wymowa.pieces(text))


def is_phone(token_ids):
    """Boolean list/tensor helper: tokens that must be realised as audio."""
    return [int(i) in PHONE_IDS for i in token_ids]
