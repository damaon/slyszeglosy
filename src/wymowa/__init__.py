"""wymowa: Polish text -> phones with lexical stress, for slyszeglosy.

Rules and a small exception list, no neural model and no external program,
so it can ship under Apache-2.0 with the model.

    >>> import wymowa
    >>> wymowa.transcribe("Chrząszcz brzmi w trzcinie.")
    'xʂˈɔ̃ʐdʐ bʐmˈi ftʂtɕˈiɲɛ .'
"""

import hashlib
from pathlib import Path
import re

from .g2p import phrase_phones
from .normalize import normalize
from .symbols import PHONE_SET, PHONES, PUNCTUATION, SYMBOLS, VOWELS

VERSION = 1
__all__ = ["VERSION", "SYMBOLS", "PHONES", "VOWELS", "PUNCTUATION", "normalize", "pieces", "transcribe", "identity"]

_PUNCTUATION = "".join(re.escape(p) for p in PUNCTUATION)
_TOKEN = re.compile(rf"[{_PUNCTUATION}]|[^\s{_PUNCTUATION}]+")
_SOURCES = ("__init__.py", "symbols.py", "g2p.py", "lexicon.py", "normalize.py", "numbers.py")


def pieces(text):
    """``[(is_punctuation, value)]``: a punctuation mark, or the words of one
    phrase as lists of ``(phone, stress)`` with stress 0, 1 (primary) or 2."""
    out, words = [], []

    def flush():
        if words:
            phrase = [w for w in phrase_phones(words) if w]
            if phrase:
                out.append((False, phrase))
            words.clear()

    for token in _TOKEN.findall(normalize(text)):
        if token in PUNCTUATION:
            flush()
            out.append((True, token))
        else:
            words.append(token)
    flush()
    return out


def transcribe(text):
    """Readable IPA: words separated by spaces, ˈ/ˌ before stressed vowels."""
    parts = []
    for punctuation, value in pieces(text):
        if punctuation:
            parts.append(value)
            continue
        for word in value:
            parts.append("".join(("ˈ" if level == 1 else "ˌ" if level == 2 else "") + phone for phone, level in word))
    return " ".join(parts)


def identity():
    """Version and a fingerprint of the exact rules: annotations made with one
    build are refused by another."""
    digest = hashlib.sha256()
    for name in _SOURCES:
        source = (Path(__file__).parent / name).read_text(encoding="utf-8")
        digest.update("\n".join(source.splitlines()).encode())     # same on CRLF checkouts
    return {"name": "wymowa", "version": VERSION, "fingerprint": digest.hexdigest()}


assert all(p in PHONE_SET for p in PHONES)
