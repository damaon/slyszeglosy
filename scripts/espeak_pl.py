"""eSpeak NG (voice ``pl``) through its shared library, for comparisons only.

eSpeak NG is GPL-3.0: it is never imported by the model, the front end or the
training code, only by scripts that measure how ``wymowa`` differs from it.
The library comes from the ``espeakng-loader`` wheel (``scripts/setup_env.sh``).

    python scripts/espeak_pl.py "Chrząszcz brzmi w trzcinie."
"""

import ctypes
import sys

_engine = None


def _load():
    global _engine
    if _engine is None:
        import espeakng_loader
        library = ctypes.CDLL(espeakng_loader.get_library_path())
        data = espeakng_loader.get_data_path().encode()
        if library.espeak_Initialize(0x02, 0, ctypes.c_char_p(data), 0) <= 0:   # AUDIO_OUTPUT_SYNCHRONOUS
            raise RuntimeError("espeak_Initialize failed")
        if library.espeak_SetVoiceByName(b"pl") != 0:
            raise RuntimeError("eSpeak has no 'pl' voice")
        library.espeak_TextToPhonemes.restype = ctypes.c_char_p
        library.espeak_TextToPhonemes.argtypes = (ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int)
        _engine = library
    return _engine


def ipa(text, separator="_"):
    """IPA of ``text`` clause by clause, phones joined by ``separator``,
    words by spaces. Stress marks are kept as eSpeak writes them."""
    library = _load()
    buffer = ctypes.create_string_buffer(text.encode("utf-8") + b"\0")
    pointer = ctypes.c_void_p(ctypes.addressof(buffer))
    mode = 0x02 | (ord(separator) << 8)          # espeakPHONEMES_IPA, separator in bits 8-23
    clauses = []
    while pointer.value:
        out = library.espeak_TextToPhonemes(ctypes.byref(pointer), 1, mode)   # espeakCHARS_UTF8
        if out:
            clauses.append(out.decode("utf-8"))
    return " ".join(clauses).strip()


if __name__ == "__main__":
    for line in sys.argv[1:] or sys.stdin:
        print(ipa(line.strip()))
