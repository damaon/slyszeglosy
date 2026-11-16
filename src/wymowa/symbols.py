"""Token inventory of the Polish front end.

One token is one phone (IPA). Lexical stress is a parallel 0/1/2 feature, not
a token. Punctuation, word boundaries and utterance edges are explicit tokens
that may last zero frames.

Append-only once a checkpoint exists: changing the order or removing a symbol
invalidates every annotation and checkpoint made with it.
"""

PAD, BOS, EOS, SP, UNK = "<pad>", "<bos>", "<eos>", "<sp>", "<unk>"
SPECIALS = (PAD, BOS, EOS, SP, UNK)
PUNCTUATION = (",", ".", "?", "!", ";", ":", '"', "'", "(", ")", "—", "…")

# Oral vowels, then the two nasal vowels (realised as such before fricatives,
# vowels and word-finally for ą; before stops they become vowel + nasal).
VOWELS = ("a", "ɛ", "i", "ɨ", "ɔ", "u", "ɛ̃", "ɔ̃")
CONSONANTS = (
    "p", "b", "t", "d", "k", "ɡ", "c", "ɟ",          # stops; c ɟ = ki, gi
    "f", "v", "s", "z", "ʂ", "ʐ", "ɕ", "ʑ", "x",     # fricatives; ʂ ʐ = sz, ż/rz; ɕ ʑ = ś, ź
    "ts", "dz", "tʂ", "dʐ", "tɕ", "dʑ",             # affricates: c, dz, cz, dż, ć, dź
    "m", "n", "ɲ", "ŋ",                             # nasals; ŋ before k/g
    "l", "r", "w", "j",                             # liquids and glides; w = ł
)
PHONES = CONSONANTS + VOWELS
SYMBOLS = SPECIALS + PUNCTUATION + PHONES

VOICELESS = frozenset(("p", "t", "k", "c", "f", "s", "ʂ", "ɕ", "x", "ts", "tʂ", "tɕ"))
VOICED_PARTNER = {"p": "b", "t": "d", "k": "ɡ", "c": "ɟ", "f": "v", "s": "z", "ʂ": "ʐ", "ɕ": "ʑ",
                  "ts": "dz", "tʂ": "dʐ", "tɕ": "dʑ"}
VOICELESS_PARTNER = {voiced: voiceless for voiceless, voiced in VOICED_PARTNER.items()}
OBSTRUENTS = frozenset(VOICED_PARTNER) | frozenset(VOICELESS_PARTNER) | {"x"}
SONORANTS = frozenset(("m", "n", "ɲ", "ŋ", "l", "r", "w", "j"))
VOWEL_SET = frozenset(VOWELS)
PHONE_SET = frozenset(PHONES)

if len(set(SYMBOLS)) != len(SYMBOLS):
    raise AssertionError("duplicate symbol in the inventory")
