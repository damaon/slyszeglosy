import unittest
from unittest.mock import patch

from tiny_voca import text
from tiny_voca.align import ALIGNER_LABEL_FALLBACK
from tiny_voca.config import Config
from tiny_voca.text import (BOS, EOS, PHONE_SET, PHONES, SP, SYMBOLS, SYMBOL_TO_ID, UNK, NothingToPronounce,
                            frontend_fingerprint, phonemize, tokenize)


class FrontEndTests(unittest.TestCase):
    def test_inventory_is_consistent_with_the_model_vocabulary(self):
        self.assertEqual(len(SYMBOLS), len(set(SYMBOLS)))
        self.assertEqual(Config().vocab_size, len(SYMBOLS))
        self.assertEqual(SYMBOL_TO_ID["<pad>"], 0)
        self.assertEqual(len(PHONES), 39)
        with self.assertRaises(ValueError):
            Config(vocab_size=76)     # the English inventory is gone

    def test_pieces_become_tokens_with_stress_features(self):
        words = [[("tʂ", 0), ("ɛ", 1), ("ɕ", 0), ("tɕ", 0)], [("ɔ", 0), ("l", 0), ("a", 1)]]
        result = tokenize([(True, '"'), (False, words), (True, ","), (True, ","), (False, [[("ɲ", 0), ("ɛ", 0)]]), (True, "?")])
        self.assertEqual(result.symbols, (BOS, '"', "tʂ", "ɛ", "ɕ", "tɕ", SP, "ɔ", "l", "a", ",", "ɲ", "ɛ", "?", EOS))
        self.assertEqual(result.stress, (0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0))
        self.assertEqual(result.ids, tuple(SYMBOL_TO_ID[s] for s in result.symbols))

    def test_out_of_inventory_phones_are_reported(self):
        result = tokenize([(False, [[("t", 0), ("ʘ", 0), ("a", 1)]])])
        self.assertEqual(result.symbols, (BOS, "t", UNK, "a", EOS))
        self.assertEqual(result.unknown, ("ʘ",))
        with self.assertRaises(NothingToPronounce):
            tokenize([(True, ".")])

    def test_fingerprint_tracks_inventory_and_version(self):
        baseline = frontend_fingerprint()
        with patch.object(text, "FRONTEND_VERSION", 999):
            self.assertNotEqual(frontend_fingerprint(), baseline)
        with patch.object(text, "SYMBOLS", SYMBOLS + ("new",)):
            self.assertNotEqual(frontend_fingerprint(), baseline)

    def test_phonemize_real_polish(self):
        result = phonemize("Zażółć gęślą jaźń, proszę!")
        self.assertEqual(result.unknown, ())
        self.assertEqual([s for s in result.symbols if s in ",!"], [",", "!"])
        self.assertEqual(result.symbols[:9], (BOS, "z", "a", "ʐ", "u", "w", "dʑ", SP, "ɡ"))   # ć voiced before g
        self.assertNotEqual(phonemize("Tak.").ids, phonemize("Tak?").ids)
        self.assertEqual(phonemize("Tak, jest.").ids, phonemize("  Tak,   jest. ").ids)
        self.assertTrue(all(s in PHONE_SET or s in SYMBOLS for s in result.symbols))
        self.assertEqual(sum(level > 0 and s not in PHONE_SET for s, level in zip(result.symbols, result.stress)), 0)
        with self.assertRaises(NothingToPronounce):
            phonemize("🙂 !!!")

    def test_numbers_and_long_sentences_stay_in_the_inventory(self):
        sample = ("W 2026 r. o 15:30 prof. Nowak zapłacił 12,50 zł za 3 kg jabłek; NFZ, PKP i ZUS — to skróty. "
                  "Chrząszcz brzmi w trzcinie, a źdźbło dżdżystej trawy drży.")
        result = phonemize(sample)
        self.assertEqual(result.unknown, ())
        self.assertGreater(len(set(result.symbols) & PHONE_SET), 30)

    def test_every_phone_has_an_aligner_label_mapping_or_is_its_own_label(self):
        # The mapped labels themselves are checked against vocab.json when the recogniser loads.
        self.assertTrue(set(ALIGNER_LABEL_FALLBACK) - {UNK} <= PHONE_SET)


if __name__ == "__main__":
    unittest.main()
