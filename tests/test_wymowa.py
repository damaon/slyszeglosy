"""wymowa: hand-checked transcriptions (the gold standard), stress and text
normalisation. Pure Python: no torch, no eSpeak.

Transcriptions follow the Warsaw norm of careful standard Polish: a
word-final obstruent is voiceless before a vowel or sonorant, and spelling
is kept where everyday speech simplifies (jabłko keeps its ł).
"""

import unittest

import wymowa
from wymowa.g2p import phrase_phones
from wymowa.numbers import cardinal, ordinal, plural
from wymowa.symbols import PHONE_SET, SYMBOLS


def phones(text):
    """Space-separated phones of each word, words joined by ' | '."""
    return " | ".join(" ".join(p for p, _ in word) for word in phrase_phones(text.split()))


def stressed(word):
    """(index of the primary-stressed vowel, index of a secondary-stressed vowel or None)."""
    [segments] = phrase_phones([word])
    vowels = [level for phone, level in segments if phone in wymowa.VOWELS]
    return vowels.index(1) if 1 in vowels else None, vowels.index(2) if 2 in vowels else None


GOLD = {
    # spelling: digraphs and single letters
    "chrząszcz": "x ʂ ɔ̃ ʂ tʂ", "szczęście": "ʂ tʂ ɛ̃ ɕ tɕ ɛ", "dżem": "dʐ ɛ m", "dźwięk": "dʑ v j ɛ ŋ k",
    "dzwon": "dz v ɔ n", "czapka": "tʂ a p k a", "rzeka": "ʐ ɛ k a", "żaba": "ʐ a b a", "chata": "x a t a",
    "herbata": "x ɛ r b a t a", "łódź": "w u tɕ", "góra": "ɡ u r a", "ryba": "r ɨ b a", "cebula": "ts ɛ b u l a",
    "dziecko": "dʑ ɛ ts k ɔ", "dzień": "dʑ ɛ ɲ", "koń": "k ɔ ɲ", "ćma": "tɕ m a", "źle": "ʑ l ɛ",
    "środa": "ɕ r ɔ d a", "taxi": "t a k s i", "rzym": "ʐ ɨ m", "jeż": "j ɛ ʂ", "chodź": "x ɔ tɕ", "jedź": "j ɛ tɕ",
    "szczotka": "ʂ tʂ ɔ t k a", "pszczoła": "p ʂ tʂ ɔ w a", "grzegorz": "ɡ ʐ ɛ ɡ ɔ ʂ", "źdźbło": "ʑ dʑ b w ɔ",
    "dżdżysty": "dʐ dʐ ɨ s t ɨ",
    # i softens the consonant before it, or is a glide j before a vowel
    "nie": "ɲ ɛ", "niebo": "ɲ ɛ b ɔ", "siano": "ɕ a n ɔ", "ziarno": "ʑ a r n ɔ", "ciocia": "tɕ ɔ tɕ a",
    "dzisiaj": "dʑ i ɕ a j", "zima": "ʑ i m a", "cisza": "tɕ i ʂ a", "nic": "ɲ i ts", "siła": "ɕ i w a",
    "pies": "p j ɛ s", "miasto": "m j a s t ɔ", "biały": "b j a w ɨ", "wiara": "v j a r a", "fiołek": "f j ɔ w ɛ k",
    "kiedy": "c ɛ d ɨ", "kino": "c i n ɔ", "gitara": "ɟ i t a r a", "giełda": "ɟ ɛ w d a",
    "historia": "x i s t ɔ r j a", "anglii": "a ŋ ɡ l j i", "maria": "m a r j a", "radio": "r a d j ɔ",
    "chiński": "x i ɲ s c i", "dzika": "dʑ i k a", "ćwiczenie": "tɕ f i tʂ ɛ ɲ ɛ",
    # nasal vowels
    "ręka": "r ɛ ŋ k a", "ząb": "z ɔ m p", "wąs": "v ɔ̃ s", "idę": "i d ɛ", "są": "s ɔ̃", "pięć": "p j ɛ ɲ tɕ",
    "wziął": "v ʑ ɔ w", "wzięli": "v ʑ ɛ l i", "kąt": "k ɔ n t", "mądry": "m ɔ n d r ɨ", "tęcza": "t ɛ n tʂ a",
    "gęś": "ɡ ɛ̃ ɕ", "mąż": "m ɔ̃ ʂ", "wąwóz": "v ɔ̃ v u s", "język": "j ɛ̃ z ɨ k", "zęby": "z ɛ m b ɨ",
    "dąb": "d ɔ m p", "piękny": "p j ɛ ŋ k n ɨ", "sędzia": "s ɛ ɲ dʑ a", "mięso": "m j ɛ̃ s ɔ",
    "zając": "z a j ɔ n ts", "tą": "t ɔ̃", "proszę": "p r ɔ ʂ ɛ",
    # n before a velar
    "bank": "b a ŋ k", "anglia": "a ŋ ɡ l j a", "okienko": "ɔ c ɛ ŋ k ɔ",
    # voicing: final devoicing, regressive assimilation, w and rz after voiceless
    "bóg": "b u k", "chleb": "x l ɛ p", "róg": "r u k", "wódz": "v u ts", "nóż": "n u ʂ", "krew": "k r ɛ f",
    "marchew": "m a r x ɛ f", "prośba": "p r ɔ ʑ b a", "liczba": "l i dʐ b a", "babka": "b a p k a",
    "wszystko": "f ʂ ɨ s t k ɔ", "twój": "t f u j", "kwiat": "k f j a t", "swój": "s f u j", "krzak": "k ʂ a k",
    "przy": "p ʂ ɨ", "trzy": "t ʂ ɨ", "drzewo": "d ʐ ɛ v ɔ", "drzwi": "d ʐ v i", "wrzesień": "v ʐ ɛ ɕ ɛ ɲ",
    "chrzan": "x ʂ a n", "wtedy": "f t ɛ d ɨ", "wdowa": "v d ɔ v a", "odwaga": "ɔ d v a ɡ a",
    "zwykle": "z v ɨ k l ɛ", "szwecja": "ʂ f ɛ ts j a", "twarz": "t f a ʂ", "jabłko": "j a b w k ɔ",
    "zjazd": "z j a s t", "gwiazda": "ɡ v j a z d a", "pszenica": "p ʂ ɛ ɲ i ts a", "sad": "s a t",
    "ogród": "ɔ ɡ r u t", "też": "t ɛ ʂ", "lud": "l u t", "obiad": "ɔ b j a t", "ciężki": "tɕ ɛ̃ ʂ c i",
    "lekko": "l ɛ k k ɔ", "rozszerzyć": "r ɔ s ʂ ɛ ʐ ɨ tɕ", "kraków": "k r a k u f", "wrocław": "v r ɔ ts w a f",
    "gdańsk": "ɡ d a ɲ s k", "mózg": "m u s k", "brzeg": "b ʐ ɛ k", "trzeba": "t ʂ ɛ b a",
    "przecież": "p ʂ ɛ tɕ ɛ ʂ", "szósty": "ʂ u s t ɨ", "żółw": "ʐ u w f", "ćwierć": "tɕ f j ɛ r tɕ",
    # au, eu
    "europa": "ɛ w r ɔ p a", "auto": "a w t ɔ", "nauka": "n a u k a", "muzeum": "m u z ɛ u m",
    "mateusz": "m a t ɛ u ʂ", "zaufanie": "z a u f a ɲ ɛ", "pauza": "p a w z a", "neutralny": "n ɛ w t r a l n ɨ",
    # morpheme boundaries and exceptions
    "marznąć": "m a r z n ɔ ɲ tɕ", "zmarzlina": "z m a r z l i n a", "odżywiać": "ɔ d ʐ ɨ v j a tɕ",
    "nadzieja": "n a dʑ ɛ j a", "odzież": "ɔ dʑ ɛ ʂ", "nadzwyczajny": "n a d z v ɨ tʂ a j n ɨ",
    "podzielić": "p ɔ dʑ ɛ l i tɕ", "tarzan": "t a r z a n",
    # loanwords
    "weekend": "w i k ɛ n t", "pizza": "p i ts a", "jazz": "dʐ ɛ s", "chopin": "ʂ ɔ p ɛ n", "show": "ʂ ɔ w",
}

PHRASES = {
    "jest bardzo": "j ɛ z d | b a r dz ɔ",
    "w domu": "v d ɔ m u", "w kącie": "f k ɔ ɲ tɕ ɛ", "z tobą": "s t ɔ b ɔ̃", "w oknie": "v ɔ k ɲ ɛ",
    "z ojcem": "z ɔ j ts ɛ m", "w polsce": "f p ɔ l s ts ɛ",
    "brat ali": "b r a t | a l i", "bóg wie": "b u k | v j ɛ", "chleb w kuchni": "x l ɛ p | f k u x ɲ i",
    "chleb w domu": "x l ɛ b | v d ɔ m u", "kot wrócił": "k ɔ t | v r u tɕ i w", "pięć dni": "p j ɛ ɲ dʑ | d ɲ i",
    "teraz już": "t ɛ r a s | j u ʂ", "też tak": "t ɛ ʂ | t a k", "jak dobrze": "j a ɡ | d ɔ b ʐ ɛ",
    "bez wody": "b ɛ s | v ɔ d ɨ", "chrząszcz brzmi": "x ʂ ɔ̃ ʐ dʐ | b ʐ m i",
}

STRESS = {
    "dom": (0, None), "matka": (0, None), "telewizor": (2, 0), "muzyka": (0, None), "muzyki": (0, None),
    "muzyce": (0, None), "muzykami": (2, 0), "matematyka": (2, 0), "fabryka": (1, None), "afryka": (0, None),
    "zrobiliśmy": (1, None), "zrobilibyśmy": (1, None), "zrobiłbym": (0, None), "chciałabym": (0, None),
    "byliśmy": (0, None), "uniwersytet": (2, None), "okej": (1, None), "rzeczpospolita": (2, None),
    "mateusz": (1, None), "nauka": (1, None), "europa": (1, None), "nie": (None, None), "się": (None, None),
    "konstantynopol": (3, 0), "pięćdziesiąt": (1, None),
}

NORMALIZED = {
    "Spotkanie o 15:30.": "Spotkanie o piętnastej trzydzieści.",
    "Od 9:05 do 17:00.": "Od dziewiątej zero pięć do siedemnastej.",
    "Przyjdę przed 8:15.": "Przyjdę przed ósmą piętnaście.",
    "Przesunęliśmy to na 15:30.": "Przesunęliśmy to na piętnastą trzydzieści.",
    "W XXI wieku jest inaczej.": "W dwudziestym pierwszym wieku jest inaczej.",
    "Jest 3:45, a pi to 3.14.": "Jest trzecia czterdzieści pięć, a pi to trzy przecinek czternaście.",
    "Urodził się 12.03.2026 r.": "Urodził się dwunastego marca dwa tysiące dwudziestego szóstego roku.",
    "Było to 5 maja 1989 r. w Gdańsku.": "Było to piątego maja tysiąc dziewięćset osiemdziesiątego dziewiątego roku w Gdańsku.",
    "W 2026 r. było ciepło.": "W dwa tysiące dwudziestym szóstym roku było ciepło.",
    "w 2000 roku": "w dwutysięcznym roku",
    "Kosztuje 12,50 zł.": "Kosztuje dwanaście złotych pięćdziesiąt groszy.",
    "1 zł, 2 zł, 5 zł, 22 zł, 12 zł": "jeden złoty, dwa złote, pięć złotych, dwadzieścia dwa złote, dwanaście złotych",
    "$5 i 3 € i 1,99 USD": "pięć dolarów i trzy euro i jeden dolar dziewięćdziesiąt dziewięć centów",
    "Rabat 15%.": "Rabat piętnaście procent.",
    "To 2,5 km, 3 kg i 1 l.": "To dwa przecinek pięć kilometra, trzy kilogramy i jeden litr.",
    "Przyszło 22 tys. osób.": "Przyszło dwadzieścia dwa tysiące osób.",
    "Jan Paweł II żył w XX wieku.": "Jan Paweł drugi żył w dwudziestym wieku.",
    "To był XIX w.": "To był dziewiętnasty wiek.",
    "Bronisz Częstochowy\nI w Ostrej świecisz Bramie!": "Bronisz Częstochowy I w Ostrej świecisz Bramie!",
    "Zygmunt I Stary": "Zygmunt pierwszy Stary",
    "Kupiłem np. dom.": "Kupiłem na przykład dom.",
    "To prof. Nowak i dr Kowalski.": "To profesor Nowak i doktor Kowalski.",
    "Mieszkam na ul. Długiej.": "Mieszkam na ulicy Długiej.",
    "Ul. Długa jest krótka.": "ulica Długa jest krótka.",
    "Szkoła im. Kopernika.": "Szkoła imienia Kopernika.",
    "Dałem im. Potem wyszli.": "Dałem im. Potem wyszli.",
    "Psy, koty itd.": "Psy, koty i tak dalej.",
    "Jest -5 stopni.": "Jest minus pięć stopni.",
    "Agent 007.": "Agent zero zero siedem.",
    "3,05 i 10.000 i 1 000 000": "trzy przecinek zero pięć i dziesięć tysięcy i milion",
    "NFZ i NFZ-u": "en|ef|ˈzet i en|ef|zetu",
    "NATO, UWAGA, OK": "nato, uwaga, oˈkej",
    "Wyślij e-mail przez Wi-Fi.": "Wyślij imejl przez łifi.",
    "biało-czerwony": "biało czerwony",
    "„Cytat” – myślnik…": '"Cytat" — myślnik…',
}


class GoldStandardTests(unittest.TestCase):
    def test_words(self):
        wrong = {word: (phones(word), expected) for word, expected in GOLD.items() if phones(word) != expected}
        self.assertEqual(wrong, {}, "word: (got, expected)")

    def test_phrases_assimilate_across_word_boundaries(self):
        wrong = {text: (phones(text), expected) for text, expected in PHRASES.items() if phones(text) != expected}
        self.assertEqual(wrong, {}, "phrase: (got, expected)")

    def test_stress(self):
        wrong = {word: (stressed(word), expected) for word, expected in STRESS.items() if stressed(word) != expected}
        self.assertEqual(wrong, {}, "word: (got, expected) as (primary vowel index, secondary vowel index)")

    def test_every_output_phone_is_in_the_inventory(self):
        for text in list(GOLD) + list(PHRASES):
            for word in phrase_phones(text.split()):
                for phone, level in word:
                    self.assertIn(phone, PHONE_SET, text)
                    self.assertIn(level, (0, 1, 2))
                    if level:
                        self.assertIn(phone, wymowa.VOWELS, f"stress on a consonant in {text}")


class NumberTests(unittest.TestCase):
    def test_cardinals(self):
        cases = {0: "zero", 1: "jeden", 12: "dwanaście", 21: "dwadzieścia jeden", 100: "sto", 101: "sto jeden",
                 215: "dwieście piętnaście", 1000: "tysiąc", 1001: "tysiąc jeden", 2000: "dwa tysiące",
                 5000: "pięć tysięcy", 12000: "dwanaście tysięcy", 22000: "dwadzieścia dwa tysiące",
                 1_000_000: "milion", 2_500_000: "dwa miliony pięćset tysięcy", 3_000_000_000: "trzy miliardy",
                 1234567: "milion dwieście trzydzieści cztery tysiące pięćset sześćdziesiąt siedem", -4: "minus cztery"}
        self.assertEqual({n: cardinal(n) for n in cases}, cases)

    def test_ordinals_in_every_form_the_normaliser_uses(self):
        cases = {(1, "m_nom"): "pierwszy", (2, "m_gen"): "drugiego", (3, "f_nom"): "trzecia", (3, "f_inst"): "trzecią",
                 (3, "m_gen"): "trzeciego", (2, "f_nom"): "druga", (2, "f_inst"): "drugą", (15, "f_loc"): "piętnastej",
                 (8, "f_inst"): "ósmą", (21, "m_gen"): "dwudziestego pierwszego",
                 (2026, "m_gen"): "dwa tysiące dwudziestego szóstego",
                 (1989, "m_loc"): "tysiąc dziewięćset osiemdziesiątym dziewiątym", (2000, "m_loc"): "dwutysięcznym",
                 (1900, "m_gen"): "tysiąc dziewięćsetnego", (100, "m_nom"): "setny", (0, "f_nom"): "zerowa"}
        self.assertEqual({key: ordinal(*key) for key in cases}, cases)

    def test_noun_agreement(self):
        forms = ("dom", "domy", "domów")
        self.assertEqual([plural(n, forms) for n in (1, 2, 4, 5, 11, 12, 14, 21, 22, 25, 102, 112)],
                         ["dom", "domy", "domy", "domów", "domów", "domów", "domów", "domów", "domy", "domów", "domy", "domów"])


class NormalizationTests(unittest.TestCase):
    def test_sentences(self):
        wrong = {text: (wymowa.normalize(text), expected) for text, expected in NORMALIZED.items()
                 if wymowa.normalize(text) != expected}
        self.assertEqual(wrong, {}, "text: (got, expected)")

    def test_no_digits_survive(self):
        for text in ("Mam 3 koty i 12 psów.", "Tel. 601 234 567", "Wynik 3:1 po 90 minutach", "Pokój 12B, piętro 3."):
            self.assertNotRegex(wymowa.normalize(text), r"\d", text)


class PiecesTests(unittest.TestCase):
    def test_punctuation_separates_phrases(self):
        result = wymowa.pieces("Ala ma kota, a kot ma Alę!")
        self.assertEqual([value for punctuation, value in result if punctuation], [",", "!"])
        self.assertEqual([len(value) for punctuation, value in result if not punctuation], [3, 4])
        self.assertEqual(wymowa.pieces("🙂 ..."), [(True, "…")])

    def test_inventory_and_identity(self):
        self.assertEqual(len(SYMBOLS), 56)
        identity = wymowa.identity()
        self.assertEqual((identity["name"], identity["version"]), ("wymowa", wymowa.VERSION))
        self.assertRegex(identity["fingerprint"], r"^[0-9a-f]{64}$")

    def test_transcribe_is_readable(self):
        self.assertEqual(wymowa.transcribe("Dzień dobry."), "dʑˈɛɲ dˈɔbrɨ .")


if __name__ == "__main__":
    unittest.main()
