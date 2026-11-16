"""Exceptions to the spelling rules, written as Polish respellings.

A respelling is ordinary Polish orthography with two extra marks:

* ``|`` is a morpheme boundary: letters on its two sides never form a digraph
  (``mar|znąć``: r + z, not rz) and a diphthong never spans it.
* ``ˈ`` before a syllable puts the primary stress on that syllable's vowel.

Respelling instead of IPA keeps the list short and readable, and every entry
still goes through the same phonology (voicing, nasal vowels) as other words.
"""

# Whole words (lowercase). Mostly loanwords read the Polish way, and native
# words whose stress is not on the penultimate syllable.
WORDS = {
    # stress exceptions (native or long assimilated)
    "rzeczpospolita": "rzeczpoˈspolita", "rzeczypospolitej": "rzeczypoˈspolitej",
    "uniwersytet": "uniˈwersytet", "opera": "ˈopera", "okej": "oˈkej", "ok": "oˈkej",
    "aha": "aˈha", "hej": "hej",
    # rz read as r + z
    "tarzan": "tar|zan", "erzac": "er|zac", "mierzić": "mier|zić",
    # eu, au that are two syllables
    "ateusz": "ate|usz", "ateizm": "ate|izm", "ateista": "ate|ista", "kofeina": "kofe|ina",
    # loanwords
    "weekend": "łikend", "weekendy": "łikendy", "weekendu": "łikendu", "weekendzie": "łikendzie",
    "e-mail": "imejl", "email": "imejl", "mail": "mejl", "maila": "mejla", "maile": "mejle",
    "iphone": "ajfon", "google": "gugl", "facebook": "fejsbuk", "youtube": "jutub", "netflix": "netfliks",
    "twitter": "tłiter", "windows": "łindołs", "wi-fi": "łifi", "wifi": "łifi", "online": "onlajn",
    "show": "szoł", "hobby": "hobbi", "jazz": "dżez", "pizza": "pica", "pizzy": "picy", "pizzę": "picę",
    "pizzeria": "picerja", "cappuccino": "kapuczino", "latte": "late", "toffi": "tofi",
    "manager": "menedżer", "menedżer": "menedżer", "laptop": "laptop", "smartphone": "smartfon",
    "software": "softłer", "hardware": "hardłer", "playlista": "plejlista", "selfie": "selfi",
    "shakespeare": "szekspir", "chopin": "szopen", "chopina": "szopena", "washington": "łoszington",
    "york": "jork", "new": "nju", "chicago": "szikago", "hollywood": "holiłud", "harry": "hari",
    "jack": "dżek", "john": "dżon", "george": "dżordż", "michael": "majkel", "william": "łiljam",
    "james": "dżejms", "steve": "stiw", "elon": "ilon", "bill": "bil", "joe": "dżo", "taylor": "tejlor",
    "bayern": "bajern", "monachium": "monachjum", "london": "london", "paris": "pari",
}

# Substrings inside words (first match wins, longest listed first). For
# prefix + stem combinations that the letter rules would read as a digraph.
STEMS = (
    ("zmarzl", "zmar|zl"), ("marzn", "mar|zn"), ("marzł", "mar|zł"),
    ("odzwierciedl", "od|zwierciedl"), ("odzysk", "od|zysk"), ("odznacz", "od|znacz"), ("odznak", "od|znak"),
    ("odzwycz", "od|zwycz"), ("nadzwycz", "nad|zwycz"), ("nadzor", "nad|zor"), ("nadzór", "nad|zór"),
    ("podzwrotnik", "pod|zwrotnik"), ("podzbior", "pod|zbior"), ("podzbiór", "pod|zbiór"),
    ("odżyw", "od|żyw"), ("nadżer", "nad|żer"), ("podżeg", "pod|żeg"), ("odżegn", "od|żegn"),
    ("odzew", "od|zew"), ("odzywa", "od|zywa"), ("odzywk", "od|zywk"),
)

# Stems of nouns in -yka/-ika that keep the stress on the antepenultimate
# syllable in forms ending in -a/-i/-ę/-ą (and -ce for the c-stem: muzyce).
ANTEPENULTIMATE_IKA = frozenset("""
muzyk fizyk matematyk polityk technik gramatyk logik republik informatyk botanik mechanik etyk estetyk
akustyk optyk dynamik fonetyk statystyk lingwistyk elektronik ceramik grafik klinik kronik panik
praktyk krytyk taktyk tematyk problematyk retoryk logistyk plastyk gimnastyk symbolik ameryk afryk
korsyk mozaik harmonik metafizyk fizjonomik semantyk pragmatyk cybernetyk robotyk genetyk
dydaktyk publicystyk turystyk elektryk liryk epik
""".split())

# Monosyllabic function words that do not take stress of their own.
UNSTRESSED = frozenset("""
w we z ze na do od po o u za bez dla nad pod przed przez przy ku spod znad zza
i a że by bo czy lub ni aż gdy lecz więc
się mi ci go mu ją je nie no
""".split())

# Single-consonant prepositions: they join the following word (no word
# boundary, no final devoicing before a vowel: "w oknie" is [v ɔkɲɛ]).
PROCLITICS = frozenset(("w", "z"))
