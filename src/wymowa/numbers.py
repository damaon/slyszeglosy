"""Polish numbers in words: cardinals, ordinals in the cases the normaliser
needs, and noun agreement after a number."""

ONES = ("zero", "jeden", "dwa", "trzy", "cztery", "pięć", "sześć", "siedem", "osiem", "dziewięć", "dziesięć",
        "jedenaście", "dwanaście", "trzynaście", "czternaście", "piętnaście", "szesnaście", "siedemnaście",
        "osiemnaście", "dziewiętnaście")
TENS = ("", "", "dwadzieścia", "trzydzieści", "czterdzieści", "pięćdziesiąt", "sześćdziesiąt", "siedemdziesiąt",
        "osiemdziesiąt", "dziewięćdziesiąt")
HUNDREDS = ("", "sto", "dwieście", "trzysta", "czterysta", "pięćset", "sześćset", "siedemset", "osiemset", "dziewięćset")
SCALES = ((10 ** 9, ("miliard", "miliardy", "miliardów")), (10 ** 6, ("milion", "miliony", "milionów")),
          (10 ** 3, ("tysiąc", "tysiące", "tysięcy")))

# Nominative masculine ordinals; other forms are derived by ``decline``.
ORDINAL_ONES = ("zerowy", "pierwszy", "drugi", "trzeci", "czwarty", "piąty", "szósty", "siódmy", "ósmy", "dziewiąty",
                "dziesiąty", "jedenasty", "dwunasty", "trzynasty", "czternasty", "piętnasty", "szesnasty",
                "siedemnasty", "osiemnasty", "dziewiętnasty")
ORDINAL_TENS = ("", "", "dwudziesty", "trzydziesty", "czterdziesty", "pięćdziesiąty", "sześćdziesiąty",
                "siedemdziesiąty", "osiemdziesiąty", "dziewięćdziesiąty")
ORDINAL_HUNDREDS = ("", "setny", "dwusetny", "trzechsetny", "czterechsetny", "pięćsetny", "sześćsetny",
                    "siedemsetny", "osiemsetny", "dziewięćsetny")
THOUSAND_PREFIX = ("", "", "dwu", "trzy", "cztero", "pięcio", "sześcio", "siedmio", "ośmio", "dziewięcio")

# Adjective endings: (hard stem -y, soft/velar stem -i)
ENDINGS = {
    "m_nom": ("y", "i"), "m_gen": ("ego", "iego"), "m_loc": ("ym", "im"),
    "f_nom": ("a", "a"), "f_gen": ("ej", "iej"), "f_loc": ("ej", "iej"), "f_inst": ("ą", "ą"), "f_acc": ("ą", "ą"),
}


def plural(n, forms):
    """``forms`` = (1, 2-4, 5+) noun forms: jeden dom, dwa domy, pięć domów."""
    one, few, many = forms[:3]
    if n == 1:
        return one
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return few
    return many


def cardinal(n):
    if n < 0:
        return "minus " + cardinal(-n)
    if n < 20:
        return ONES[n]
    words = []
    for size, forms in SCALES:
        if n >= size:
            count, n = divmod(n, size)
            words.append(forms[0] if count == 1 else f"{cardinal(count)} {plural(count, forms)}")
    if n >= 100:
        words.append(HUNDREDS[n // 100])
        n %= 100
    if n >= 20:
        words.append(TENS[n // 10])
        n %= 10
        if n:
            words.append(ONES[n])
    elif n:
        words.append(ONES[n])
    return " ".join(words)


def decline(nominative, form):
    """``pierwszy`` -> ``pierwszego`` etc. for the forms in ``ENDINGS``."""
    soft = nominative.endswith("i")
    stem = nominative[:-1]
    ending = ENDINGS[form][1 if soft else 0]
    if soft and ending in ("a", "ą") and not stem.endswith(("g", "k")):
        ending = "i" + ending                      # trzecia, trzecią (but druga)
    return stem + ending


def ordinal(n, form="m_nom"):
    """Ordinal of ``n`` in ``form``. Only the last two digits (or the last
    nonzero group) are ordinal words: 1989 -> tysiąc dziewięćset
    osiemdziesiąty dziewiąty, 2000 -> dwutysięczny."""
    if n < 0:
        raise ValueError("ordinal of a negative number")
    low = n % 100
    high = n - low
    if low == 0 and n:
        if n % 1000 == 0 and n < 10000:
            word = THOUSAND_PREFIX[n // 1000] + "tysięczny"
            return decline(word, form)
        if n % 1000:
            prefix = cardinal(n - n % 1000) + " " if n >= 1000 else ""
            return prefix + decline(ORDINAL_HUNDREDS[(n % 1000) // 100], form)
        return decline(cardinal(n // 1000) + "tysięczny", form)
    parts = [cardinal(high)] if high else []
    if low < 20:
        parts.append(decline(ORDINAL_ONES[low], form))
    else:
        parts.append(decline(ORDINAL_TENS[low // 10], form))
        if low % 10:
            parts.append(decline(ORDINAL_ONES[low % 10], form))
    return " ".join(parts)


def digits(text):
    """Digit by digit: ``"0147"`` -> ``"zero jeden cztery siedem"``."""
    return " ".join(ONES[int(d)] for d in text)
