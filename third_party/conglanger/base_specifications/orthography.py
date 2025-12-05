BASELINE = (
    "Orthography rules (global):\n"
    "- All words are lowercase letters a–z only.\n"
    "- No punctuation, digits, diacritics, or other symbols.\n"
    "- No capital letters.\n"
    "\n"
    "Word shape:\n"
    "- Content words: at least 4 letters.\n" # LLMs love 2 letter English words & more letters = less likely overlap with new word generation
    "- Function words (particles, case markers, etc.): 1–3 letters.\n" # used frequently & mirror actual natural languages
    "\n"
    "Anti-English:\n"
    "- Do NOT use real English words (e.g. \"the\", \"and\", \"is\", \"in\", \"to\").\n"
    "- Avoid common English morphemes: \"ing\", \"ed\", \"tion\", \"sion\", \"able\",\n"
    "  \"less\", \"ful\", \"pre\", \"re\", \"un\", \"dis\".\n"
    "- If a word looks close to an English word, change at least 2 letters.\n"
    "\n"
)

RULES_PER_LANGUAGE = {
    # Used when there is no specific target LRL
    "random": (
        "Language-specific orthography (random synthetic):\n"
        "- Letter inventory: {a, e, i, o, u, b, d, f, g, h, j, k, l, m, n, q, r, s, t, w, y, z}.\n"
        "- Use only these letters with the global CV/CVC syllable rules.\n"
    ),

    "arabic": (
        "Language-specific orthography:\n"
        "- Letter inventory: {a, i, u, b, d, f, g, h, j, k, l, m, n, q, r, s, t, w, y, z}.\n"
        "- Use default CV/CVC syllables; do not spell long vowels as double letters.\n"
    ),

    "bulgarian": (
        "Language-specific orthography:\n"
        "- Letter inventory: {a, e, i, o, u, b, v, g, d, z, k, l, m, n, p, r, s, t, f, h}.\n"
        "- Overrides default: allows simple consonant clusters in onset and coda (e.g. CCV, VCC), but avoid very long clusters.\n"
    ),

    "german": (
        "Language-specific orthography:\n"
        "- Letter inventory: {a, e, i, o, u, b, c, d, f, g, h, j, k, l, m, n, p, r, s, t, w}.\n"
        "- Overrides default: permits dense consonant clusters (CCV, CVCC); long vowels are inferred, not spelled with double letters.\n"
    ),

    "greek": (
        "Language-specific orthography:\n"
        "- Letter inventory: {a, e, i, o, u, b, g, d, z, k, l, m, n, p, r, s, t, f, h, x}.\n"
        "- Overrides default: allows CV, CVC, and some CCV syllables; no explicit stress or aspiration marking.\n"
    ),

    "english": (
        "Language-specific orthography:\n"
        "- Letter inventory: {a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z}.\n"
        "- Overrides default: allows very complex syllables (e.g. CCCV, CVCCC) and irregular spellings.\n"
    ),

    "spanish": (
        "Language-specific orthography:\n"
        "- Letter inventory: {a, e, i, o, u, b, c, d, f, g, h, j, k, l, m, n, p, q, r, s, t, v, y}.\n"
        "- Uses mainly simple CV and CVC syllables; consonant clusters are limited.\n"
    ),

    "french": (
        "Language-specific orthography:\n"
        "- Letter inventory: {a, e, i, o, u, y, b, c, d, f, g, h, j, k, l, m, n, p, q, r, s, t, v}.\n"
        "- Overrides default: allows CV, CVC, and some clusters; multiple vowel letters can represent similar vowel qualities.\n"
    ),

    "hindi": (
        "Language-specific orthography:\n"
        "- Letter inventory: {a, e, i, o, u, b, c, d, g, h, j, k, l, m, n, p, r, s, t, v, y}.\n"
        "- Uses mostly CV and CVC syllables; aspiration/retroflexion are encoded by fixed letter sequences, not new symbols.\n"
    ),

    "russian": (
        "Language-specific orthography:\n"
        "- Letter inventory: {a, e, i, o, u, y, b, v, g, d, j, k, l, m, n, p, r, s, t, f, z}.\n"
        "- Overrides default: allows moderate clusters in onsets and codas; softness is implied by vowel context and patterns.\n"
    ),

    "swahili": (
        "Language-specific orthography:\n"
        "- Letter inventory: {a, e, i, o, u, b, d, f, g, h, j, k, l, m, n, p, r, s, t, w, y}.\n"
        "- Favors simple CV and CVC syllables; mostly avoids complex consonant clusters.\n"
    ),

    "thai": (
        "Language-specific orthography:\n"
        "- Letter inventory: {a, e, i, o, u, b, d, f, g, h, k, l, m, n, p, r, s, t, w, y}.\n"
        "- Builds words from simple syllable-like groups; tone and vowel length are not marked.\n"
    ),

    "turkish": (
        "Language-specific orthography:\n"
        "- Letter inventory: {a, e, i, o, u, b, c, d, f, g, h, j, k, l, m, n, p, r, s, t, v, y, z}.\n"
        "- Uses mainly CV and CVC syllables; clusters are limited and usually simple.\n"
    ),

    "urdu": (
        "Language-specific orthography:\n"
        "- Letter inventory: {a, e, i, o, u, b, d, f, g, h, j, k, l, m, n, p, q, r, s, t, w, y, z}.\n"
        "- Uses CV and CVC syllables with occasional simple clusters; emphatic/aspirated contrasts use fixed letter sequences.\n"
    ),

    "vietnamese": (
        "Language-specific orthography:\n"
        "- Letter inventory: {a, e, i, o, u, y, b, c, d, g, h, k, l, m, n, p, q, r, s, t, v}.\n"
        "- Words are built from simple syllable blocks; tone is not indicated in spelling.\n"
    ),

    "chinese": (
        "Language-specific orthography:\n"
        "- Letter inventory: {a, e, i, o, u, b, c, d, f, g, h, j, k, l, m, n, p, q, r, s, t, w, x, y}.\n"
        "- Words are made from one or more syllable-like blocks with an initial and a final; tone is not marked.\n"
    ),
}
