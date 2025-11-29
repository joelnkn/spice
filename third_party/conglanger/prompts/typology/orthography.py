BASELINE = (
    "Orthography rules:\n"
    "• All words are lowercase a–z only.\n"
    "• No punctuation inside words (no apostrophes, hyphens, periods).\n"
    "• No digits.\n"
    "• No IPA symbols or diacritics.\n"
    "• No capital letters."
)

RULES_PER_LANGUAGE = {
    "arabic": "Uses only the letters {a, i, u, b, d, f, g, h, j, k, l, m, n, q, r, s, t, w, y, z}. Distinguishes many consonants with separate letters but does not mark vowel length; long and short vowels use the same vowel symbols.",
    "bulgarian": "Uses only the letters {a, e, i, o, u, b, v, g, d, z, k, l, m, n, p, r, s, t, f, h}. Employs a small vowel set and allows various consonant clusters with no special marking for palatal or softened consonants.",
    "german": "Uses only the letters {a, e, i, o, u, b, c, d, f, g, h, j, k, l, m, n, p, r, s, t, w}. Permits dense consonant clusters and double consonants; long vowels are not indicated with extra symbols and are inferred from context.",
    "greek": "Uses only the letters {a, e, i, o, u, b, g, d, z, k, l, m, n, p, r, s, t, f, h, x}. Employs a limited set of vowel symbols, some overlapping in sound, and does not mark stress or aspiration with special letters; certain consonant sequences may function as single sounds.",
    "english": "Uses only the letters {a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z}. Allows complex syllable structures with cluster-heavy onsets and codas, and includes spellings that are not strictly phonemic.",
    "spanish": "Uses only the letters {a, e, i, o, u, b, c, d, f, g, h, j, k, l, m, n, p, q, r, s, t, v, y}. Prefers simple consonant clusters and maintains a near one-to-one mapping between letters and phonemes for both vowels and consonants.",
    "french": "Uses only the letters {a, e, i, o, u, y, b, c, d, f, g, h, j, k, l, m, n, p, q, r, s, t, v}. Relies on multiple vowel letters and letter combinations whose spellings can merge several similar vowel qualities without extra markings.",
    "hindi": "Uses only the letters {a, e, i, o, u, b, c, d, g, h, j, k, l, m, n, p, r, s, t, v, y}. Represents consonants with plain letters and does not use special symbols for aspiration or retroflexion; such contrasts rely on fixed letter sequences.",
    "russian": "Uses only the letters {a, e, i, o, u, y, b, v, g, d, j, k, l, m, n, p, r, s, t, f, z}. Uses basic consonant letters without explicit softness markers, relying on surrounding vowels and fixed patterns rather than separate soft-sign symbols.",
    "swahili": "Uses only the letters {a, e, i, o, u, b, d, f, g, h, j, k, l, m, n, p, r, s, t, w, y}. Favors simple consonant–vowel patterns, mostly avoids complex clusters, and uses a small, regular set of vowel symbols.",
    "thai": "Uses only the letters {a, e, i, o, u, b, d, f, g, h, k, l, m, n, p, r, s, t, w, y}. Builds words from syllable-like letter groups without indicating tone or vowel length; syllables that differ only in tone or length share the same spelling.",
    "turkish": "Uses only the letters {a, e, i, o, u, b, c, d, f, g, h, j, k, l, m, n, p, r, s, t, v, y, z}. Employs a regular vowel system that tends to show harmony patterns, and represents all consonants with single plain letters without extra marks.",
    "urdu": "Uses only the letters {a, e, i, o, u, b, d, f, g, h, j, k, l, m, n, p, q, r, s, t, w, y, z}. Represents consonants with plain letters and does not use extra symbols for emphatic or aspirated series; these contrasts are encoded by consistent letter sequences.",
    "vietnamese": "Uses only the letters {a, e, i, o, u, y, b, c, d, g, h, k, l, m, n, p, q, r, s, t, v}. Employs a modest set of vowel letters and does not indicate tone or other diacritic-based contrasts; forms differing only in tone share the same spelling.",
    "chinese": "Uses only the letters {a, e, i, o, u, b, c, d, f, g, h, j, k, l, m, n, p, q, r, s, t, w, x, y}. Forms words from one or more syllable-like blocks made of an initial and a final, and does not mark tone; tonal differences are not reflected in the spelling."
}
