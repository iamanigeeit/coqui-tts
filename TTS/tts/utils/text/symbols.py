# -*- coding: utf-8 -*-
"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run
through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
"""


def make_symbols(
    characters,
    phonemes=None,
    punctuations="!'(),-.:;? ",
    pad="_",
    eos="~",
    bos="^",
    make_phonemes_unique=True,
):  # pylint: disable=redefined-outer-name
    """Function to create symbols and phonemes
    TODO: create phonemes_to_id and symbols_to_id dicts here."""
    _symbols = list(characters)
    assert pad and eos and bos
    _symbols = [pad, eos, bos] + _symbols
    if phonemes is None:
        _phonemes = None
    else:
        # this is to keep previous models compatible.
        _phonemes_sorted = sorted(set(phonemes)) if make_phonemes_unique else sorted(phonemes)
        # Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
        # _arpabet = ["@" + s for s in _phonemes_sorted]
        # Export all symbols:
        _phonemes = [pad, eos, bos] + _phonemes_sorted + list(punctuations)
        # _symbols += _arpabet
    return _symbols, _phonemes


"""
Update: The set of phonemes has been updated to match
    https://www.internationalphoneticassociation.org/sites/default/files/phonsymbol.pdf
The old set was missing phonemes and could not handle MSEA languages due to missing ʰ and tone marks
Since the current set is comprehensive, we should use custom phonemes
    only when loading old models or when we really want a restricted phoneme set. 
IPA already provides a list of integer-to-phoneme mappings (not in running order, but consistent after updates)
100-184 consonants, 301-397 vowels, 401-433 diacritics, 501-509 suprasegmentals, 510-533 tonal marks
"""

_pad = "_"
_eos = "~"
_bos = "^"
_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!'(),-.:;? 1234567890"

# Phonemes definition (All IPA characters)
_punctuations = "!'(),-.:;? "
_pulmonic_consonants = "pbmʙɸβɱⱱfvʋtdnrɾθðszʃʒɬɮɹlʈɖɳɽʂʐɻɭcɟɲçʝjʎkɡŋxɣɰʟqɢɴʀχʁħʕʔhɦ"
_non_pulmonic_consonants = "ʘ ǀ ǃ ǂ ǁ ɓ ɗ ʄ ɠ ʛ ʼ "
_other_symbols = "ʍ w ɥ ʜ ʢ ʡ ɕ ʑ ɺ ɧ ͡ ͜ "
_vowels = "i y ɪ ʏ e ø ɛ œ æ a ɶ ɨ ʉ ɘ ɵ ə ɜ ɞ ɐ ɯ u ʊ ɤ o ʌ ɔ ɑ ɒ"
_diacritics = "̥ ̊ ̬ ʰ ̹ ̜ ̟ ̠ ̈ ̽ ̩ ̯ ˞ ̤ ̰ ̼ ʷ ʲ ˠ ˤ ̴ ̝ ̞ ̘ ̙ ̪ ̺ ̻ ̃ n ˡ ̚"
_suprasegmentals = "ˈ ˌ ː ˑ ̆ | . "
_tones_and_accents = "̋ ˥ ́ ˦ ̄ ˧ ̀ ˨ ̏ ˩ ̌ ̂  ᷄ ᷄ ᷅ ᷅  ᷈ ꜜ ꜛ ↗ ↘ "
_unicode_extras = " ɚ ɝ ʣ ʤ ʥ ʦ ʧ ʨ ɫ ʱ ʳ ʴ ʵ ʶ ˀ ̢ ᷆᷆  ᷇  ᷈ ᷉ "

_phonemes = _pulmonic_consonants + _non_pulmonic_consonants + _other_symbols +\
            _vowels + _diacritics + _suprasegmentals + _tones_and_accents + _unicode_extras
_phonemes = _phonemes.replace(' ', '')

symbols, phonemes = make_symbols(_characters, _phonemes, _punctuations, _pad, _eos, _bos)

# Generate ALIEN language
# from random import shuffle
# shuffle(phonemes)


def parse_symbols():
    return {
        "pad": _pad,
        "eos": _eos,
        "bos": _bos,
        "characters": _characters,
        "punctuations": _punctuations,
        "phonemes": _phonemes,
    }


if __name__ == "__main__":
    print(" > TTS symbols {}".format(len(symbols)))
    print(symbols)
    print(" > TTS phonemes {}".format(len(phonemes)))
    print("".join(sorted(phonemes)))
