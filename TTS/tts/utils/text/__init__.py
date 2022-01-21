# -*- coding: utf-8 -*-
# adapted from https://github.com/keithito/tacotron

import re
from typing import Dict, List

import gruut

from TTS.tts.utils.text import cleaners
from TTS.tts.utils.text.chinese_mandarin.phonemizer import chinese_text_to_phonemes
from TTS.tts.utils.text.japanese.phonemizer import japanese_text_to_phonemes
from TTS.tts.utils.text.symbols import _bos, _eos, _punctuations, make_symbols, phonemes, symbols

# pylint: disable=unnecessary-comprehension
# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

_phonemes_to_id = {s: i for i, s in enumerate(phonemes)}
_id_to_phonemes = {i: s for i, s in enumerate(phonemes)}

_symbols = symbols
_phonemes = phonemes

# Regular expression matching text enclosed in curly braces:
_CURLY_RE = re.compile(r"(.*?)\{(.+?)\}(.*)")

# Regular expression matching punctuations, ignoring empty space
PHONEME_PUNCTUATION_PATTERN = r"[" + _punctuations.replace(" ", "") + "]+"

# Table for str.translate to fix Gruut inconsistencies (always use IPA!)
GRUUT_TRANS_TABLE = str.maketrans("g", "ɡ")


"""Changed for consistent function naming.
    - "text" will refer to the string of ASCII symbols used as input
    - "phones" will refer to the string of phonemes corresponding to the text
    - "sequence" will be a list of IDs (symbol IDs for text, phoneme IDs for phones) 
"""


# should not have multiple cleaners (if needed, make a new cleaner with multiple steps)
def text_to_phones(text, language, cleaner_name=None, phonemizer=None, phone_sep='', word_sep=' '):
    """Convert graphemes to phonemes.
    Parameters:
            text (str): text to phonemize
            language (str): language of the text
    Returns:
            ph (str): phonemes as a string seperated by "|"
                    ph = "ɪ|g|ˈ|z|æ|m|p|ə|l"
    """
    if cleaner_name:
        text = _clean_text(text, cleaner_name)

    # TO REVIEW : How to have a good implementation for this?
    if language == "zh-CN":
        ph = chinese_text_to_phonemes(text)
        return ph

    if language == "ja-jp":
        ph = japanese_text_to_phonemes(text)
        return ph

    if gruut.is_language_supported(language):
        # Use gruut for phonemization

        ph_list = gruut.text_to_phonemes(
            text,
            lang=language,
            return_format="word_phonemes",
            phonemizer=phonemizer,
        )

        # Join and re-split to break apart dipthongs, suprasegmentals, etc.
        phones_words = [phone_sep.join(word_phonemes) for word_phonemes in ph_list]
        phones = word_sep.join(phones_words)

        # Fix a few phonemes
        phones = phones.translate(GRUUT_TRANS_TABLE)

        return phones

    raise ValueError(f" [!] Language {language} is not supported for phonemization.")


def intersperse(sequence, token):
    result = [token] * (len(sequence) * 2 + 1)
    result[1::2] = sequence
    return result


def pad_with_eos_bos(phoneme_sequence, character_config=None):
    # pylint: disable=global-statement
    global _phonemes_to_id, _bos, _eos
    if character_config:
        _bos = character_config["bos"]
        _eos = character_config["eos"]
        _, _phonemes = make_symbols(**character_config)
        _phonemes_to_id = {s: i for i, s in enumerate(_phonemes)}

    return [_phonemes_to_id[_bos]] + list(phoneme_sequence) + [_phonemes_to_id[_eos]]


# Renamed. This function converts text to a sequence of phone ids, not phonemes to sequence!
def text_to_phoneme_ids(
    text: str,
    cleaner_name: str,
    language: str,
    phonemizer: gruut.Phonemizer = None,
    custom_symbols: List[str] = None,
    character_config: Dict = None,
    add_blank: bool = False,
) -> List[int]:
    """Converts a string of phonemes to a sequence of IDs.
    If `custom_symbols` is provided, it will override the default symbols.

    Args:
      text (str): string to convert to a phoneme sequence
      cleaner_name (List[str]): names of the cleaner functions to run the text through
      language (str): text language key for phonemization.
      enable_eos_bos (bool): whether to append the end-of-sentence and beginning-of-sentence tokens.
      character_config (Dict): dictionary of character parameters to use a custom character set.
      add_blank (bool): option to add a blank token between each token.
      use_espeak_phonemes (bool): use espeak based lexicons to convert phonemes to sequenc

    Returns:
      List[int]: List of integers corresponding to the symbols in the text
    """
    # pylint: disable=global-statement

    phones = text_to_phones(text, language, cleaner_name, phonemizer)
    sequence = phones_to_phoneme_ids(phones, custom_symbols=custom_symbols,
                                     character_config=character_config, add_blank=add_blank)
    return sequence


def phones_to_phoneme_ids(phones: str, custom_symbols: List[str] = None, character_config: Dict = None, add_blank=False):
    remake_phoneme_ids(custom_symbols, character_config)

    sequence = [_phonemes_to_id[phoneme] for phoneme in phones]
    if add_blank:
        sequence = intersperse(phones, len(_phonemes))  # add a blank token (new), whose id number is len(_phonemes)

    return sequence


def remake_symbol_ids(custom_symbols: List[str] = None, character_config: Dict = None):
    global _symbols, _id_to_symbols
    if custom_symbols:
        _symbols = custom_symbols
        _id_to_symbols = {i: s for i, s in enumerate(_symbols)}
    elif character_config:
        _symbols, _ = make_symbols(**character_config)
        _id_to_symbols = {i: s for i, s in enumerate(_symbols)}


def remake_phoneme_ids(custom_phonemes: List[str] = None, character_config: Dict = None):
    global _phonemes, _id_to_phonemes
    if custom_phonemes:
        _phonemes = custom_phonemes
        _id_to_phonemes = {i: s for i, s in enumerate(_phonemes)}
    elif character_config:
        _, _phonemes = make_symbols(**character_config)
        _id_to_phonemes = {i: s for i, s in enumerate(_phonemes)}
        

def phoneme_ids_to_phones(sequence: List, custom_phonemes: List[str] = None,
                          character_config: Dict = None, add_blank=False):
    # pylint: disable=global-statement
    """Converts a sequence of IDs back to a string. This is a 1-1 mapping."""
    
    if add_blank:
        sequence = list(filter(lambda x: x != len(_phonemes), sequence))

    remake_phoneme_ids(custom_phonemes, character_config)

    return ''.join(_id_to_phonemes[phoneme_id] for phoneme_id in sequence)



def text_to_symbol_ids(
    text: str, cleaner_names: List[str], custom_symbols: List[str] = None,
    character_config: Dict = None, add_blank: bool = False
) -> List[int]:
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    If `custom_symbols` is provided, it will override the default symbols.

    Args:
      text (str): string to convert to a sequence
      cleaner_names (List[str]): names of the cleaner functions to run the text through
      character_config (Dict): dictionary of character parameters to use a custom character set.
      add_blank (bool): option to add a blank token between each token.

    Returns:
      List[int]: List of integers corresponding to the symbols in the text
    """
    # pylint: disable=global-statement

    if custom_symbols:
        _symbols = custom_symbols
        _symbol_to_id = {s: i for i, s in enumerate(_symbols)}
    elif character_config:
        _symbols, _ = make_symbols(**character_config)
        _symbol_to_id = {s: i for i, s in enumerate(_symbols)}

    sequence = []

    # Check for curly braces and treat their contents as ARPAbet:
    while text:
        m = _CURLY_RE.match(text)
        if not m:
            sequence = _symbols_to_sequence(_clean_text(text, cleaner_names))
            break
        sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)

    if add_blank:
        sequence = intersperse(sequence, len(_symbols))  # add a blank token (new), whose id number is len(_symbols)
    return sequence


def symbol_ids_to_text(sequence: List, character_config: Dict = None, add_blank=False, custom_symbols: List[str] = None):
    """Converts a sequence of IDs back to a string. This is NOT 1-1 because text_to_symbol_ids cleans the text."""
    # pylint: disable=global-statement
    global _id_to_symbol, _symbols
    if add_blank:
        sequence = list(filter(lambda x: x != len(_symbols), sequence))

    if custom_symbols:
        _symbols = custom_symbols
        _id_to_symbol = {i: s for i, s in enumerate(_symbols)}
    elif character_config:
        _symbols, _ = make_symbols(**character_config)
        _id_to_symbol = {i: s for i, s in enumerate(_symbols)}

    result = ""
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            # Enclose ARPAbet back in curly braces:
            if len(s) > 1 and s[0] == "@":
                s = "{%s}" % s[1:]
            result += s
    return result.replace("}{", " ")


def _clean_text(text, cleaner_name):
    cleaner = getattr(cleaners, cleaner_name)
    if not cleaner:
        raise Exception("Unknown cleaner: %s" % cleaner_name)
    text = cleaner(text)
    return text


def _symbols_to_sequence(syms):
    return [_symbol_to_id[s] for s in syms if _should_keep_symbol(s)]


def _phoneme_to_sequence(phons):
    return [_phonemes_to_id[s] for s in list(phons) if _should_keep_phoneme(s)]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence(["@" + s for s in text.split()])


def _should_keep_symbol(s):
    return s in _symbol_to_id and s not in ["~", "^", "_"]


def _should_keep_phoneme(p):
    return p in _phonemes_to_id and p not in ["~", "^", "_"]
