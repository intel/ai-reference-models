""" adapted from https://github.com/keithito/tacotron """

# Copyright (c) 2017 Keith Ito

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
    1. "english_cleaners" for English text
    2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
         the Unidecode library (https://pypi.python.org/pypi/Unidecode)
    3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
         the symbols in symbols.py to match your data).
'''

import re
from .abbreviations import normalize_abbreviations
from .acronyms import normalize_acronyms, spell_acronyms
from .datestime import normalize_datestime
from .letters_and_numbers import normalize_letters_and_numbers
from .numerical import normalize_numbers
from .unidecoder import unidecoder


# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')


def expand_abbreviations(text):
    return normalize_abbreviations(text)


def expand_numbers(text):
    return normalize_numbers(text)


def expand_acronyms(text):
    return normalize_acronyms(text)


def expand_datestime(text):
    return normalize_datestime(text)


def expand_letters_and_numbers(text):
    return normalize_letters_and_numbers(text)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def separate_acronyms(text):
    text = re.sub(r"([0-9]+)([a-zA-Z]+)", r"\1 \2", text)
    text = re.sub(r"([a-zA-Z]+)([0-9]+)", r"\1 \2", text)
    return text


def convert_to_ascii(text):
    return unidecoder(text)


def basic_cleaners(text):
    '''Basic pipeline that collapses whitespace without transliteration.'''
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    '''Pipeline for non-English text that transliterates to ASCII.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
    '''Pipeline for English text, with number and abbreviation expansion.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners_v2(text):
    text = convert_to_ascii(text)
    text = expand_datestime(text)
    text = expand_letters_and_numbers(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = spell_acronyms(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    # compatibility with basic_english symbol set
    text = re.sub(r'/+', ' ', text)
    return text
