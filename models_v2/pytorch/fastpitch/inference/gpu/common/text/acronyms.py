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

import re
from . import cmudict

_letter_to_arpabet = {
    'A': 'EY1',
    'B': 'B IY1',
    'C': 'S IY1',
    'D': 'D IY1',
    'E': 'IY1',
    'F': 'EH1 F',
    'G': 'JH IY1',
    'H': 'EY1 CH',
    'I': 'AY1',
    'J': 'JH EY1',
    'K': 'K EY1',
    'L': 'EH1 L',
    'M': 'EH1 M',
    'N': 'EH1 N',
    'O': 'OW1',
    'P': 'P IY1',
    'Q': 'K Y UW1',
    'R': 'AA1 R',
    'S': 'EH1 S',
    'T': 'T IY1',
    'U': 'Y UW1',
    'V': 'V IY1',
    'X': 'EH1 K S',
    'Y': 'W AY1',
    'W': 'D AH1 B AH0 L Y UW0',
    'Z': 'Z IY1',
    's': 'Z'
}

# Acronyms that should not be expanded
hardcoded_acronyms = [
    'BMW', 'MVD', 'WDSU', 'GOP', 'UK', 'AI', 'GPS', 'BP', 'FBI', 'HD',
    'CES', 'LRA', 'PC', 'NBA', 'BBL', 'OS', 'IRS', 'SAC', 'UV', 'CEO', 'TV',
    'CNN', 'MSS', 'GSA', 'USSR', 'DNA', 'PRS', 'TSA', 'US', 'GPU', 'USA',
    'FPCC', 'CIA']

# Words and acronyms that should be read as regular words, e.g., NATO, HAPPY, etc.
uppercase_whiteliset = []

acronyms_exceptions = {
    'NVIDIA': 'N.VIDIA',
}

non_uppercase_exceptions = {
    'email': 'e-mail',
}

# must ignore roman numerals
_acronym_re = re.compile(r'([a-z]*[A-Z][A-Z]+)s?\.?')
_non_uppercase_re = re.compile(r'\b({})\b'.format('|'.join(non_uppercase_exceptions.keys())), re.IGNORECASE)


def _expand_acronyms_to_arpa(m, add_spaces=True):
    acronym = m.group(0)

    # remove dots if they exist
    acronym = re.sub('\.', '', acronym)

    acronym = "".join(acronym.split())
    arpabet = cmudict.lookup(acronym)

    if arpabet is None:
        acronym = list(acronym)
        arpabet = ["{" + _letter_to_arpabet[letter] + "}" for letter in acronym]
        # temporary fix
        if arpabet[-1] == '{Z}' and len(arpabet) > 1:
            arpabet[-2] = arpabet[-2][:-1] + ' ' + arpabet[-1][1:]
            del arpabet[-1]

        arpabet = ' '.join(arpabet)
    elif len(arpabet) == 1:
        arpabet = "{" + arpabet[0] + "}"
    else:
        arpabet = acronym

    return arpabet


def normalize_acronyms(text):
    text = re.sub(_acronym_re, _expand_acronyms_to_arpa, text)
    return text


def expand_acronyms(m):
    text = m.group(1)
    if text in acronyms_exceptions:
        text = acronyms_exceptions[text]
    elif text in uppercase_whiteliset:
        text = text
    else:
        text = '.'.join(text) + '.'

    if 's' in m.group(0):
        text = text + '\'s'

    if text[-1] != '.' and m.group(0)[-1] == '.':
        return text + '.'
    else:
        return text


def spell_acronyms(text):
    text = re.sub(_non_uppercase_re, lambda m: non_uppercase_exceptions[m.group(0).lower()], text)
    text = re.sub(_acronym_re, expand_acronyms, text)
    return text
