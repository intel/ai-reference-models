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
_letters_and_numbers_re = re.compile(
    r"((?:[a-zA-Z]+[0-9]|[0-9]+[a-zA-Z])[a-zA-Z0-9']*)", re.IGNORECASE)

_hardware_re = re.compile(
    '([0-9]+(?:[.,][0-9]+)?)(?:\s?)(tb|gb|mb|kb|ghz|mhz|khz|hz|mm)', re.IGNORECASE)
_hardware_key = {'tb': 'terabyte',
                 'gb': 'gigabyte',
                 'mb': 'megabyte',
                 'kb': 'kilobyte',
                 'ghz': 'gigahertz',
                 'mhz': 'megahertz',
                 'khz': 'kilohertz',
                 'hz': 'hertz',
                 'mm': 'millimeter',
                 'cm': 'centimeter',
                 'km': 'kilometer'}

_dimension_re = re.compile(
    r'\b(\d+(?:[,.]\d+)?\s*[xX]\s*\d+(?:[,.]\d+)?\s*[xX]\s*\d+(?:[,.]\d+)?(?:in|inch|m)?)\b|\b(\d+(?:[,.]\d+)?\s*[xX]\s*\d+(?:[,.]\d+)?(?:in|inch|m)?)\b')
_dimension_key = {'m': 'meter',
                  'in': 'inch',
                  'inch': 'inch'}




def _expand_letters_and_numbers(m):
    text = re.split(r'(\d+)', m.group(0))

    # remove trailing space
    if text[-1] == '':
        text = text[:-1]
    elif text[0] == '':
        text = text[1:]

    # if not like 1920s, or AK47's , 20th, 1st, 2nd, 3rd, etc...
    if text[-1] in ("'s", "s", "th", "nd", "st", "rd") and text[-2].isdigit():
        text[-2] = text[-2] + text[-1]
        text = text[:-1]

    # for combining digits 2 by 2
    new_text = []
    for i in range(len(text)):
        string = text[i]
        if string.isdigit() and len(string) < 5:
            # heuristics
            if len(string) > 2 and string[-2] == '0':
                if string[-1] == '0':
                    string = [string]
                else:
                    string = [string[:-2], string[-2], string[-1]]
            elif len(string) % 2 == 0:
                string = [string[i:i+2] for i in range(0, len(string), 2)]
            elif len(string) > 2:
                string = [string[0]] + [string[i:i+2] for i in range(1, len(string), 2)]
            new_text.extend(string)
        else:
            new_text.append(string)

    text = new_text
    text = " ".join(text)
    return text


def _expand_hardware(m):
    quantity, measure = m.groups(0)
    measure = _hardware_key[measure.lower()]
    if measure[-1] != 'z' and float(quantity.replace(',', '')) > 1:
        return "{} {}s".format(quantity, measure)
    return "{} {}".format(quantity, measure)


def _expand_dimension(m):
    text = "".join([x for x in m.groups(0) if x != 0])
    text = text.replace(' x ', ' by ')
    text = text.replace('x', ' by ')
    if text.endswith(tuple(_dimension_key.keys())):
        if text[-2].isdigit():
            text = "{} {}".format(text[:-1], _dimension_key[text[-1:]])
        elif text[-3].isdigit():
            text = "{} {}".format(text[:-2], _dimension_key[text[-2:]])
    return text


def normalize_letters_and_numbers(text):
    text = re.sub(_hardware_re, _expand_hardware, text)
    text = re.sub(_dimension_re, _expand_dimension, text)
    text = re.sub(_letters_and_numbers_re, _expand_letters_and_numbers, text)
    return text
