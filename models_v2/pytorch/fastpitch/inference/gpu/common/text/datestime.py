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
_ampm_re = re.compile(
    r'([0-9]|0[0-9]|1[0-9]|2[0-3]):?([0-5][0-9])?\s*([AaPp][Mm]\b)')


def _expand_ampm(m):
    matches = list(m.groups(0))
    txt = matches[0]
    txt = txt if int(matches[1]) == 0 else txt + ' ' + matches[1]

    if matches[2][0].lower() == 'a':
        txt += ' a.m.'
    elif matches[2][0].lower() == 'p':
        txt += ' p.m.'

    return txt


def normalize_datestime(text):
    text = re.sub(_ampm_re, _expand_ampm, text)
    #text = re.sub(r"([0-9]|0[0-9]|1[0-9]|2[0-3]):([0-5][0-9])?", r"\1 \2", text)
    return text
