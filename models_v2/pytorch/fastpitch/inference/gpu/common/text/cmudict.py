""" from https://github.com/keithito/tacotron """

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
import sys
import urllib.request
from pathlib import Path


valid_symbols = [
  'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1', 'AH2',
  'AO', 'AO0', 'AO1', 'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0', 'AY1', 'AY2',
  'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1', 'EH2', 'ER', 'ER0', 'ER1', 'ER2', 'EY',
  'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0', 'IH1', 'IH2', 'IY', 'IY0', 'IY1',
  'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OW0', 'OW1', 'OW2', 'OY', 'OY0',
  'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW',
  'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'
]

_valid_symbol_set = set(valid_symbols)


class CMUDict:
  '''Thin wrapper around CMUDict data. http://www.speech.cs.cmu.edu/cgi-bin/cmudict'''
  def __init__(self, file_or_path=None, heteronyms_path=None, keep_ambiguous=True):
    self._entries = {}
    self.heteronyms = []
    if file_or_path is not None:
      self.initialize(file_or_path, heteronyms_path, keep_ambiguous)

  def initialize(self, file_or_path, heteronyms_path, keep_ambiguous=True):
    if isinstance(file_or_path, str):
      if not Path(file_or_path).exists():
        print("CMUdict missing. Downloading to data/cmudict/.")
        self.download()

      with open(file_or_path, encoding='latin-1') as f:
        entries = _parse_cmudict(f)

    else:
      entries = _parse_cmudict(file_or_path)

    if not keep_ambiguous:
      entries = {word: pron for word, pron in entries.items() if len(pron) == 1}
    self._entries = entries

    if heteronyms_path is not None:
      with open(heteronyms_path, encoding='utf-8') as f:
        self.heteronyms = [l.rstrip() for l in f]

  def __len__(self):
    if len(self._entries) == 0:
      raise ValueError("CMUDict not initialized")
    return len(self._entries)

  def lookup(self, word):
    '''Returns list of ARPAbet pronunciations of the given word.'''
    if len(self._entries) == 0:
      raise ValueError("CMUDict not initialized")
    return self._entries.get(word.upper())

  def download(self):
    url = 'https://github.com/Alexir/CMUdict/raw/master/cmudict-0.7b'
    try:
      Path('cmudict').mkdir(parents=False, exist_ok=True)
      urllib.request.urlretrieve(url, filename='cmudict/cmudict-0.7b')
    except:
      print("Automatic download of CMUdict failed. Try manually with:")
      print()
      print("    bash scripts/download_cmudict.sh")
      print()
      print("and re-run the script.")
      sys.exit(0)


_alt_re = re.compile(r'\([0-9]+\)')


def _parse_cmudict(file):
  cmudict = {}
  for line in file:
    if len(line) and (line[0] >= 'A' and line[0] <= 'Z' or line[0] == "'"):
      parts = line.split('  ')
      word = re.sub(_alt_re, '', parts[0])
      pronunciation = _get_pronunciation(parts[1])
      if pronunciation:
        if word in cmudict:
          cmudict[word].append(pronunciation)
        else:
          cmudict[word] = [pronunciation]
  return cmudict


def _get_pronunciation(s):
  parts = s.strip().split(' ')
  for part in parts:
    if part not in _valid_symbol_set:
      return None
  return ' '.join(parts)
