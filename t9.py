import sys
import BaseHTTPServer
from SimpleHTTPServer import SimpleHTTPRequestHandler
import webbrowser
import json

import re
from nltk.util import ngrams
from nltk.corpus import brown
from itertools import product

class T9(object):
  START_TOKEN = "<s>"
  END_TOKEN = "</s>"
  
  DIGITS = {1: "\\.\\!\\?", 2: "abc", 3: "def", 4: "ghi", 5: "jkl", 6: "mno", 7: "pqrs", 8: "tuv", 9: "wxyz"}

  def __init__(self, corpus = brown, n = 3, files = None):
    """Creates a new word predictor.

    Warning: This takes quite a bit of time. Also due to the fact that nltk.book.FreqDist
             seems to load a bunch of texts when they are not even needed.
             The ngram generation itself also takes up a lot of time, though.

    Keyword Arguments:
    corpus -- text corpus to be used (default Brown University corpus)
    n      -- n-gram dimension (default 2)
    files  -- corpus files to restrict words and sentences to (default None)"""
    self.corpus = corpus
    self.words = set(corpus.words(files))
    self.sentences = corpus.sents(files)
    self.ngram_dimension = n
    self.ngrams = []
    self.letter_ngrams = []

    for sentence in self.sentences:
      tokens = [self.START_TOKEN] + sentence + [self.END_TOKEN]
      for n in range(self.ngram_dimension):
        self.ngrams += ngrams(tokens, n + 1)

    for word in self.words:
      tokens = [self.START_TOKEN] + list(word) + [self.END_TOKEN]
      for n in range(self.ngram_dimension):
        self.letter_ngrams += ngrams(tokens, n + 1)

    from nltk.book import FreqDist

    self.frequencies = FreqDist(self.ngrams)
    self.letter_frequencies = FreqDist(self.letter_ngrams)

  def word_seq_prob(self, words, freq = None):
    """Calculates the probability of a given sequence of words using this WordPredictor's ngram model."""
    prefix = words[-(self.ngram_dimension - 1):-1]
    seq = words[-self.ngram_dimension:]
    if freq is None:
      freq = self.frequencies

    if len(prefix) < self.ngram_dimension - 1 and self.START_TOKEN not in prefix:
      prefix = [self.START_TOKEN] + prefix
    if freq[tuple(prefix)] > 0:
      return freq[tuple(seq)] * 1.0 / freq[tuple(prefix)]
    else:
      return 0.0

  def letter_seq_prob(self, letters):
    """Calculates the probability of a given sequence of letters using this WordPredictor's ngram model."""
    if isinstance(letters, str):
      letters = list(letters)

    return self.word_seq_prob(letters, freq = self.letter_frequencies)

  def possible_words(self, number, words = None):
    """Converts a sequence of digits (1-9) into a set of possible words according to T9."""
    if words is None:
      words = self.words
    letters = map(lambda c: "[" + self.DIGITS[int(c)] + "]", str(number))
    regex = "^" + "".join(letters) + "$"

    return [w for w in words if re.search(regex, w, re.IGNORECASE)]

  def next_possibilities(self, prefix, t9input):
    """Finds for a given sequence of digits a list of possible resulting words as well as their likelihood.

    Keyword Arguments:
    prefix  -- previous input (a list of words)
    t9input -- a sequence of digits (1-9, either as a number or string)"""
    return map(lambda w: (w, self.word_seq_prob(prefix + [w])), self.possible_words(t9input))

  def next_word(self, prefix, t9input, possibilities_function = None, freq = None):
    """Finds the most likely word resulting from a given sequence of digits.
    
    Keyword Arguments:
    prefix  -- previous input (a list of words)
    t9input -- a sequence of digits (1-9, either as a number or string)"""
    if possibilities_function is None:
      possibilities_function = self.next_possibilities
    if freq is None:
      freq = self.frequencies
    possibilities = possibilities_function(prefix, t9input)

    if sum([1 for p in possibilities if p[1] != 0.0]) > 0: # any possible at all?
      return max(possibilities, key = lambda tuple: tuple[1])[0]
    elif len(prefix) > 0: # try to find match with shorter prefix before giving up entirely
      return self.next_word(prefix[:-1], t9input, possibilities_function)
    else:
      # Possibility functions use a start token as an implicit prefix.
      # If nothing could be found like that, just return the word that occurs
      # the most within the corpus (independent of any prefix).
      occurences = map(lambda tuple: (tuple[0], freq[(tuple[0],)]), possibilities)
      if sum([1 for o in occurences if o[1] != 0]):
        return max(occurences, key = lambda tuple: tuple[1])[0]
      else:
        return None

  def get_string(self, t9input):
    print "input: ", t9input
    ps = self.letter_probabilities(t9input, 1)
    if len(ps) > 0:
      return ps[0][0]
    else:
      return None

  def possible_strings(self, digits):
    letters = map(lambda d: self.DIGITS[int(d)].replace("\\", ""), str(digits))
    return map(lambda p: "".join(p), product(*letters))

  def letter_probability(self, word, n = None):
    from math import tanh

    if n is None:
      n = self.ngram_dimension

    grams = ngrams([self.START_TOKEN] + list(word) + [self.END_TOKEN], n)
    probs = map(lambda ls: self.letter_seq_prob(list(ls)), grams)

    return reduce(lambda a, b: a * b, probs)

  def letter_probabilities(self, digits, limit = -1):
    probs = [(word, self.letter_probability(word)) for word in self.possible_strings(digits)]
    probs = sorted(probs, key = lambda tuple: -tuple[1])

    return probs[:limit]

###############################
### HTML GUI - not relevant ###
###############################

class CustomHandler(SimpleHTTPRequestHandler):
  def respond(self, statusCode, statusText, body, contentType = "application/json"):
    self.send_response(statusCode, statusText)
    self.send_header("Content-Type", contentType)
    self.end_headers()

    self.wfile.write(body)

  def do_POST(self):
    if self.path.find("/t9/") != -1:
      method_name = self.path[self.path.find("/t9/")+4:]
      if hasattr(t9, method_name):
        try:
          method = getattr(t9, method_name)
          args = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
          result = json.dumps(method(*args))

          return self.respond(200, "OK", result, "application/json")
        except Exception, e:
          return self.respond(500, "Call failed", str(e), "text/plain")
      else:
        return self.respond(400, "Invalid request", "unknown method: " + method_name, "text/plain")
    elif self.path.endswith("/init"):
      global t9
      if "t9" not in globals():
        print "Intializing T9"
        t9 = T9()
        print "T9 ready"
        return self.send_response(200, "Initialization started")
      else:
        return self.send_response(200, "Already initialized")

    return SimpleHTTPRequestHandler.do_GET(self)

def open_web_gui():
  HandlerClass = CustomHandler
  ServerClass  = BaseHTTPServer.HTTPServer
  Protocol     = "HTTP/1.0"

  if sys.argv[1:]:
    port = int(sys.argv[1])
  else:
    port = 8000
    server_address = ('0.0.0.0', port)

  HandlerClass.protocol_version = Protocol
  httpd = ServerClass(server_address, HandlerClass)

  sa = httpd.socket.getsockname()

  new = 2 # open in a new tab, if possible

  url = "http://localhost:" + str(sa[1]) + "/t9.html"
  webbrowser.open(url,new=new)

  print "Serving T9 on", sa[0], "port", sa[1], "..."
  httpd.serve_forever()

if __name__ == '__main__':
  open_web_gui()