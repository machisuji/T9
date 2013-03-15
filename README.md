# T9 in Python

Just a little toy T9 implementation in Python.
Requires nltk.

## How to run

### (micro) Benchmark

`python t9.py --benchmark`

*Notes*: As expected the letter model does not work that well (1/3 of words guessed right) for a mostly unrelated corpus.
         It naturally works better when words from the training corpus are used.
         You can also use the letter model in the demo (see below) and there it doesn't work too bad,
         but obviously still not as good as the word model, which also considers context.

### Demo

`python t9.py`

This will open a web browser that navigates to a HTML GUI.
