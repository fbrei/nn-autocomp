# Language agnostic Completion Engine

This project explores the possibility of making smart autocompletion suggestions using neural networks.
The key difference is that no rules need to be hard coded, the enginge learns useful suggestions by looking at source code.

The model is pre trained on Python source code. To run the completer, first make sure that you have all requirements fulfilled.

```bash
pip install -r requirements.txt
```

Now you can ask the completer for ideas if you get stuck, for example:

```
$ ./nn-completer.py "def"
<ID>
__init__
main
get
test

$ ./nn-completer "def" "<ID>"
(
<EOL>
.
[
=
```

Where `<ID>` stands for identifier and should be interpreted as a placeholder for a variable or function name for example.
