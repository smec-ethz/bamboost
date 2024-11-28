import argparse
from functools import lru_cache

import argcomplete


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.completer = argcomplete.finders.CompletionFinder(self)

    def set_prefix(self, prefix):
        self._prefix = prefix

    @property
    def prefix(self):
        if hasattr(self, "_prefix"):
            return self._prefix
        self._prefix = ""
        return self._prefix

    def current_suggestions(self, text):
        # preprend prefix to text
        text = self.prefix + text

        # split text into parts
        cword_prequote, cword_prefix, cword_suffix, comp_words, first_colon_pos = (
            argcomplete.lexers.split_line(text)
        )
        # inject dummy word at 0
        comp_words.insert(0, "dummy")
        matches = self.completer._get_completions(
            comp_words, cword_prefix, cword_prequote, first_colon_pos
        )

        # remove options (--) if not started with it
        if not cword_prefix.startswith("-"):
            matches = [m for m in matches if not m.startswith("-")]
        return sorted(matches, key=lambda x: len(x))

    def eval(self, text):
        # preprend prefix to text
        text = self.prefix + text
        args = self.parse_args(text.split())

    def parse_string(self, text):
        # preprend prefix to text
        text = self.prefix + text
        arg_list = text.split()
        return self.parse_args(arg_list)
