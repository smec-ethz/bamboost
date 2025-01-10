#!/usr/bin/env python
import sys

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import NestedCompleter, WordCompleter
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.styles import Style
from pygments.lexers.sql import SqlLexer

command_completer = NestedCompleter.from_nested_dict(
    {
        "index": {
            "show": None,
            "clean": None,
            "scan": None,
        }
    }
)

prompt_style = Style.from_dict(
    {
        "prompt": "ansiblue",
    }
)

def main():
    session = PromptSession("bamboost > ", completer=command_completer, style=prompt_style)

    while True:
        try:
            text = session.prompt()
        except KeyboardInterrupt:
            continue  # Control-C pressed. Try again.
        except EOFError:
            break  # Control-D pressed.

    print("GoodBye!")


if __name__ == "__main__":
    main()

