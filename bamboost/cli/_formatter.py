import argparse

import rich_argparse
from simple_parsing.wrappers.field_metavar import get_metavar

TEMPORARY_TOKEN = "<__TEMP__>"
from argparse import Action
from typing import Optional, Type


class Formatter(
    rich_argparse.RawDescriptionRichHelpFormatter,
    # rich_argparse.MetavarTypeRichHelpFormatter,
    rich_argparse.ArgumentDefaultsRichHelpFormatter,
):
    """Little shorthand for using some useful HelpFormatters from argparse.

    This class inherits from argparse's `ArgumentDefaultHelpFormatter`,
    `MetavarTypeHelpFormatter` and `RawDescriptionHelpFormatter` classes.

    This produces the following resulting actions:
    - adds a "(default: xyz)" for each argument with a default
    - uses the name of the argument type as the metavar. For example, gives
      "-n int" instead of "-n N" in the usage and description of the arguments.
    - Conserves the formatting of the class and argument docstrings, if given.
    """

    def _get_default_metavar_for_optional(self, action: argparse.Action):
        try:
            return super()._get_default_metavar_for_optional(action)
        except BaseException:
            metavar = self._get_metavar_for_action(action)
            return metavar

    def _get_default_metavar_for_positional(self, action: argparse.Action):
        try:
            return super()._get_default_metavar_for_positional(action)
        except BaseException:
            metavar = self._get_metavar_for_action(action)
            return metavar

    def _get_metavar_for_action(self, action: argparse.Action) -> str:
        return self._get_metavar_for_type(action.type)  # type: ignore

    def _get_metavar_for_type(self, t: Type) -> str:
        return get_metavar(t) or str(t)

    def _get_help_string(self, action: Action) -> Optional[str]:
        help = super()._get_help_string(action=action)
        if help is not None:
            help = help.replace(TEMPORARY_TOKEN, "")
        return help


Formatter.styles.update(
    {
        "argparse.args": "yellow i",
        "argparse.groups": "white bold",
        "argparse.help": "default",
        "argparse.metavar": "white",
        "argparse.syntax": "green",
        "argparse.text": "default",
        "argparse.default": "italic",
    }
)
Formatter.group_name_formatter = str.upper
