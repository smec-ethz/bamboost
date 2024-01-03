# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code

import subprocess

__all__ = ["GitStateGetter"]


class GitStateGetter:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _git_command(command: str) -> str:
        process = subprocess.Popen(
            command.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        stdout, stderr = process.communicate()
        return str(stdout)

    def create_git_string(self) -> str:
        self.string = ""

        self.string += "\n"
        self.string += "----- REMOTE ------ \n"
        self.string += self._git_command("git remote -v")

        self.string += "\n"
        self.string += "----- BRANCH ------ \n"
        self.string += self._git_command("git branch -v")

        self.string += "\n"
        self.string += "----- LAST COMMIT ------ \n"
        self.string += self._git_command("git rev-parse HEAD")

        self.string += "\n"
        self.string += "----- STATUS ------ \n"
        self.string += self._git_command("git status")

        self.string += "\n"
        self.string += "----- DIFFERENCE ------ \n"
        self.string += self._git_command("git diff HEAD")

        return self.string
