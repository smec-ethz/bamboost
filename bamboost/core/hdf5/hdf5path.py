from __future__ import annotations

from pathlib import PurePosixPath
from typing import Generator, Union


class HDF5Path(str):
    def __new__(cls, path: str, absolute: bool = True):
        if isinstance(path, HDF5Path):
            return path
        prefix = "/" if absolute else ""
        return super().__new__(cls, prefix + "/".join(filter(None, path.split("/"))))

    def __truediv__(self, other: str) -> HDF5Path:
        return self.joinpath(other)

    def joinpath(self, *other: str) -> HDF5Path:
        return HDF5Path("/".join([self, *other]))

    def relative_to(self, other: Union[HDF5Path, str]) -> HDF5Path:
        other = HDF5Path(other)
        # Ensure we are comparing components, not just string prefix
        self_parts = self.path.parts
        other_parts = other.path.parts

        if (
            len(self_parts) < len(other_parts)
            or self_parts[: len(other_parts)] != other_parts
        ):
            raise ValueError(f"{self} is not a subpath of {other}")

        return HDF5Path("/".join(self_parts[len(other_parts) :]), absolute=False)

    @property
    def parent(self) -> HDF5Path:
        return HDF5Path(self.rsplit("/", 1)[0] or "/")

    @property
    def parents(self) -> Generator[HDF5Path, None, None]:
        current = self
        while current != "/":
            current = current.parent
            yield current

    @property
    def basename(self) -> str:
        return self.name

    @property
    def name(self) -> str:
        return self.rsplit("/", 1)[-1]

    @property
    def path(self) -> PurePosixPath:
        return PurePosixPath(self)

    def is_child_of(self, other: HDF5Path) -> bool:
        return other in self.parents
