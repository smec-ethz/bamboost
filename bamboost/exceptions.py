class InvalidCollectionError(OSError):
    """Raised when a collection is invalid or does not exist."""

    pass


class DuplicateSimulationError(ValueError):
    """Raised when trying to create a simulation whose parameters already exist in the
    collection."""

    def __init__(self, duplicates: tuple[str, ...]) -> None:
        super().__init__()
        self.duplicates = duplicates

    def __str__(self) -> str:
        base_message = super().__str__()
        return f"{base_message} Duplicates: {self.duplicates}"
