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


class InvalidSimulationUIDError(ValueError):
    """Raised when a simulation UID is invalid or does not exist."""

    pass


class ForbiddenParameterKeyError(ValueError):
    """Raised when a parameter key is reserved or contains a period."""

    def __init__(self, key: str) -> None:
        super().__init__()
        self.key = key

    def __str__(self) -> str:
        base_message = super().__str__()
        return f"{base_message} Invalid key: '{self.key}'"
