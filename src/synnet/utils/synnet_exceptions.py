class FailedReconstructionError(Exception):
    """Raised when the no valid reactant ist found."""

    pass


class NoSuitableReactantError(FailedReconstructionError):
    """Raised when the no valid reactant ist found."""

    pass


class StateEmbeddingError(FailedReconstructionError):
    """Raise when State Embedding failed"""

    pass
