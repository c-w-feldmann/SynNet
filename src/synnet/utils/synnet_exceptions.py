"""Custom exceptions for the synnet package."""

class FailedReconstructionError(Exception):
    """Raised when the no valid reactant ist found."""


class NoSuitableReactantError(FailedReconstructionError):
    """Raised when the no valid reactant ist found."""


class StateEmbeddingError(FailedReconstructionError):
    """Raise when State Embedding failed"""
