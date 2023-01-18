"""Exceptions for data generation module."""

class NoReactantAvailableError(Exception):
    """No second reactant available for the bimolecular reaction."""
    # TODO: for hb.txt, 2 bi-molecular rxn templates (id 78,79) have no matching bblock
    def __init__(self, message):
        super().__init__(message)


class NoReactionAvailableError(Exception):
    """Reactant does not match any reaction template, so no reaction is available."""

    def __init__(self, message):
        super().__init__(message)


class NoBiReactionAvailableError(Exception):
    """Reactants do not match any reaction template."""

    def __init__(self, message):
        super().__init__(message)


class NoReactionPossibleError(Exception):
    """`rdkit` can not yield a valid reaction product, so no reaction is possible."""

    def __init__(self, message):
        super().__init__(message)


class NoMergeReactionPossibleError(Exception):
    """`rdkit` can not yield a valid reaction product, so no reaction is possible.
    Only raised for merge actions to differentiate from add/expand actions."""

    def __init__(self, message):
        super().__init__(message)


class MaxNumberOfActionsError(Exception):
    """Synthetic Tree has exceeded its maximum number of actions."""

    def __init__(self, message):
        super().__init__(message)
