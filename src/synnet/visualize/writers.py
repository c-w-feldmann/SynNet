"""Module for writing mermaid diagrams to a file."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Optional, Self

from synnet.utils.custom_types import PathType

TAB_CHAR = " " * 4


class PrefixWriter:  # pylint: disable=too-few-public-methods
    """Class for writing mermaid diagrams with a prefix."""

    prefix: list[str]

    def __init__(self, file: Optional[str] = None):
        self.prefix = self._default_prefix() if file is None else self._load(file)

    def _default_prefix(self) -> list[str]:
        md = [
            "# Synthetic Tree Visualisation",
            "",
            "Legend",
            "- :green_square: Building Block",
            "- :orange_square: Intermediate",
            "- :blue_square: Final Molecule",
            "- :red_square: Target Molecule",
            "",
        ]
        start = ["```mermaid"]
        theming = [
            "%%{init: {",
            "    'theme': 'base',",
            "    'themeVariables': {",
            "        'backgroud': '#ffffff',",
            "        'primaryColor': '#ffffff',",
            "        'clusterBkg': '#ffffff',",
            "        'clusterBorder': '#000000',",
            "        'edgeLabelBackground':'#dbe1e1',",
            "        'fontSize': '20px'",
            "        }",
            "    }",
            "}%%",
        ]
        diagram_id = ["graph BT"]
        style = [
            "classDef buildingblock stroke:#00d26a,stroke-width:2px",
            "classDef intermediate stroke:#ff6723,stroke-width:2px",
            "classDef final stroke:#0074ba,stroke-width:2px",
            "classDef target stroke:#f8312f,stroke-width:2px",
        ]
        return md + start + theming + diagram_id + style

    @staticmethod
    def _load(file: PathType) -> list[str]:
        """Load the prefix from a file.

        Parameters
        ----------
        file : PathType
            The file to load the prefix from.
        """
        with open(file, "rt", encoding="UTF-8") as f:
            out = [line.removesuffix("\n") for line in f]
        return out

    def write(self) -> list[str]:
        """Return the prefix.

        Returns
        -------
        list[str]
            The prefix.
        """
        return self.prefix


class PostfixWriter:  # pylint: disable=too-few-public-methods
    """Class for writing mermaid diagrams with a postfix."""

    @staticmethod
    def write() -> list[str]:
        """Return the postfix.

        Returns
        -------
        list[str]
            The postfix.
        """
        return ["```"]


class SynTreeWriter:
    """Class for writing mermaid diagrams."""

    prefixer: PrefixWriter
    postfixer: PostfixWriter
    _text: Optional[list[str]]

    def __init__(
        self,
        prefixer: PrefixWriter = PrefixWriter(),
        postfixer: PostfixWriter = PostfixWriter(),
    ) -> None:
        self.prefixer = prefixer
        self.postfixer = postfixer
        self._text = None

    def write(self, out: list[str]) -> Self:
        """Write the text to the writer."""
        out = self.prefixer.write() + out + self.postfixer.write()
        self._text = out
        return self

    def to_file(self, file: PathType, text: Optional[list[str]] = None) -> None:
        """Write the text to a file.

        Parameters
        ----------
        file : PathType
            The file to write the text to.
        text : Optional[list[str]], optional
            The text to write, by default None.
        """
        text = text or self._text
        if text is None:
            raise ValueError("No text to write.")

        with open(file, "wt", encoding="UTF-8") as f:
            f.writelines((line.rstrip() + "\n" for line in text))


def subgraph(
    argument: str = "",
) -> Callable[[Callable[[Any], Any]], Callable[[Any], Any]]:
    """Decorator that writes a named mermaid subparagraph.

    Example output:
    ```
    subparagraph argument
        <output of function that is decorated>
    end
    ```
    """

    def _subgraph(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
        @wraps(func)
        def wrapper(*args: list[Any], **kwargs: dict[str, Any]) -> list[str]:
            out = f"subgraph {argument}"
            inner = func(*args, **kwargs)
            # add a tab to inner
            inner = [f"{TAB_CHAR}{line}" for line in inner]
            return [out] + inner + ["end"]

        return wrapper

    return _subgraph
