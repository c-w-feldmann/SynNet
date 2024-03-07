from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Optional

from synnet.utils.custom_types import PathType


class PrefixWriter:
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
        with open(file, "rt") as f:
            out = [line.removesuffix("\n") for line in f]
        return out

    def write(self) -> list[str]:
        return self.prefix


class PostfixWriter:
    """Class for writing mermaid diagrams with a postfix."""

    @staticmethod
    def write() -> list[str]:
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

    def write(self, out: list[str]) -> SynTreeWriter:
        out = self.prefixer.write() + out + self.postfixer.write()
        self._text = out
        return self

    def to_file(self, file: PathType, text: Optional[list[str]] = None) -> None:
        text = text or self._text
        if text is None:
            raise ValueError("No text to write.")

        with open(file, "wt") as f:
            f.writelines((line.rstrip() + "\n" for line in text))

    @property
    def text(self) -> list[str]:
        return self.text


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
            TAB_CHAR = " " * 4
            inner = [f"{TAB_CHAR}{line}" for line in inner]
            return [out] + inner + ["end"]

        return wrapper

    return _subgraph
