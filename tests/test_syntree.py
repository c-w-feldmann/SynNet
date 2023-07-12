from __future__ import annotations
from typing import Any
import json
import logging

from pathlib import Path
import pytest

from synnet.utils.data_utils import SyntheticTree, NodeChemical

logger = logging.getLogger(__name__)

base_dir = Path(__file__).parent.absolute()
SYNTREE_FILE = base_dir / "assets/syntree-small.json"


def blake2b(key: str) -> str:
    from hashlib import blake2b as _blake2b

    return _blake2b(key.encode("ascii"), digest_size=16).hexdigest()


def hash_syntree(syntree: SyntheticTree) -> str:
    """Asserting equality in syntrees for amateurs"""
    key = ""
    key = "&".join((node.smiles for node in syntree.chemicals))
    key += "&&" + "&".join(str(node.rxn_id) for node in syntree.reactions)
    return blake2b(key)


@pytest.fixture
def reference_hash() -> str:
    return "56a21aa7ed31577f313401cb9945fc43"


@pytest.fixture
def syntree_as_dict() -> dict[str, Any]:
    with open(SYNTREE_FILE, "rt") as f:
        syntree_dict = json.load(f)
    return syntree_dict


def test_syntree_from_dict(syntree_as_dict: dict[str, Any]) -> None:
    syntree = SyntheticTree.from_dict(syntree_as_dict)
    assert syntree.actions == [0, 0, 2, 1, 3]


def test_create_small_syntree(syntree_as_dict: dict[str, Any]) -> None:
    """Test creating a small syntree.
    This tree should be fairly representative as it has:
        - all 4 actions
        - uni- and bi-molecular rxns
    It does not have:
        - duplicate reactants and a merge reaction (will result in 2 root mols -> bug)
    Rough sketch:

               ┬             ◄─ 5. Action: End
              ┌┴─┐
              │H │
              └┬─┘
              rxn 49         ◄─ 4. Action: Expand
              ┌┴─┐                start: most_recent = H
              │G │                end:   most_recnet = G
              └┬─┘
        ┌────rxn 12 ──┐      ◄─ 3. Action: Merge
       ┌┴─┐          ┌┴─┐          start: most_recent = F
       │C │          │F │          end:   most_recnet = G
       └┬─┘          └┬─┘
        │            rxn 15  ◄─ 2. Action: Add
        │          ┌─┴┐  ┌┴─┐      start: most_recent = C
        │          │D │  │E │      end:   most_recnet = F
        │          └──┘  └──┘
       rxn 47                ◄─ 1. Action: Add
      ┌─┴┐  ┌┴─┐                   start: most_recent = None
      │A │  │B │                   end:   most_recnet = C
      └──┘  └──┘
    """

    A = "CCOc1ccc(CCNC(=O)CCl)cc1OCC"
    B = "C#CCN1CCC(C(=O)O)CC1.Cl"
    C = "CCOc1ccc(CCNC(=O)CN2N=NC=C2CN2CCC(C(=O)O)CC2)cc1OCC"
    D = "C=C(C)C(=O)OCCN=C=O"
    E = "Cc1cc(C#N)ccc1NC1CC1"
    F = "C=C(C)C(=O)OCCNC(=O)N(c1ccc(C#N)cc1C)C1CC1"
    G = "C=C(C)C(=O)OCCNC(=O)N(c1ccc(C2=NNC(C3CCN(Cc4cnnn4CC(=O)NCCc4ccc(OCC)c(OCC)c4)CC3)=N2)cc1C)C1CC1"
    H = "C=C(C)C(=O)OCCNC(=O)N(c1ccc(-c2n[nH]c(C3CCN(Cc4cnnn4CC4=NCCc5cc(OCC)c(OCC)cc54)CC3)n2)cc1C)C1CC1"
    syntree = SyntheticTree()
    assert syntree.depth == 0
    assert syntree.root is None
    assert syntree.get_state() == (None, None), f"{syntree.get_state()=}"
    # 0: Add (bi)
    syntree.update(0, 12, A, B, C)
    assert syntree.depth == 1
    assert syntree.get_state() == (C, None), f"{syntree.get_state()=}"
    # 1: Add (bi)
    syntree.update(0, 47, D, E, F)
    assert syntree.depth == 1
    assert syntree.get_state() == (
        F,
        C,
    ), f"{syntree.get_state()=}"  # the most recent root mol will be at index 0
    # 2: Merge (bi)
    syntree.update(2, 15, F, C, G)
    assert syntree.depth == 2
    assert syntree.get_state() == (G, None), f"{syntree.get_state()=}"
    # 3: Expand (uni)
    syntree.update(1, 49, G, None, H)
    assert syntree.depth == 3
    assert syntree.get_state() == (H, None), f"{syntree.get_state()=}"
    # 4: End
    syntree.update(3, None, None, None, None)
    assert syntree.depth == 3
    assert syntree.get_state() == (H, None), f"{syntree.get_state()=}"
    assert isinstance(syntree.root, NodeChemical)
    assert syntree.root.smiles == H
    assert isinstance(syntree.to_dict(), dict)
    assert syntree.to_dict() == syntree_as_dict


def test_syntree_with_repeating_bblock() -> None:
    """Test creating a syntree with repeating bblock.

    This resulted in a bug with 3 root mols until the fixed in 5ff1218.
    """
    syntree = SyntheticTree()
    # Steps:
    # 1: add uni
    #   A + None -> B
    # 2: add uni
    #   C + None -> D
    # 3: expand bi
    #   D + E -> F
    # 4: merge
    #   F + B -> G
    # 6: add uni
    #   C + None -> D
    # 7: expand bi
    #  D + E -> F
    _ACT = {act: i for i, act in enumerate("add expand merge end".split())}
    syntree = SyntheticTree()
    assert syntree.get_state() == (None, None), f"{syntree.get_state()=}"
    # 1: add uni
    syntree.update(_ACT["add"], 0, "A", None, "B")
    assert syntree.get_state() == ("B", None), f"{syntree.get_state()=}"
    logger.debug(f"Iteration 1 | Syntree depth: {syntree.depth}, state: {syntree.get_state()}")
    # 2: add uni
    syntree.update(_ACT["add"], 1, "C", None, "D")
    assert syntree.get_state() == ("D", "B"), f"{syntree.get_state()=}"
    logger.debug(f"Iteration 2 | Syntree depth: {syntree.depth}, state: {syntree.get_state()}")
    # 3: expand bi
    syntree.update(_ACT["expand"], 2, "D", "E", "F")
    assert syntree.get_state() == ("F", "B"), f"{syntree.get_state()=}"
    logger.debug(f"Iteration 3 | Syntree depth: {syntree.depth}, state: {syntree.get_state()}")
    # 4: merge
    syntree.update(_ACT["merge"], 3, "F", "B", "G")
    assert syntree.get_state() == ("G", None), f"{syntree.get_state()=}"
    logger.debug(f"Iteration 4 | Syntree depth: {syntree.depth}, state: {syntree.get_state()}")
    # 5: add uni
    syntree.update(_ACT["add"], 1, "C", None, "D")
    assert syntree.get_state() == ("D", "G"), f"{syntree.get_state()=}"
    logger.debug(f"Iteration 5 | Syntree depth: {syntree.depth}, state: {syntree.get_state()}")
    # 6: expand bi
    syntree.update(_ACT["expand"], 2, "D", "E", "F")
    assert syntree.get_state() == ("F", "G"), f"{syntree.get_state()=}"
    logger.debug(f"Iteration 6 | Syntree depth: {syntree.depth}, state: {syntree.get_state()}")


def test_syntree_state() -> None:
    """Test is using same small syntree as above."""
    A = "CCOc1ccc(CCNC(=O)CCl)cc1OCC"
    B = "C#CCN1CCC(C(=O)O)CC1.Cl"
    C = "CCOc1ccc(CCNC(=O)CN2N=NC=C2CN2CCC(C(=O)O)CC2)cc1OCC"
    D = "C=C(C)C(=O)OCCN=C=O"
    E = "Cc1cc(C#N)ccc1NC1CC1"
    F = "C=C(C)C(=O)OCCNC(=O)N(c1ccc(C#N)cc1C)C1CC1"
    G = "C=C(C)C(=O)OCCNC(=O)N(c1ccc(C2=NNC(C3CCN(Cc4cnnn4CC(=O)NCCc4ccc(OCC)c(OCC)c4)CC3)=N2)cc1C)C1CC1"
    syntree = SyntheticTree()
    assert len(syntree.get_state()) == 2
    assert syntree.get_state()[0] is None
    assert syntree.get_state()[1] is None
    # 0: Add (bi)
    syntree.update(0, 12, A, B, C)
    assert len(syntree.get_state()) == 2
    assert syntree.get_state()[1] is None
    assert syntree.get_state()[0] == C

    # 1: Add (bi)
    syntree.update(0, 47, D, E, F)
    assert len(syntree.get_state()) == 2
    assert syntree.get_state()[1] == C
    assert syntree.get_state()[0] == F

    # 2: Merge (bi)
    syntree.update(2, 15, F, C, G)
    assert len(syntree.get_state()) == 2
    assert syntree.get_state()[1] is None
    assert syntree.get_state()[0] == G
