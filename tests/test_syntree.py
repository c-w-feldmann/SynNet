"""Unit test for the SyntheticTree class."""

import json
import unittest
from pathlib import Path

from loguru import logger

from synnet.utils.data_utils import NodeChemical, SyntheticTree

BASE_PATH = Path(__file__).parent.absolute()
SYNTREE_FILE = BASE_PATH / "assets/syntree-small.json"


class TestSynTree(unittest.TestCase):
    """Test the SyntheticTree class."""

    def setUp(self) -> None:
        self.reference_hash = "56a21aa7ed31577f313401cb9945fc43"
        with open(SYNTREE_FILE, "rt", encoding="UTF-8") as f:
            self.syntree_dict = json.load(f)

    def test_syntree_from_dict(self) -> None:
        """Test creating a syntree from a dict."""
        syntree = SyntheticTree.from_dict(self.syntree_dict)
        self.assertEqual(syntree.actions, [0, 0, 2, 1, 3])

    def test_create_small_syntree(self) -> None:
        """Test creating a small syn_tree.
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
        smiles_list = [
            "CCOc1ccc(CCNC(=O)CCl)cc1OCC",
            "C#CCN1CCC(C(=O)O)CC1.Cl",
            "CCOc1ccc(CCNC(=O)CN2N=NC=C2CN2CCC(C(=O)O)CC2)cc1OCC",
            "C=C(C)C(=O)OCCN=C=O",
            "Cc1cc(C#N)ccc1NC1CC1",
            "C=C(C)C(=O)OCCNC(=O)N(c1ccc(C#N)cc1C)C1CC1",
            "C=C(C)C(=O)OCCNC(=O)N(c1ccc(C2=NNC(C3CCN(Cc4cnnn4CC(=O)NCCc4ccc(OCC)c(OCC)c4)CC3)=N2)cc1C)C1CC1",
            "C=C(C)C(=O)OCCNC(=O)N(c1ccc(-c2n[nH]c(C3CCN(Cc4cnnn4CC4=NCCc5cc(OCC)c(OCC)cc54)CC3)n2)cc1C)C1CC1",
        ]
        syn_tree = SyntheticTree()
        self.assertEqual(syn_tree.depth, 0)
        self.assertEqual(syn_tree.root, None)
        self.assertEqual(syn_tree.get_state(), (None, None), f"{syn_tree.get_state()=}")
        # 0: Add (bi)
        syn_tree.update(0, 12, smiles_list[0], smiles_list[1], smiles_list[2])
        self.assertEqual(syn_tree.depth, 1)
        self.assertEqual(
            syn_tree.get_state(), (smiles_list[2], None), f"{syn_tree.get_state()=}"
        )
        # 1: Add (bi)
        syn_tree.update(0, 47, smiles_list[4], smiles_list[5], smiles_list[6])
        self.assertEqual(syn_tree.depth, 2)
        self.assertEqual(
            syn_tree.get_state(), (smiles_list[6], None), f"{syn_tree.get_state()=}"
        )

        # 2: Merge (bi)
        syn_tree.update(2, 15, smiles_list[5], smiles_list[2], smiles_list[6])
        self.assertEqual(syn_tree.depth, 2)
        self.assertEqual(
            syn_tree.get_state(), (smiles_list[6], None), f"{syn_tree.get_state()=}"
        )

        # 3: Expand (uni)
        syn_tree.update(1, 49, smiles_list[6], None, smiles_list[7])
        self.assertEqual(syn_tree.depth, 3)
        self.assertEqual(
            syn_tree.get_state(), (smiles_list[7], None), f"{syn_tree.get_state()=}"
        )
        # 4: End
        syn_tree.update(3, None, None, None, None)
        self.assertEqual(syn_tree.depth, 3)
        self.assertEqual(
            syn_tree.get_state(), (smiles_list[7], None), f"{syn_tree.get_state()=}"
        )
        self.assertIsInstance(syn_tree.root, NodeChemical)
        self.assertEqual(syn_tree.root.smiles, smiles_list[7])
        self.assertIsInstance(syn_tree.to_dict(), dict)
        self.assertEqual(syn_tree.to_dict(), self.syntree_dict)

    def test_syntree_with_repeating_bblock(self) -> None:
        """Test creating a syntree with repeating bblock.

        This resulted in a bug with 3 root mols until the fixed in 5ff1218.

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

        """

        action_dict = {
            "add": 0,
            "expand": 1,
            "merge": 2,
            "end": 3,
        }
        syntree = SyntheticTree()
        self.assertEqual(syntree.depth, 0)
        self.assertEqual(syntree.root, None)
        self.assertEqual(syntree.get_state(), (None, None), f"{syntree.get_state()=}")
        # 1: add uni
        syntree.update(action_dict["add"], 0, "A", None, "B")
        self.assertEqual(syntree.get_state(), ("B", None), f"{syntree.get_state()=}")
        logger.debug(
            f"Iteration 1 | Syntree depth: {syntree.depth}, state: {syntree.get_state()}"
        )
        # 2: add uni
        syntree.update(action_dict["add"], 1, "C", None, "D")
        self.assertEqual(syntree.get_state(), ("D", "B"), f"{syntree.get_state()=}")
        logger.debug(
            f"Iteration 2 | Syntree depth: {syntree.depth}, state: {syntree.get_state()}"
        )
        # 3: expand bi
        syntree.update(action_dict["expand"], 2, "D", "E", "F")
        self.assertEqual(syntree.get_state(), ("F", "B"), f"{syntree.get_state()=}")
        logger.debug(
            f"Iteration 3 | Syntree depth: {syntree.depth}, state: {syntree.get_state()}"
        )
        # 4: merge
        syntree.update(action_dict["merge"], 3, "F", "B", "G")
        self.assertEqual(syntree.get_state(), ("G", None), f"{syntree.get_state()=}")
        logger.debug(
            f"Iteration 4 | Syntree depth: {syntree.depth}, state: {syntree.get_state()}"
        )
        # 5: add uni
        syntree.update(action_dict["add"], 1, "C", None, "D")
        self.assertEqual(syntree.get_state(), ("D", "G"), f"{syntree.get_state()=}")
        logger.debug(
            f"Iteration 5 | Syntree depth: {syntree.depth}, state: {syntree.get_state()}"
        )
        # 6: expand bi
        syntree.update(action_dict["expand"], 2, "D", "E", "F")
        self.assertEqual(syntree.get_state(), ("F", "G"), f"{syntree.get_state()=}")
        logger.debug(
            f"Iteration 6 | Syntree depth: {syntree.depth}, state: {syntree.get_state()}"
        )

    def test_syntree_state(self) -> None:
        """Test is using same small syntree as above."""
        smiles_list = [
            "CCOc1ccc(CCNC(=O)CCl)cc1OCC",
            "C#CCN1CCC(C(=O)O)CC1.Cl",
            "CCOc1ccc(CCNC(=O)CN2N=NC=C2CN2CCC(C(=O)O)CC2)cc1OCC",
            "C=C(C)C(=O)OCCN=C=O",
            "Cc1cc(C#N)ccc1NC1CC1",
            "C=C(C)C(=O)OCCNC(=O)N(c1ccc(C#N)cc1C)C1CC1",
            "C=C(C)C(=O)OCCNC(=O)N(c1ccc(C2=NNC(C3CCN(Cc4cnnn4CC(=O)NCCc4ccc(OCC)c(OCC)c4)CC3)=N2)cc1C)C1CC1",
        ]
        syntree = SyntheticTree()
        self.assertEqual(len(syntree.get_state()), 2)
        self.assertEqual(syntree.get_state(), (None, None))
        # 0: Add (bi)
        syntree.update(0, 12, smiles_list[0], smiles_list[1], smiles_list[2])
        self.assertEqual(len(syntree.get_state()), 2)
        self.assertEqual(syntree.get_state(), (smiles_list[2], None))

        # 1: Add (bi)
        syntree.update(0, 47, smiles_list[3], smiles_list[4], smiles_list[5])
        self.assertEqual(len(syntree.get_state()), 2)
        self.assertEqual(syntree.get_state(), (smiles_list[5], smiles_list[2]))

        # 2: Merge (bi)
        syntree.update(2, 15, smiles_list[5], smiles_list[2], smiles_list[6])
        self.assertEqual(len(syntree.get_state()), 2)
        self.assertEqual(syntree.get_state(), (smiles_list[6], None))
