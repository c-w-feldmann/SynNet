"""Module for molecular embeddings."""

import abc
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from loguru import logger
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.neighbors import BallTree

try:
    from pathos import multiprocessing as mp
except ImportError:
    logger.warning("Could not import pathos, using multiprocessing instead.")
    import multiprocessing as mp

from synnet.config import MAX_PROCESSES
from synnet.utils.custom_types import MetricType, PathType


class MolecularEmbedder(abc.ABC):
    """Base class for molecular embedding."""

    _length: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a new instance from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary with the class name and module name as keys.

        Returns
        -------
        Self
            New instance of the class.
        """
        module_name = data.pop("module")
        class_name = data.pop("class")
        if module_name != cls.__module__:
            raise ValueError(
                f"Cannot create {cls.__name__} from {module_name}.{class_name}"
            )
        mod = __import__(module_name, fromlist=[class_name])
        klass = getattr(mod, class_name)

        return klass(**data)

    @property
    def length(self) -> int:
        """Return the length of the embedding."""
        return self._length

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the embedding.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of the embedding.
        """
        return {
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
        }

    @abc.abstractmethod
    def transform(self, mol: Chem.Mol) -> npt.NDArray[Any]:
        """Return a molecular embedding.

        Parameters
        ----------
        mol : Chem.Mol
            RDKit molecule.

        Returns
        -------
        npt.NDArray[Any]
            Molecular embedding.
        """

    def transform_smiles(self, smiles: str) -> npt.NDArray[Any]:
        """Return a molecular embedding.

        Parameters
        ----------
        smiles : str
            SMILES string of molecule.

        Returns
        -------
        npt.NDArray[Any]
            Molecular embedding.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES string: {smiles}")
        return self.transform(mol)


class MorganFingerprintEmbedding(MolecularEmbedder):
    """Morgan fingerprint embedding."""

    def __init__(self, radius: int = 2, n_bits: int = 4096) -> None:
        """Initialize the Morgan fingerprint embedding.

        Parameters
        ----------
        radius : int
            Radius of the fingerprint.
        n_bits : int
            Length of the fingerprint.
        """
        self.radius = radius
        self._length = n_bits

    def transform(self, mol: Chem.Mol) -> npt.NDArray[np.bool_]:
        """Transform a RDKit molecule into a Morgan fingerprint.

        Parameters
        ----------
        mol : Chem.Mol
            RDKit molecule.

        Returns
        -------
        npt.NDArray[np.bool_]
            Morgan fingerprint.
        """
        fingerprint = AllChem.GetMorganGenerator(radius=self.radius, fpSize=self.length)
        return fingerprint.GetFingerprintAsNumPy(mol).astype(bool)

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the embedding.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of the embedding.
        """
        data = super().to_dict()
        data.update({"radius": self.radius, "n_bits": self.length})
        return data


class MolecularEmbeddingManager:
    """Class for computing and storing molecular embeddings."""

    building_blocks: Union[list[str], npt.NDArray[np.int_]]
    embeddings: npt.NDArray[np.int_]
    embedding_method: MolecularEmbedder
    kdtree: BallTree
    kdtree_metric: MetricType
    n_jobs: int

    def __init__(
        self,
        smiles_list: Optional[Iterable[str]] = None,
        embedding_method: Optional[MolecularEmbedder] = None,
        precalculated_embeddings: Optional[npt.NDArray[np.int_]] = None,
        kdtree_metric: MetricType = "euclidean",
        n_jobs: int = MAX_PROCESSES,
    ) -> None:
        """Initialize MolecularEmbeddingManager.

        This object is ment for managing the embeddings of a set of molecules.

        Parameters
        ----------
        smiles_list: Iterable[str]
            List of SMILES strings.
        embedding_method: Optional[MolecularEmbedder]
            Embedding method to use. If None, MorganFingerprintEmbedding is used.
        precalculated_embeddings: Optional[npt.NDArray[np.int_]]
            Precalculated embeddings. If None, they are computed.
        kdtree_metric: str
            Metric used for the kdtree. Default is cosine.
        n_jobs: int
            Number of cores used for computing the embeddings.

        Raises
        ------
        ValueError
            If the number of SMILES strings and embeddings do not match.
        """
        if smiles_list is None:
            smiles_list = []
        self._smiles_idx_df = pd.DataFrame()
        self._smiles_idx_df["smiles"] = smiles_list
        self._smiles_idx_df.set_index("smiles", inplace=True)
        self._smiles_idx_df["index"] = np.arange(len(self._smiles_idx_df))
        self.n_jobs = n_jobs
        if embedding_method is None:
            embedding_method = MorganFingerprintEmbedding(2, 256)
        self.embedding_method = embedding_method

        # Compute embeddings if no precalculated values are given.
        if precalculated_embeddings is None:
            if self.smiles_array.shape[0] > 0:
                if self.n_jobs == 1:
                    array_list = [
                        self.embedding_method.transform_smiles(smi)
                        for smi in smiles_list
                    ]
                else:
                    with mp.Pool(self.n_jobs) as pool:
                        array_list = pool.map(
                            self.embedding_method.transform_smiles, smiles_list
                        )
                precalculated_embeddings = np.vstack(array_list)
            else:
                precalculated_embeddings = np.array([])

        if precalculated_embeddings.shape[0] != self.smiles_array.shape[0]:
            raise ValueError(
                f"Number of SMILES strings ({len(self.smiles_array)}) and "
                f"embeddings ({len(precalculated_embeddings)}) do not match."
            )

        self.embeddings = precalculated_embeddings
        self.kdtree_metric = kdtree_metric
        if self.embeddings.shape[0] > 0:
            self.kdtree = BallTree(self.get_embeddings(), metric=kdtree_metric)
        else:
            self.kdtree = None

    @classmethod
    def from_files(
        cls,
        configuration_file: PathType,
        compound_list_file: PathType,
        precalculated_embedding_file: Optional[PathType] = None,
    ) -> Self:
        """Create a new instance from a file.

        Parameters
        ----------
        configuration_file : PathType
            File with the configuration.
        compound_list_file : PathType
            File with the list of compounds.
        precalculated_embedding_file : Optional[PathType]
            File with the precalculated embeddings.

        Returns
        -------
        Self
            Returns the instance.
        """
        compound_list = np.loadtxt(compound_list_file, dtype=str, comments=None)
        with open(configuration_file, "r", encoding="UTF-8") as f:
            config = yaml.safe_load(f)

        if precalculated_embedding_file:
            precalculated_embeddings = np.load(precalculated_embedding_file)
        else:
            precalculated_embeddings = None
        embedding_method = MolecularEmbedder.from_dict(config.pop("embedding_method"))
        kd_tree_metric = config.pop("kdtree_metric")
        if isinstance(kd_tree_metric, str):
            kd_tree_metric = kd_tree_metric.lower()
        elif isinstance(kd_tree_metric, dict):
            module = kd_tree_metric["module"]
            name = kd_tree_metric["name"]
            kd_tree_metric = getattr(__import__(module, fromlist=[name]), name)
        else:
            raise ValueError(f"Invalid type for kdtree_metric: {type(kd_tree_metric)}")
        return cls(
            compound_list,
            embedding_method=embedding_method,
            precalculated_embeddings=precalculated_embeddings,
            kdtree_metric=kd_tree_metric,
            **config,
        )

    @classmethod
    def from_folder(cls, folder: PathType) -> Self:
        """Initialize from a folder.

        Parameters
        ----------
        folder : PathType
            Folder with the files.

        Returns
        -------
        Self
            Returns the instance.
        """
        folder = Path(folder)
        return cls.from_files(
            configuration_file=folder / "configuration.yaml",
            compound_list_file=folder / "building_blocks.txt",
            precalculated_embedding_file=folder / "embeddings.npy",
        )

    @property
    def smiles_array(self) -> npt.NDArray[np.str_]:
        """Return the SMILES array."""
        return self._smiles_idx_df.index.values

    def get_embeddings(self) -> npt.NDArray[np.int_]:
        """Return `self.embeddings` as 2d-array.

        Returns
        -------
        npt.NDArray[np.int_]
            Embeddings as 2d-array.
        """
        return np.atleast_2d(self.embeddings)

    def get_embedding_for(self, smiles_list: list[str]) -> npt.NDArray[np.int_]:
        """Return the embeddings for the given SMILES strings.

        Parameters
        ----------
        smiles_list : list[str]
            List of SMILES strings.

        Returns
        -------
        npt.NDArray[np.int_]
            Embeddings for the given SMILES strings.
        """
        positions: npt.NDArray[np.int_]
        smiles_array = np.atleast_1d(smiles_list)
        positions = self._smiles_idx_df.loc[smiles_array, "index"].values.astype(int)
        return self.embeddings[positions]

    def _compute_mp(self, data: list[str]) -> list[npt.NDArray[Any]]:
        """Compute embeddings in parallel.

        Parameters
        ----------
        data : list[str]
            List of SMILES strings.

        Returns
        -------
        list[npt.NDArray[Any]]
            List of embeddings.
        """
        with mp.Pool(processes=self.n_jobs) as pool:
            embeddings = pool.map(self.embedding_method.transform_smiles, data)
        return embeddings

    def compute_embeddings(self, building_blocks: list[str]) -> Self:
        """Compute embeddings for a list of building blocks.

        Parameters
        ----------
        building_blocks: list[str]
            List of SMILES strings.

        Returns
        -------
        Self
            Returns the instance with the computed embeddings.
        """
        logger.info(f"Will compute embedding with {self.n_jobs} processes.")
        if self.n_jobs == 1:
            embeddings = list(
                map(self.embedding_method.transform_smiles, building_blocks)
            )
        else:
            embeddings = self._compute_mp(building_blocks)
        logger.info("Finished computing embeddings.")
        embedding_matrix = np.vstack(embeddings)
        self.embeddings = embedding_matrix
        self.kdtree = BallTree(self.get_embeddings(), metric=self.kdtree_metric)
        return self

    def _save_npy(self, file: str) -> Self:
        """Save the embeddings to a file.

        Parameters
        ----------
        file : str
            File to save the embeddings.

        Returns
        -------
        Self
            Returns the instance.
        """
        if self.embeddings is None:
            raise ValueError("Must have computed embeddings to save.")

        embeddings = np.atleast_2d(self.embeddings)  # (n,d)
        np.save(file, embeddings)
        logger.info(f"Successfully saved data (shape={embeddings.shape}) to {file} .")
        return self

    def init_balltree(
        self,
        metric: Union[Callable[[npt.NDArray[Any], npt.NDArray[Any]], np.float64], str],
    ) -> Self:
        """Initialize the BallTree.

        Notes
        -----
        Can take a couple of minutes.

        Parameters
        ----------
        metric : Union[Callable[[npt.NDArray[Any], npt.NDArray[Any]], np.float64], str]
            Metric to use for the kdtree.

        Returns
        -------
        Self
            Returns the instance with the kdtree initialized.
        """
        if self.embeddings is None:
            raise ValueError("Need emebddings to compute kdtree.")
        self.kdtree_metric = metric.__name__ if not isinstance(metric, str) else metric
        self.kdtree = BallTree(self.embeddings, metric=metric)

        return self

    def to_files(
        self,
        configuration_file: PathType,
        compound_list_file: PathType,
        precalculated_embedding_file: Optional[PathType] = None,
    ) -> None:
        """Save the instance to files.

        Parameters
        ----------
        configuration_file : PathType
            File to save the configuration.
        compound_list_file : PathType
            File to save the list of compounds.
        precalculated_embedding_file : Optional[PathType]
            File to save the precalculated embeddings.
        """
        np.savetxt(compound_list_file, self.smiles_array, fmt="%s")
        config = {
            "embedding_method": self.embedding_method.to_dict(),
            "n_jobs": self.n_jobs,
            "kdtree_metric": self.kdtree_metric,
        }
        if isinstance(self.kdtree_metric, str):
            config["kdtree_metric"] = self.kdtree_metric
        elif callable(self.kdtree_metric):
            config["kdtree_metric"] = {
                "module": self.kdtree_metric.__module__,
                "name": self.kdtree_metric.__name__,
            }
        else:
            raise TypeError(
                f"Invalid type for kdtree_metric: {type(self.kdtree_metric)}"
            )

        with open(configuration_file, "w", encoding="UTF-8") as f:
            yaml.dump(config, f)
        if precalculated_embedding_file is not None:
            np.save(precalculated_embedding_file, self.embeddings)

    def to_folder(self, folder: PathType) -> Self:
        """Save the instance to a folder.

        Parameters
        ----------
        folder : PathType
            Folder to save the files

        Returns
        -------
        Self
            Returns the instance.
        """
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.to_files(
            configuration_file=folder / "configuration.yaml",
            compound_list_file=folder / "building_blocks.txt",
            precalculated_embedding_file=folder / "embeddings.npy",
        )
        return self
