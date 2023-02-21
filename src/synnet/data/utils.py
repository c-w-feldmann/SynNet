import logging
from pathlib import Path
from typing import Iterable, Optional, Union

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, Subset

from synnet.config import NUM_RXN_TEMPLATES, SPLIT_RATIO
from synnet.data.datasets import (
    ActSyntreeDataset,
    RT1SyntreeDataset,
    RT2SyntreeDataset,
    RXNSyntreeDataset,
    SyntreeDataset,
)
from synnet.data_generation.syntrees import (
    IdentityIntEncoder,
    MorganFingerprintEncoder,
    OneHotEncoder,
)
from synnet.utils.datastructures import SyntheticTree

logger = logging.getLogger(__name__)
logging.info(f"Number of reaction templates set to {NUM_RXN_TEMPLATES}.")


def get_dataset(data: Union[str, Path, Iterable[SyntheticTree]]) -> SyntreeDataset:
    """Get a `SyntreeDataset`."""
    dataset = SyntreeDataset(dataset=data)
    logger.debug(f"Dataset size: {len(dataset)}")
    return dataset


def get_splits(
    dataset: SyntreeDataset, SPLIT_RATIO: tuple[int] = SPLIT_RATIO, **kwargs
) -> tuple[Subset]:
    """Split a dataset into train, valid, and test sets."""
    train, valid, test = torch.utils.data.random_split(
        dataset, SPLIT_RATIO, generator=torch.Generator().manual_seed(42)
    )
    logger.debug(f"  Train size: {len(train)}")
    logger.debug(f"  Valid size: {len(valid)}")
    logger.debug(f"  Test size: {len(test)}")
    return train, valid, test


def get_datasets_act(
    train: Subset,
    valid: Subset,
    test: Optional[Subset] = None,
    **kwargs,
) -> list[Optional[ActSyntreeDataset]]:
    """Get datasets for ACT model."""
    # Get hparams
    num_workers = kwargs.get("num_workers", 16)
    verbose = kwargs.get("verbose", False)

    dim_state = kwargs["embedding_state_nbits"]
    r_state = kwargs["embedding_state_radius"]

    # Model specific featurizers
    featurizer_state = MorganFingerprintEncoder(r_state, dim_state)

    logger.debug(f"ActSyntreeDataset: {featurizer_state=}")

    # Datasets
    datasets: list = []
    for data in (train, valid, test):
        if data is None:
            datasets += [None]
            continue
        _dataset = ActSyntreeDataset(
            (st for st in data),
            featurizer=featurizer_state,
            verbose=verbose,
            num_workers=num_workers,
        )
        datasets += [_dataset]

    return datasets


def get_datasets_rt1(
    train: Optional[Subset] = None,
    valid: Optional[Subset] = None,
    test: Optional[Subset] = None,
    **kwargs,
) -> list[Optional[RT1SyntreeDataset]]:
    """Get datasets for RT1 model."""
    # Get hparams
    num_workers = kwargs.get("num_workers", 16)
    verbose = kwargs.get("verbose", False)

    dim_state = kwargs["embedding_state_nbits"]
    r_state = kwargs["embedding_state_radius"]
    dim_rct = kwargs["embedding_rct_nbits"]
    r_rct = kwargs["embedding_state_radius"]

    # Model specific featurizers
    featurizer_state = MorganFingerprintEncoder(r_state, dim_state)
    featurizer_rct = MorganFingerprintEncoder(r_rct, dim_rct)

    logger.debug(f"RT1SyntreeDataset: {featurizer_state=}, {featurizer_rct=}")

    # Datasets
    datasets: list = []
    for data in (train, valid, test):
        if data is None:
            datasets += [None]
            continue
        _dataset = RT1SyntreeDataset(
            (st for st in data),
            featurizer=featurizer_state,
            reactant_1_featurizer=featurizer_rct,
            verbose=verbose,
            num_workers=num_workers,
        )
        datasets += [_dataset]

    return datasets


def get_datasets_rxn(
    train: Subset,
    valid: Subset,
    test: Optional[Subset] = None,
    **kwargs,
) -> list[Optional[RXNSyntreeDataset]]:
    """Get datasets for RXN model."""
    # Get hparams
    num_workers = kwargs.get("num_workers", 16)
    verbose = kwargs.get("verbose", False)

    dim_state = kwargs["embedding_state_nbits"]
    r_state = kwargs["embedding_state_radius"]

    # Model specific featurizers
    featurizer_state = MorganFingerprintEncoder(r_state, dim_state)
    if kwargs["embedding_rxn"] == "onehot":  # => classification problem
        featurizer_rxn = IdentityIntEncoder()
    elif kwargs["embedding_rxn"] == "rdkitfp":
        from synnet.data_generation.preprocessing import ReactionTemplateFileHandler
        from synnet.encoding.rxntemplates import RdkitRxnFPConfig, RXNFingerprintEncoder

        tmplts = ReactionTemplateFileHandler().load("data/assets/reaction-templates/hb.txt")
        rxn_map = {i: rxn for i, rxn in enumerate(tmplts)}
        rdkit_params = RdkitRxnFPConfig(fpSize=kwargs["embedding_rxn_nbits"])
        featurizer_rxn = RXNFingerprintEncoder(
            rxn_map=rxn_map,
            params=RdkitRxnFPConfig(fpSize=kwargs["embedding_rxn_nbits"]).params,
        )

    logger.debug(f"RXNSyntreeDataset: {featurizer_state=}, {featurizer_rxn=}")

    # Datasets
    datasets: list = []
    for data in (train, valid, test):
        if data is None:
            datasets += [None]
            continue
        _dataset = RXNSyntreeDataset(
            (st for st in data),
            featurizer=featurizer_state,
            rxn_featurizer=featurizer_rxn,
            verbose=verbose,
            num_workers=num_workers,
        )
        datasets += [_dataset]

    return datasets


def get_datasets_rt2(
    train: Subset,
    valid: Subset,
    test: Optional[Subset] = None,
    **kwargs,
) -> list[Optional[RT2SyntreeDataset]]:
    """Get datasets for RT2 model."""
    # Get hparams
    num_workers = kwargs.get("num_workers", 16)
    verbose = kwargs.get("verbose", False)

    dim_state = kwargs["embedding_state_nbits"]
    r_state = kwargs["embedding_state_radius"]
    dim_rct = kwargs["embedding_rct_nbits"]
    r_rct = kwargs["embedding_rct_radius"]

    # Model specific featurizers
    featurizer_state = MorganFingerprintEncoder(r_state, dim_state)
    featurizer_rct = MorganFingerprintEncoder(r_rct, dim_rct)
    if kwargs["embedding_rxn"] == "onehot":  # => INOF: must 1-hot encode here
        featurizer_rxn = OneHotEncoder(NUM_RXN_TEMPLATES)
    elif kwargs["embedding_rxn"] == "rdkit-fp-TODO:":
        raise NotImplementedError()
    elif kwargs["embedding_rxn"] == "rdkit-fp-TODO:":
        raise NotImplementedError()

    logger.debug(f"RT2SyntreeDataset: {featurizer_state=}, {featurizer_rxn=}, {featurizer_rct=}")

    # Datasets
    datasets: list = []
    for data in (train, valid, test):
        if data is None:
            datasets += [None]
            continue
        _dataset = RT2SyntreeDataset(
            (st for st in data),
            featurizer=featurizer_state,
            rxn_featurizer=featurizer_rxn,
            reactant_2_featurizer=featurizer_rct,
            verbose=verbose,
            num_workers=num_workers,
        )
        datasets += [_dataset]

    return datasets


def get_dataloaders(
    train_dataset: Optional[Dataset] = None,
    valid_dataset: Optional[Dataset] = None,
    test_dataset: Optional[Dataset] = None,
    **kwargs,
) -> list[DataLoader]:
    """Get dataloaders for train, valid, test datasets."""
    batch_size = kwargs["batch_size"]
    num_workers = kwargs.get("num_workers", 16)
    assert batch_size >= 32, "pls use a batch size >= 32"
    logger.debug(f"Creating dataloaders with: {batch_size=}, {num_workers=}")

    # Get dataloaders
    dataloaders: list = []
    for dataset, shuffle in zip((train_dataset, valid_dataset, test_dataset), (True, False, False)):
        if dataset is None:
            dataloaders += [None]
            continue
        _dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        dataloaders += [_dataloader]

    return dataloaders
