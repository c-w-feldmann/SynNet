"""Main client to run the synnet package from commandline."""

from pathlib import Path

import click
from loguru import logger

from synnet.config import MAX_PROCESSES
from synnet.data_generation.preprocessing import (
    BuildingBlockFileHandler,
    BuildingBlockFilterHeuristics,
    BuildingBlockFilterMatchRxn,
    ReactionTemplateFileHandler,
    parse_sdf_file,
)
from synnet.encoding.embedding import (
    MolecularEmbeddingManager,
    MorganFingerprintEmbedding,
)
from synnet.utils.data_utils import ReactionSet


@click.group()
def synnet() -> None:
    """Main command for the synnet package. Requires subcommands."""


@synnet.command()
@click.argument("input_file", type=str, nargs=1)
@click.argument(
    "output_file",
    type=str,
    nargs=1,
)
def extract_smiles(input_file: str, output_file: str) -> None:
    """Extract chemicals as SMILES from a downloaded `*.sdf*` file.

    Parameters
    ----------
    input_file : str
        An `*.sdf` file
    output_file : str
        Output file name for the resulting `pandas.DataFrame`.
    """
    if input_file == output_file:
        raise ValueError("Input and output files must be different.")
    df = parse_sdf_file(input_file)
    df.to_csv(output_file, index=False)


@synnet.command()
@click.argument("building_blocks_file", type=str, nargs=1)
@click.argument("rxn_templates_file", type=str, nargs=1)
@click.argument("output_bblock_file", type=str, nargs=1)
@click.argument("output_rxns_collection_file", type=str, nargs=1)
@click.option("--ncpu", type=int, default=4, help="Number of cpus")
@click.option("--verbose", default=False, is_flag=True)
def filter_building_blocks(
    building_blocks_file: str,
    rxn_templates_file: str,
    output_bblock_file: str,
    output_rxns_collection_file: str,
    ncpu: int,
    verbose: bool,
) -> None:
    """Filter building blocks based to remove those that cannot react with any template.

    Also filters reactions for which no reactants are available.

    Parameters
    ----------
    building_blocks_file : str
        Input file containing building blocks.
    rxn_templates_file : str
        Input file containing reaction templates.
    output_bblock_file : str
        Output file for filtered building blocks.
    output_rxns_collection_file : str
        Output file for filtered reactions.
    ncpu : int
        Number of cpus to use.
    verbose : bool
        Whether to print verbose output.
    """
    logger.info("Start.")
    print(Path(".").resolve())
    # 1. Load assets
    bblocks = BuildingBlockFileHandler().load(building_blocks_file)
    rxn_templates = ReactionTemplateFileHandler().load(rxn_templates_file)

    # 2. Filter
    #   building blocks on heuristics
    filtered_bblocks = BuildingBlockFilterHeuristics.filter(bblocks, verbose=verbose)

    #   building blocks that cannot react with any template
    filtered_bblocks_list, reactions = BuildingBlockFilterMatchRxn().filter(
        filtered_bblocks, rxn_templates, ncpu=ncpu, verbose=verbose
    )

    # 3. Save
    #   filtered building blocks
    BuildingBlockFileHandler().save(output_bblock_file, filtered_bblocks_list)

    #   initialized reactions (these are initialized with available reactants)
    ReactionSet(reactions).save(
        output_rxns_collection_file, skip_without_building_block=False
    )
    logger.info("Completed.")


@synnet.command()
@click.argument("building_blocks_file", type=str, nargs=1)
@click.argument("output_folder", type=str, nargs=1)
@click.argument("featurization_fct", type=str, nargs=1)
@click.option("--ncpu", type=int, default=MAX_PROCESSES, help="Number of cpus")
def compute_embeddings(
    building_blocks_file: str,
    output_folder: str,
    featurization_fct: str,
    n_jobs: int = MAX_PROCESSES,
) -> None:
    """Compute embeddings for a list of building blocks.

    Parameters
    ----------
    building_blocks_file : str
        File containing building blocks.
    output_folder : str
        Folder to save the embeddings.
    featurization_fct : str
        Name of the featurization function to use.
    n_jobs : int
        Number of cpus to use.
    """
    functions: dict[str, MorganFingerprintEmbedding] = {
        "fp_4096": MorganFingerprintEmbedding(radius=2, n_bits=4096),
        "fp_2048": MorganFingerprintEmbedding(radius=2, n_bits=2048),
        "fp_1024": MorganFingerprintEmbedding(radius=2, n_bits=1024),
        "fp_512": MorganFingerprintEmbedding(radius=2, n_bits=512),
        "fp_256": MorganFingerprintEmbedding(radius=2, n_bits=256),
    }
    logger.info("Start.")

    # Load building blocks
    bblocks = BuildingBlockFileHandler().load(building_blocks_file)
    logger.info(f"Successfully read {building_blocks_file}.")
    logger.info(f"Total number of building blocks: {len(bblocks)}.")

    # Compute embeddings
    embedding_method: MorganFingerprintEmbedding = functions[featurization_fct]

    mol_embedder = MolecularEmbeddingManager(
        smiles_list=bblocks,
        embedding_method=embedding_method,
        n_jobs=n_jobs,
    )

    # Save?
    mol_embedder.to_folder(output_folder)
    logger.info("Completed.")


if __name__ == "__main__":
    synnet()
