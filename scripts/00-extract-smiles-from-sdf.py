"""Extract chemicals as SMILES from a downloaded `*.sdf*` file."""

# pylint: disable=invalid-name
# pylint: enable=invalid-name  # disable and enable to ignore the file name only.

import argparse
import json

from loguru import logger

from synnet.data_generation.preprocessing import parse_sdf_file


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
    print(df.shape)
    df.to_csv(output_file, index=False)


def get_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, help="An `*.sdf` file")
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file name for the resulting `pandas.DataFrame`.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logger.info("Start.")

    # Parse input args
    args = get_args()
    logger.info(f"Arguments: {json.dumps(vars(args), indent=2)}")

    logger.info("Start parsing SDF file...")
    extract_smiles(args.input_file, args.output_file)

    logger.info("Complete.")
