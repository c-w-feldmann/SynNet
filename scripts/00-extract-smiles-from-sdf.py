"""Extract chemicals as SMILES from a downloaded `*.sdf*` file.
"""

import argparse
import json
import logging

from synnet.data_generation.preprocessing import parse_sdf_file

logger = logging.getLogger(__name__)


def main(input_file: str, output_file: str) -> None:
    assert not input_file == output_file, "Input and output files must be different."
    df = parse_sdf_file(input_file)
    print(df.shape)
    df.to_csv(output_file, index=False)
    return None


def get_args() -> argparse.Namespace:
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
    logger.info(f"Arguments: {json.dumps(vars(args),indent=2)}")

    logger.info("Start parsing SDF file...")
    main(args.input_file, args.output_file)

    logger.info("Complete.")
