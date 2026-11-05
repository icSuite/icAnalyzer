"""Train the two-stage-mask CVAE; edit settings below or pass CLI overrides."""

#%% Imports and repository paths

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.CVAE.train_automatic_cvae import parse_arguments, run

CACHE = ROOT / "data/training_data/two_stage_cvae_valid5_20261015"
OUTPUT = ROOT / "figures/cvae_runs/conductance_two_stage_valid5_20261015"

#%% Same settings as the previous user-trained CVAE

EPOCHS = 500
BATCH_SIZE = 264
BETA = 0.048
LATENT_DIM = 36
SPLINE_CONTROL_POINTS = 30
SPLINE_DEGREE = 3
LEARNING_RATE = 0.0001
SEED = 1
DEVICE = "cuda"  # Use "cpu" explicitly if desired; CUDA is checked before writing.

#%% Optional spatial coverage weighting; enabling selects a separate output

COVERAGE_WEIGHTING = False
COVERAGE_BALANCE = "full"  # "full" removes the cap; "capped" uses the limit below.
COVERAGE_SIGMA = 1.5  # Gaussian width in grid pixels, applied before inversion.
COVERAGE_MAX_BOOST = 3.0
COVERAGE_KP_EDGES = [0., 9.]  # One shared map; e.g. [0, 2, 4, 6, 9] for broad Kp bins.


#%% Train; existing datasets, architecture and loss are reused

def main(argv=None):
    overrides = sys.argv[1:] if argv is None else argv
    coverage_options = ["--coverage-weighting"] if COVERAGE_WEIGHTING else []
    arguments = parse_arguments([
        str(CACHE), "--output", str(OUTPUT),
        "--epochs", str(EPOCHS), "--batch-size", str(BATCH_SIZE), "--beta", str(BETA),
        "--latent-dim", str(LATENT_DIM), "--spline-control-points", str(SPLINE_CONTROL_POINTS),
        "--spline-degree", str(SPLINE_DEGREE), "--learning-rate", str(LEARNING_RATE),
        "--seed", str(SEED), "--training-holes", "--device", DEVICE,
        "--coverage-balance", COVERAGE_BALANCE, "--coverage-sigma", str(COVERAGE_SIGMA), "--coverage-max-boost", str(COVERAGE_MAX_BOOST),
        "--coverage-kp-edges", *[str(edge) for edge in COVERAGE_KP_EDGES], *coverage_options,
        *overrides
    ])
    explicit_output = any(value == "--output" or value.startswith("--output=") for value in overrides)
    if arguments.coverage_weighting and not explicit_output:
        suffix = "_coverage_full" if arguments.coverage_balance == "full" else "_coverage"
        arguments.output = OUTPUT.with_name(OUTPUT.name + suffix)
    return run(arguments)


if __name__ == "__main__":
    main()
