"""Figures 3a--c: Hall/Pedersen anchor correlations and joint explained variance."""

#%% Shared full-ensemble calculation, with conductance fields and provenance

import json

from scripts.IMAGE.vae_pipeline.plot_conductance_reconstruction import validate_conductance_propagation
from scripts.IMAGE.vae_pipeline import plot_wic_si12_teleconnections as teleconnections


def run(arguments):
    run_directory = arguments.run_directory.expanduser()
    output = arguments.output.expanduser()
    names = [f"figure_3{letter}.png" for letter in "abc"] + ["figure_3_teleconnections.npz", "figure_3_teleconnections.json"]
    if not arguments.overwrite and any((output / name).exists() for name in names):
        raise FileExistsError("conductance figure 3 already exists; use --overwrite or choose another output directory")
    if arguments.dpi <= 0:
        raise ValueError("DPI must be positive")
    native = validate_conductance_propagation(run_directory)
    result = teleconnections.run(arguments, field_prefix="conductance", field_names=("Hall", "Pedersen"))
    report = {
        "run_directory": str(run_directory.resolve()),
        "field_names": ["Hall", "Pedersen"],
        "grid_id": arguments.grid_id,
        "anchor_row_column": result["anchor_row_column"].tolist(),
        "sample_count": int(result["sample_count"]),
        "state_space": "physical Hall/Pedersen conductance in S; not log-space covariance",
        "figure_3a": "Pearson correlation of Hall/Pedersen at each cell with Hall at the anchor; [-1,1]",
        "figure_3b": "Pearson correlation of Hall/Pedersen at each cell with Pedersen at the anchor; [-1,1]",
        "figure_3c": "fraction of each cell's variance explained jointly by Hall and Pedersen at the anchor; R^2 in [0,1], unsigned",
        "anchor_correlation": result["anchor_correlation"].tolist(),
        "reference_kind": "native conductance_cs VAE ensemble" if native else "fixed-2-keV WIC-derived conductance ensemble; Hall/Pedersen are proportional",
        "support_policy": "saved training support excluded and unsupported anchors rejected; zero-variance cells blank" if "conductance_supported" in result else "zero-variance cells blank; anchors must have positive variance in both fields",
        "measurement_uncertainty_used": False,
        "interpretation": "full stored ensemble statistical relationships, not measured-error covariance, data assimilation or calibrated predictive skill"
    }
    (output / "figure_3_teleconnections.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


#%% Same zero-based flattened spatial ID as the WIC/SI12 version

def parse_arguments(argv=None):
    parser = teleconnections.argument_parser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="replace only the figure3 PNG/NPZ/JSON outputs; leave all other files untouched")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_arguments())
