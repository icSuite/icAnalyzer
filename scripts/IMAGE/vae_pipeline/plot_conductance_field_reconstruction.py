"""Figures 5a/b: omit Pedersen or Hall, then reconstruct both using log covariance."""

from scripts.IMAGE.vae_pipeline.plot_conductance_reconstruction import parse_arguments, run


if __name__ == "__main__":
    run(parse_arguments(description=__doc__), case="fields")
