# Script organization

The maintained IMAGE analysis workflow is being separated by scientific stage:

- `manual_masking/`: the annotation GUI and its four maintained support
  modules.
- `automatic_masking/`: the adopted Model1 CNN and Model2 U-Net preparation,
  training, inference, and two-stage export workflows.
- `CVAE/`: supported Hall/Pedersen and WIC conditional-VAE preparation,
  training, fixed-Kp sampling, and WIC-result plotting.
- `IMAGE/vae_pipeline/`: older WIC/SI12 VAE workflows, diagnostics,
  reconstruction studies, and presentation scripts.

The remaining scripts at this level and directly under `IMAGE/` are historical
or not yet classified. Their presence does not make them maintained entry
points.
