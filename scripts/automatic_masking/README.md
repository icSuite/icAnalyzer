# Automatic masking

This directory owns the two automatic-mask models: the Model2 pixel U-Net and
the single-frame Model1 whole-frame acceptance CNN.

Current production-facing entries are:

- `prepare_model2_training_data.py`: turn manual reviews into Model2 targets.
- `model2_unet.py`: reusable U-Net architecture, losses and input helpers.
- `topology.py`: the adopted topology-weight training term.
- `train_model2_unet.py`: train the current Model2 U-Net.
- `apply_model2_unet.py`: apply the frozen Model2 U-Net.
- `prepare_model1_training_data.py`: freeze Model1 inputs and orbit splits.
- `model1_cnn.py`: reusable acceptance CNN, training and routing helpers.
- `train_model1_cnn.py`: train Model1 and freeze its acceptance cutoff.
- `apply_two_stage_masks.py`: apply frozen Model1 decisions with Model2
  fallback.

Historical CNNs and the rejected random-forest, SZA, line-mask,
partial-convolution, temporal-ablation, Bad-veto and comparison workflows are
preserved in Git history rather than the live tree.
