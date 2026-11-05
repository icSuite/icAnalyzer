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

All generated datasets, checkpoints and masks belong outside the repository.
The complete supported workflow is:

```bash
SOURCE=~/IMAGE_FUV_new/precipitation_cs/image_apex_130km_46x46_v1/IR_hardy_smoothed_wic1_si121_si131
MASK_RUN=~/IMAGE_CVAE/automatic_masking/model1_model2_20261105

PYTHONPATH=. python scripts/automatic_masking/prepare_model2_training_data.py \
  --annotations data/partial_image_annotations_46x46.json \
  --source "$SOURCE" --output "$MASK_RUN/model2_training_data"

PYTHONPATH=. python scripts/automatic_masking/train_model2_unet.py \
  "$MASK_RUN/model2_training_data" --output "$MASK_RUN/model2_run" --device cuda

PYTHONPATH=. python scripts/automatic_masking/apply_model2_unet.py \
  --source "$SOURCE" --checkpoint "$MASK_RUN/model2_run/model2_unet.pt" \
  --output "$MASK_RUN/model2_masks" --device cuda

PYTHONPATH=. python scripts/automatic_masking/prepare_model1_training_data.py \
  --prepared-model2 "$MASK_RUN/model2_training_data" \
  --model2 "$MASK_RUN/model2_run/model2_unet.pt" \
  --output "$MASK_RUN/model1_training_data"

PYTHONPATH=. python scripts/automatic_masking/train_model1_cnn.py \
  --prepared "$MASK_RUN/model1_training_data" \
  --model2 "$MASK_RUN/model2_run/model2_unet.pt" \
  --output "$MASK_RUN/model1_run"

PYTHONPATH=. python scripts/automatic_masking/apply_two_stage_masks.py \
  --baseline "$MASK_RUN/model2_masks" --trial "$MASK_RUN/model1_run" \
  --output "$MASK_RUN/two_stage_masks"
```

The preparation step freezes the established held-out policy: validation days
are selected before `2000-07-09`, and all reviewed frames on or after that date
form the test split. The exact frame and orbit memberships are written into the
prepared dataset and reused unchanged by both models.
