# Manual masking

This directory owns human review of IMAGE frames and masks:

- `annotate_partial_images.py` is the current interactive annotation entry.
- `annotation_store.py`, `annotation_geometry.py`, and `native_annotations.py`
  define self-contained storage, native-grid geometry, propagation, and
  product loading.
- `annotated_data.py` applies the saved reviews to downstream CVAE inputs.

The schema-3 JSON embeds each reviewed time axis and the 46x46 magnetic grid.
It contains no source-product paths and does not require the retired IMAGE_FUV
corpus after annotation.

Completed migration utilities, historical annotation predecessors, and retired
quality-plot scripts are preserved in Git history rather than the live tree.
