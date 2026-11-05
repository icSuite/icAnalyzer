# Manual masking

This directory owns human review of IMAGE frames and masks:

- `annotate_partial_images.py` is the current interactive annotation entry.
- `annotation_store.py`, `annotation_geometry.py`, and `native_annotations.py`
  define storage, geometry, propagation, and native-product loading.
- `annotated_data.py` applies the saved reviews to downstream CVAE inputs.

Completed migration utilities, historical annotation predecessors, and retired
quality-plot scripts are preserved in Git history rather than the live tree.
