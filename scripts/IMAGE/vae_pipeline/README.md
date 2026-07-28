# Partial-image annotation

`annotate_partial_images.py` is a resumable graphical annotation tool for the
IMAGE conductance data. It avoids per-frame pixel labeling by recording:

- one usable interval per orbit;
- an optional fully good interval inside it;
- a few masks drawn on representative partial frames.

Masks can be drawn either by lassoing the **valid area to retain** or by
dragging a straight divider across the image and choosing which half is valid.
A mask becomes active at its keyframe and is held unchanged for all later
frames until another keyframe replaces it. There is no temporal interpolation.

The tool only reads `data/good_orbit_segments.csv`. It does not alter that
file. New annotations and the current cursor are saved separately in
`data/partial_image_annotations.json`.

## Run

Use the environment that provides `icreader`:

```bash
mamba activate tConductance
python scripts/IMAGE/vae_pipeline/annotate_partial_images.py
```

The default external data locations match the current project layout. They can
be overridden with `--data-dir` and `--overview-dir`.

To start or resume at a particular orbit:

```bash
python scripts/IMAGE/vae_pipeline/annotate_partial_images.py \
  --start-orbit 169 \
  --start-frame 72
```

To try the interface without writing to the main annotation file:

```bash
python scripts/IMAGE/vae_pipeline/annotate_partial_images.py \
  --start-orbit 169 \
  --start-frame 72 \
  --annotations /tmp/icAnalyzer_annotation_trial.json
```

Closing the window, pressing `Q` or Escape, or clicking **Stop and save**
records the current orbit, page, and frame. The next ordinary launch resumes
there automatically. Boundary changes, mask changes, navigation, and
selection are also saved immediately using atomic file replacement.

The `/tmp` path above is suitable only for trying the interface; an operating
system cleanup or reboot may remove it. The default
`data/partial_image_annotations.json` is the durable project location. To
resume a temporary trial at its saved cursor, omit the start overrides:

```bash
python scripts/IMAGE/vae_pipeline/annotate_partial_images.py \
  --annotations /tmp/icAnalyzer_annotation_trial.json
```

## Orbit workflow

The four boundaries describe this ordering:

```text
bad | partial | fully good | partial | bad
      ^          ^      ^             ^
   usable     full    full          usable
    start     start   stop           stop
```

Either partial interval may be absent, and the fully good core may be cleared.
The normal workflow is:

1. Set only the outer boundary that needs extending. For partial images after
   an existing fully good sequence, select the last partially useful image and
   click **Set LAST usable**. Do not change **FIRST usable**.
2. Double-click an orange partial frame, or select it and click **Edit selected
   mask**.
3. At the first orange frame, either lasso the valid region or draw a dividing
   line and flip its assignment if necessary. Save the initial keyframe. That
   mask remains active until it is replaced.
4. Add another keyframe only where the shape of the bad region changes enough
   to need it.
5. Click **Mark orbit complete**. Each partial interval must have its initial
   mask keyframe on its first frame.

For orbit 169, the existing `0:51` good sequence seeds all four boundaries.
Here, **FIRST usable** and **FIRST fully good** are correctly fixed at frame
zero. To retain later partially useful images, select the last useful frame
and click **Set LAST usable**. Frames 52 through that selected frame will turn
orange; add sparse masks within that new orange interval.
The first keyframe must be at frame 52. Its mask remains valid through the end
of the interval unless a later keyframe explicitly replaces it.

If an orbit contains no useful data, **No usable data** requires two clicks
because it clears that orbit's draft boundaries and keyframes. **Undo last
edit** restores the preceding annotation state for the current orbit.

## Mask editor

Use **Lasso valid region** for irregular boundaries. Everything inside the
lasso is valid.

For the common half-image case:

1. use the default **Draw dividing line** mode;
2. drag the boundary across any image panel, or click its two endpoints;
3. inspect the translucent red excluded half in all four panels;
4. click **Flip last line** when the wrong side of the newest line is red;
5. draw another line when another edge needs excluding;
6. use **Undo last line** to remove the newest cut, if necessary;
7. click **Save keyframe**.

Each new line is an additional cut: the final valid region is the intersection
of the valid sides of all lines. Older cuts are white and the newest cut is
cyan. **Clear line cuts** returns the draft to an all-valid image.

All line endpoints, side assignments, and the resulting 36x36 mask are stored.
Reopening that exact keyframe restores the complete line set and mask.
Intersecting half-planes produce convex valid regions; retain lasso mode for a
strongly concave or disconnected valid region.

## Display

- green border: fully good;
- orange border: partial;
- red border: outside the usable interval;
- gray border: boundaries are not yet sufficient to classify the frame;
- blue border: selected frame;
- star in the frame title: exact mask keyframe;
- translucent red pixels: excluded by the propagated validity mask.

The channel button cycles between WIC, Hall conductance, Pedersen conductance,
and the existing weight array. Arrow keys navigate pages and orbits; `E` opens
the mask editor for the selected frame.

This stage records annotations only. Rasterizing the masks together with
finite-value checks and measurement weights belongs in the processed-data
builder, so the policy remains explicit and reproducible.
