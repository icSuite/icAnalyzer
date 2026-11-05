"""Kp matching must use exact retained frame times, not nearby images."""

import numpy as np
import pytest

from scripts.IMAGE.vae_pipeline.audit_run_kp import matched_positions


def test_exact_source_frame_lookup_and_rejection_of_nearby_or_duplicate_times():
    source = np.array(["2000-05-24T10:24:43", "2000-05-24T10:26:43", "2000-05-24T10:28:43"], dtype="datetime64[ns]")
    np.testing.assert_array_equal(matched_positions(source, source[[0, 2]]), [0, 2])
    with pytest.raises(ValueError, match="exactly"):
        matched_positions(source, source[[0]] + np.timedelta64(1, "s"))
    with pytest.raises(ValueError, match="unique"):
        matched_positions(source[[0, 0, 2]], source[[0]])
