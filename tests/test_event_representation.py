"""Neighbor checks must exclude all images from the query's training orbit."""

import numpy as np

from scripts.IMAGE.vae_pipeline.check_event_representation import nearest_other_orbit


def test_other_orbit_reference_excludes_temporally_adjacent_images():
    train = np.array([[0.0], [0.1], [2.0], [2.2]])
    orbits = np.array([1, 1, 2, 2])
    reference = nearest_other_orbit(train, train, orbits, orbits)
    np.testing.assert_allclose(reference, [2.0, 1.9, 1.9, 2.1])
    unseen = nearest_other_orbit(np.array([[0.2], [2.1]]), train, orbits)
    np.testing.assert_allclose(unseen, [0.1, 0.1])
