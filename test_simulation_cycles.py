import numpy as np

from app import infer_dod_bucket


def test_infer_dod_bucket_scales_with_available_energy():
    daily_dis = np.array([50.0, 52.0, 48.0])

    # When the usable energy has faded to ~60 MWh, the median daily discharge
    # (~50 MWh) should map to an ~80% DoD bucket rather than dropping to 40%
    # because of a stale BOL reference.
    bucket_with_faded_energy = infer_dod_bucket(daily_dis, usable_mwh_available=60.0)
    bucket_with_bol_energy = infer_dod_bucket(daily_dis, usable_mwh_available=120.0)

    assert bucket_with_faded_energy == 80
    assert bucket_with_bol_energy == 40
