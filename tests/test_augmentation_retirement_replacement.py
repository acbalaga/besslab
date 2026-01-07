import numpy as np
import pandas as pd

from services.simulation_core import SimConfig, Window, simulate_project


def test_retirement_replacement_adds_energy_on_retire() -> None:
    pv_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2022-01-01", periods=24 * 365, freq="h"),
            "pv_mw": np.zeros(24 * 365),
        }
    )

    cfg = SimConfig(
        years=1,
        initial_power_mw=5.0,
        initial_usable_mwh=20.0,
        contracted_mw=0.0,
        discharge_windows=[Window(0, 24)],
        pv_availability=1.0,
        bess_availability=1.0,
        rte_roundtrip=0.9,
        calendar_fade_rate=1.0,
        use_calendar_exp_model=True,
        aug_retire_old_cohort=True,
        aug_retire_soh_pct=0.6,
        aug_retire_replacement_mode="Percent",
        aug_retire_replacement_pct_bol=0.5,
    )

    output = simulate_project(cfg, pv_df=pv_df, cycle_df=pd.DataFrame(), dod_override="Auto (infer)", need_logs=False)

    assert output.augmentation_retired_energy_mwh == [20.0]
    assert output.augmentation_energy_added_mwh == [10.0]
    assert output.augmentation_events == 1
