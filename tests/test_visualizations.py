import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import HourlyLog, prepare_soc_heatmap_data


def test_prepare_soc_heatmap_data_dimensions():
    hours = 48
    logs = HourlyLog(
        hod=np.tile(np.arange(24), 2),
        pv_mw=np.zeros(hours),
        pv_to_contract_mw=np.zeros(hours),
        bess_to_contract_mw=np.zeros(hours),
        charge_mw=np.zeros(hours),
        discharge_mw=np.zeros(hours),
        soc_mwh=np.linspace(0, 100, hours),
    )

    heatmap = prepare_soc_heatmap_data(logs, initial_energy_mwh=100.0)

    assert heatmap.shape == (2, 24)
    assert list(heatmap.columns) == list(pd.RangeIndex(0, 24, name="hour"))
    assert list(heatmap.index) == [1, 2]
    assert heatmap.loc[1, 0] == 0.0
    assert np.isclose(heatmap.loc[2, 0], logs.soc_mwh[24] / 100.0)
