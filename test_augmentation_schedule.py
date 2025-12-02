import pandas as pd

from app import (
    AugmentationScheduleEntry,
    BatteryCohort,
    SimConfig,
    SimState,
    YearResult,
    apply_augmentation,
)


def _make_year_result(year: int) -> YearResult:
    return YearResult(
        year_index=year,
        expected_firm_mwh=0.0,
        delivered_firm_mwh=0.0,
        shortfall_mwh=0.0,
        breach_days=0,
        charge_mwh=0.0,
        discharge_mwh=0.0,
        pv_to_contract_mwh=0.0,
        bess_to_contract_mwh=0.0,
        avg_rte=0.0,
        eq_cycles=0.0,
        cum_cycles=0.0,
        soh_cycle=1.0,
        soh_calendar=1.0,
        soh_total=1.0,
        eoy_usable_mwh=0.0,
        eoy_power_mw=0.0,
        pv_curtailed_mwh=0.0,
        flags={},
    )


def _make_state(cfg: SimConfig) -> SimState:
    return SimState(
        pv_df=pd.DataFrame({"pv_mw": [0.0]}),
        cycle_df=pd.DataFrame(),
        cfg=cfg,
        current_power_mw=cfg.initial_power_mw,
        current_usable_mwh_bolref=cfg.initial_usable_mwh,
        initial_bol_energy_mwh=cfg.initial_usable_mwh,
        initial_bol_power_mw=cfg.initial_power_mw,
        cohorts=[BatteryCohort(energy_mwh_bol=cfg.initial_usable_mwh, start_year=0)],
    )


def test_manual_schedule_percent_of_bol_adds_expected_energy() -> None:
    cfg = SimConfig(
        years=1,
        initial_power_mw=10.0,
        initial_usable_mwh=40.0,
        augmentation="Manual",
        augmentation_schedule=[
            AugmentationScheduleEntry(year=1, basis="Percent of BOL energy", value=10.0)
        ],
    )
    state = _make_state(cfg)
    year_result = _make_year_result(1)

    add_power, add_energy = apply_augmentation(state, cfg, year_result, discharge_hours_per_day=4.0)

    assert add_energy == 4.0
    assert add_power == 1.0


def test_manual_schedule_takes_precedence_over_threshold_logic() -> None:
    cfg = SimConfig(
        years=1,
        initial_power_mw=10.0,
        initial_usable_mwh=40.0,
        augmentation="Threshold",
        aug_trigger_type="Capability",
        aug_threshold_margin=0.10,
        aug_topup_margin=0.10,
        augmentation_schedule=[
            AugmentationScheduleEntry(year=1, basis="Fixed energy (MWh)", value=8.0)
        ],
    )
    state = _make_state(cfg)
    year_result = _make_year_result(1)
    year_result.eoy_usable_mwh = 20.0
    year_result.eoy_power_mw = 10.0

    add_power, add_energy = apply_augmentation(state, cfg, year_result, discharge_hours_per_day=4.0)

    assert add_energy == 8.0
    assert add_power == 2.0
