from __future__ import annotations

from api import server
from api.server import (
    BatchRequest,
    BatchRun,
    PVRow,
    SimulationRequest,
    SimConfigPayload,
    SweepRequest,
    UploadPayload,
    create_upload,
    simulate,
    sweep,
)


def test_simulate_with_sample_inputs() -> None:
    request = SimulationRequest()
    response = simulate(request)

    assert "summary" in response
    assert "output" in response
    assert response["output"]["results"], "Expected year-level results"
    assert any("sample" in msg for msg in response["warnings"])


def test_sweep_runs_one_candidate() -> None:
    sweep_request = SweepRequest(power_values=[10.0], duration_values=[4.0])
    response = sweep(sweep_request)

    assert response["rows"], "Expected at least one sweep row"
    assert response["rows"][0]["power_mw"] == 10.0


def test_upload_and_batch_reuse_cached_data() -> None:
    cycle_payload = UploadPayload(
        kind="cycle",
        name="unit-test-cycle",
        cycle_rows=server._sample_cycle().head(3).to_dict("records"),
    )
    cycle_upload_id = create_upload(cycle_payload)["upload_id"]

    batch_request = BatchRequest(
        data=server.DataSource(
            use_sample_pv=True,
            use_sample_cycle=False,
            cycle_upload_id=cycle_upload_id,
        ),
        runs=[BatchRun(name="cached", config=SimConfigPayload(contracted_mw=1.0, initial_power_mw=1.0))],
    )
    batch_response = server.batch(batch_request)

    assert batch_response["runs"], "Expected at least one batch run"
    assert batch_response["runs"][0]["summary"]["total_project_generation_mwh"] >= 0.0
