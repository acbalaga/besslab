from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.ui.forms import validate_dispatch_windows  # noqa: E402


def test_validate_dispatch_windows_warns_on_zero_duplicate_overlap() -> None:
    result = validate_dispatch_windows(
        discharge_text="10:00-10:00, 08:00-12:00, 08:00-12:00, 22:00-02:00, 01:00-03:00",
        charge_text="05:00-05:00",
    )

    warning_text = "\n".join(result.warnings)

    assert "Discharge window '10:00-10:00'" in warning_text
    assert "Duplicate Discharge window '08:00-12:00'" in warning_text
    assert "Discharge windows '22:00-02:00' and '01:00-03:00' overlap" in warning_text
    assert "Charge window '05:00-05:00'" in warning_text
