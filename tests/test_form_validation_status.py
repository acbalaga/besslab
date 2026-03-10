from frontend.ui.forms import _build_validation_status_message


def test_build_validation_status_message_returns_none_when_clean() -> None:
    assert _build_validation_status_message([], []) is None


def test_build_validation_status_message_prioritizes_errors() -> None:
    status = _build_validation_status_message(["missing profile"], ["window overlap"])

    assert status is not None
    kind, message = status
    assert kind == "error"
    assert "1 error" in message
    assert "1 warning" in message


def test_build_validation_status_message_warns_when_only_warnings() -> None:
    status = _build_validation_status_message([], ["window overlap", "duplicate window"])

    assert status is not None
    kind, message = status
    assert kind == "warning"
    assert "2 warnings" in message
