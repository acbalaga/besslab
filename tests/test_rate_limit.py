import unittest
from contextlib import ExitStack
from unittest import mock

import streamlit as st

from utils import enforce_rate_limit


class EnforceRateLimitTests(unittest.TestCase):
    def setUp(self) -> None:
        # Provide an isolated session_state for each test run.
        self.session_state = {}
        self.error_messages = []
        self.info_messages = []
        self.stop_calls = 0

    def _patched_streamlit(self) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(mock.patch.object(st, "session_state", self.session_state))
        stack.enter_context(
            mock.patch("utils.rate_limit.st.error", side_effect=self.error_messages.append)
        )
        stack.enter_context(
            mock.patch("utils.rate_limit.st.info", side_effect=self.info_messages.append)
        )
        stack.enter_context(mock.patch("utils.rate_limit.st.stop", side_effect=self._increment_stop))
        return stack

    def _increment_stop(self, *args, **kwargs):
        self.stop_calls += 1
        raise RuntimeError("st.stop called")

    def test_prunes_outside_window_and_records_spacing(self) -> None:
        self.session_state.update({"recent_runs": [0.0, 50.0], "last_rate_limit_ts": 0.0})
        with self._patched_streamlit(), mock.patch(
            "utils.rate_limit.time.time", side_effect=[700.0, 700.0]
        ):
            enforce_rate_limit(max_runs=3, window_seconds=100, min_spacing_seconds=2.0)
            self.assertEqual(self.stop_calls, 0)
            self.assertEqual(len(self.session_state["recent_runs"]), 1)
            self.assertGreaterEqual(self.session_state["recent_runs"][0], 700.0)

    def test_stop_called_when_over_limit(self) -> None:
        self.session_state.update({"recent_runs": [1.0, 2.0, 3.0]})
        with self._patched_streamlit(), mock.patch(
            "utils.rate_limit.time.time", return_value=4.0
        ):
            with self.assertRaises(RuntimeError):
                enforce_rate_limit(max_runs=3, window_seconds=10, min_spacing_seconds=0.0)
        self.assertGreater(self.stop_calls, 0)
        self.assertGreaterEqual(len(self.error_messages), 1)
        self.assertGreaterEqual(len(self.info_messages), 1)

    def test_min_spacing_prevents_double_count(self) -> None:
        with self._patched_streamlit(), mock.patch(
            "utils.rate_limit.time.time", side_effect=[10.0, 11.0]
        ):
            enforce_rate_limit(max_runs=5, window_seconds=100, min_spacing_seconds=2.0)
            first_len = len(self.session_state["recent_runs"])
            enforce_rate_limit(max_runs=5, window_seconds=100, min_spacing_seconds=2.0)
            self.assertEqual(first_len, len(self.session_state["recent_runs"]))
            self.assertEqual(self.stop_calls, 0)


if __name__ == "__main__":
    unittest.main()
