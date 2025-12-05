import io
import unittest

import pandas as pd

from utils.io import read_pv_profile


class ReadPvProfileTests(unittest.TestCase):
    def _make_csv_buffer(self, df: pd.DataFrame) -> io.StringIO:
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return buf

    def test_valid_profile_one_based_hours_fill_gaps(self) -> None:
        df = pd.DataFrame(
            {
                "hour_index": [1, 2, 3],
                "pv_mw": [0.5, 1.0, 1.5],
            }
        )
        csv_buf = self._make_csv_buffer(df)

        result = read_pv_profile([csv_buf])

        self.assertEqual(len(result), 8760)
        self.assertEqual(result["hour_index"].min(), 0)
        self.assertEqual(result["hour_index"].max(), 8759)
        self.assertAlmostEqual(result.loc[result["hour_index"] == 0, "pv_mw"].iat[0], 0.5)
        self.assertAlmostEqual(result.loc[result["hour_index"] == 2, "pv_mw"].iat[0], 1.5)
        self.assertEqual(result.loc[result["hour_index"] == 4, "pv_mw"].iat[0], 0.0)

    def test_duplicate_hours_are_averaged(self) -> None:
        df = pd.DataFrame(
            {
                "hour_index": [0, 0, 1, 2],
                "pv_mw": [1.0, 3.0, 2.0, 4.0],
            }
        )
        csv_buf = self._make_csv_buffer(df)

        result = read_pv_profile([csv_buf])

        self.assertAlmostEqual(result.loc[result["hour_index"] == 0, "pv_mw"].iat[0], 2.0)
        self.assertAlmostEqual(result.loc[result["hour_index"] == 1, "pv_mw"].iat[0], 2.0)
        self.assertEqual(result.loc[result["hour_index"] == 3, "pv_mw"].iat[0], 0.0)

    def test_non_numeric_rows_raise_error(self) -> None:
        df = pd.DataFrame(
            {
                "hour_index": ["bad", "invalid"],
                "pv_mw": ["x", "y"],
            }
        )
        csv_buf = self._make_csv_buffer(df)

        with self.assertRaises(RuntimeError):
            read_pv_profile([csv_buf])


if __name__ == "__main__":
    unittest.main()
