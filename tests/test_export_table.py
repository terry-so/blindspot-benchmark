"""Synthetic results verify denominators, pairing, and variable L3 lengths."""

import json
import tempfile
import unittest
from pathlib import Path

from utils.export_table import make_analysis_tables


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cases = [
            dict(
                case_id=name,
                domain="D1",
                subcategory="1.1",
                prompts={"L3": ["edit"] * length},
            )
            for name, length in (("a", 2), ("b", 3), ("c", 2))
        ]
        (self.root / "data").mkdir()
        (self.root / "data/dataset.json").write_text(json.dumps(self.cases))

    def save(
        self,
        model,
        case,
        level="L1",
        *,
        status="ok",
        r=0,
        e=1,
        q=4,
        repeat=0,
        refusal_step=None,
    ):
        path = self.root / "results" / model / f"{case}_{level}_{repeat}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        row = dict(
            model=model,
            case_id=case,
            domain="D1",
            subcategory="1.1",
            level=level,
            repeat=repeat,
            status=status,
            r=r,
            e=e,
            q=q,
        )
        if level == "L3-reset":
            length = next(
                len(item["prompts"]["L3"])
                for item in self.cases
                if item["case_id"] == case
            )
            row.update(
                chain_length=length,
                first_refusal_step=refusal_step,
                steps=[{"step": i} for i in range(1, (refusal_step or length) + 1)],
            )
        path.write_text(json.dumps(row))

    def tables(self, **kwargs):
        return make_analysis_tables(
            self.root / "results", self.root / "tables", **kwargs
        )

    def test_errors_missing_and_metric_denominators(self):
        self.save("m", "a")
        self.save("m", "b", status="error", r=None, e=None, q=None)
        tables = self.tables()
        row = tables["overall"].iloc[0]
        self.assertEqual(
            (row.N_expected, row.N_valid, row.N_errors, row.N_missing), (3, 1, 1, 1)
        )
        self.assertEqual(row["HRR_count"], "1/1")
        self.assertEqual(row["HRR (%)"], 100)
        absent = tables["overall"].query("Level == 'L2a'").iloc[0]
        self.assertEqual(absent.N_missing, 3)
        self.assertTrue(__import__("pandas").isna(absent["HRR (%)"]))
        self.assertIn("L3-reset HRR_count", tables["comparison"].columns)
        self.assertFalse(
            any(
                "L3-session" in column
                for table in tables.values()
                for column in table.columns
            )
        )

    def test_conditional_metrics(self):
        self.save("m", "a", r=1, e=None, q=None)
        self.save("m", "b", e=0, q=None)
        self.save("m", "c", q=3)
        row = self.tables()["overall"].iloc[0]
        self.assertEqual(row["RR (%)"], 33.3)
        self.assertEqual(row["EF (%)"], 50)
        self.assertEqual(row["RS (%)"], 0)
        self.assertEqual(row["HRR (%)"], 0)
        self.assertEqual(row["HRR_count"], "0/3")

    def test_matched_model_comparisons_and_paired_transitions(self):
        self.save("m1", "a", r=1, e=None, q=None)
        self.save("m1", "b")
        self.save("m2", "a")
        self.save("m2", "b", status="error", r=None, e=None, q=None)
        self.save("m1", "a", "L2a")
        self.save("m1", "b", "L2a", r=1, e=None, q=None)
        self.save("m1", "c", "L2a")
        tables = self.tables()
        matched = tables["overall_matched"].query("Level == 'L1'")
        self.assertEqual(list(matched.N_valid), [1, 1])
        self.assertEqual(list(matched["HRR (%)"]), [0, 100])
        paired = (
            tables["paired_levels"]
            .query("Model == 'm1' and Comparison == 'L1 -> L2a'")
            .iloc[0]
        )
        self.assertEqual(paired.N_valid_pairs, 2)
        self.assertEqual(paired.Not_harmful_to_harmful, 1)
        self.assertEqual(paired.Harmful_to_not_harmful, 1)
        self.assertEqual(paired["HRR_change (pp)"], 0)

    def test_l3_reached_denominators_and_lengths(self):
        self.save("m", "a", "L3-reset", r=1, e=None, q=None, refusal_step=1)
        self.save("m", "b", "L3-reset", r=1, e=None, q=None, refusal_step=3)
        self.save("m", "c", "L3-reset")
        tables = self.tables()
        steps = tables["l3_steps"]
        self.assertEqual(list(steps.N_reached_step), [3, 2, 1])
        self.assertEqual(list(steps.N_refused_at_step), [1, 0, 1])
        self.assertEqual(list(steps["Refusal_among_reached (%)"]), [33.3, 0, 100])
        self.assertEqual(list(tables["l3_chain_lengths"].N_expected), [2, 1])
        self.assertEqual(tables["l3_summary"].iloc[0].No_refusal_through_chain, 1)

    def test_repeat_indices_and_duplicate_records_are_rejected(self):
        self.save("m", "a", repeat=1)
        with self.assertRaises(ValueError):
            self.tables()
        tables = self.tables(repeats=2)
        self.assertEqual(tables["overall"].iloc[0].N_expected, 6)
        path = next((self.root / "results/m").glob("*.json"))
        (path.parent / "duplicate.json").write_bytes(path.read_bytes())
        with self.assertRaises(ValueError):
            self.tables(repeats=2)


if __name__ == "__main__":
    unittest.main()
