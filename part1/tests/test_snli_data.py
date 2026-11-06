"""Offline checks for cleaning provenance and leakage/duplicate audit counts."""

import unittest

from datasets import ClassLabel, Dataset, DatasetDict, Features, Value
import pandas as pd

from part1.snli_data import LABEL_NAMES, build_data_report, clean_split


def make_dataset(premises, hypotheses, labels, label_names=LABEL_NAMES):
    return Dataset.from_dict(
        {"premise": premises, "hypothesis": hypotheses, "label": labels},
        features=Features({
            "premise": Value("string"), "hypothesis": Value("string"),
            "label": ClassLabel(names=list(label_names)),
        }),
    )


class SnliDataTests(unittest.TestCase):
    def test_filtering_preserves_original_ids_labels_and_whitespace(self):
        source = make_dataset(
            [" A dog runs. ", "A cat sits.", "A person stands.", "", "A dog runs.", "A dog runs."],
            ["An animal moves.", "A pet sits.", "Someone stands.", "Someone moves.",
             "An animal moves.", "An animal moves."],
            [0, -1, 1, 2, 0, 2],
        )
        cleaned, removed = clean_split(source, "train")
        self.assertEqual(list(cleaned["example_id"]), ["train:0", "train:2", "train:4", "train:5"])
        self.assertEqual(list(cleaned["source_row"]), [0, 2, 4, 5])
        self.assertEqual(list(cleaned["label"]), [0, 1, 0, 2])
        self.assertEqual(cleaned[0]["premise"], " A dog runs. ")
        self.assertEqual(removed["reason"].tolist(), ["invalid_label", "missing_or_empty_text"])
        self.assertEqual(len(source), 6)
        self.assertEqual(cleaned.features["label"].names, list(LABEL_NAMES))

    def test_missing_text_and_exclusion_reasons_are_counted_once(self):
        source = make_dataset(
            [None, "A person.", "  ", "A dog."],
            ["Someone.", None, "", "An animal."],
            [-1, 0, 1, 0],
        )
        cleaned, removed = clean_split(source, "validation")
        self.assertEqual(list(cleaned["example_id"]), ["validation:3"])
        self.assertEqual(removed["reason"].tolist(), [
            "invalid_label", "missing_or_empty_text", "missing_or_empty_text",
        ])

    def test_unexpected_label_mapping_is_rejected(self):
        source = make_dataset(
            ["A dog."], ["An animal."], [0],
            label_names=("neutral", "entailment", "contradiction"),
        )
        with self.assertRaisesRegex(ValueError, "Unexpected SNLI label mapping"):
            clean_split(source, "train")

    def test_overlap_counts_unique_groups_and_affected_rows_separately(self):
        raw = DatasetDict({
            "train": make_dataset(
                ["shared", "shared", "train only"], ["same", "same", "other"], [0, 2, 1],
            ),
            "validation": make_dataset(
                ["shared", "validation only"], ["different", "other"], [1, 0],
            ),
            "test": make_dataset(["test only"], ["unseen"], [2]),
        })
        cleaned, audits = DatasetDict(), []
        for split, source in raw.items():
            cleaned[split], audit = clean_split(source, split)
            audits.append(audit)
        tables = build_data_report(raw, cleaned, pd.concat(audits, ignore_index=True))
        train = tables["split_summary"].set_index("split").loc["train"]
        self.assertEqual(train["clean_rows"], 3)
        self.assertEqual(train["unique_premises"], 2)
        self.assertEqual(train["repeated_premise_groups"], 1)
        self.assertEqual(train["extra_duplicate_pair_rows"], 1)
        self.assertEqual(train["conflicting_label_pair_groups"], 1)
        overlap = tables["split_overlap"].set_index(["left_split", "right_split"]).loc[
            ("train", "validation")
        ]
        self.assertEqual(overlap["shared_unique_premises"], 1)
        self.assertEqual(overlap["left_rows_with_shared_premise"], 2)
        self.assertEqual(overlap["right_rows_with_shared_premise"], 1)
        self.assertEqual(overlap["shared_unique_pairs"], 0)
        counts = tables["label_counts"].query("split == 'train'")
        self.assertEqual(counts["count"].tolist(), [1, 1, 1])
        self.assertEqual(tables["duplicate_pairs"].iloc[0]["distinct_labels"], 2)
        self.assertEqual(tables["row_lengths"].set_index("example_id").loc[
            "test:0", "premise_whitespace_tokens"
        ], 2)


if __name__ == "__main__":
    unittest.main()
