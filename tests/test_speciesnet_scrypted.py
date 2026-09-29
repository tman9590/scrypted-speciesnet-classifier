from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "distill"))

from speciesnet_scrypted import IMAGE_SIZE, parse_speciesnet_labels, scrypted_config


class SpeciesNetScryptedTests(unittest.TestCase):
    def test_label_parser_falls_back_and_disambiguates(self):
        labels = parse_speciesnet_labels(
            [
                "1;mammalia;carnivora;felidae;felis;catus;domestic cat",
                "2;mammalia;carnivora;felidae;other;catus;domestic cat",
                "3;aves;passeriformes;family;genus;species;",
                "4;;;;;;vehicle",
                "55555555-a;;;family;bird;one;same bird",
                "66666666-b;;;family;bird;one;same bird",
            ]
        )
        self.assertEqual(labels[0], "domestic cat (felis catus)")
        self.assertEqual(labels[1], "domestic cat (other catus)")
        self.assertEqual(labels[2], "genus species")
        self.assertEqual(labels[3], "vehicle")
        self.assertEqual(labels[4], "same bird (bird one) [55555555]")
        self.assertEqual(labels[5], "same bird (bird one) [66666666]")

    def test_scrypted_config_is_classifier_contract(self):
        config = scrypted_config(["cat", "dog"], ["model.xml", "model.bin"])
        self.assertEqual(config["model"], "resnet")
        self.assertEqual(config["input_shape"], [1, 3, IMAGE_SIZE, IMAGE_SIZE])
        self.assertEqual(config["mean"], [0.0, 0.0, 0.0])
        self.assertEqual(config["std"], [1.0, 1.0, 1.0])
        self.assertEqual(config["labels"], {"0": "cat", "1": "dog"})

    def test_artifact_order_keeps_ncnn_binary_ahead_of_openvino_binary(self):
        names = json.loads((ROOT / "config.json").read_text())["files"]
        self.assertLess(
            names.index(next(name for name in names if name.endswith(".ncnn.bin"))),
            names.index(next(name for name in names if name.endswith("/openvino/speciesnet-v4.0.3a.bin"))),
        )

    def test_bad_taxonomy_row_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected 7"):
            parse_speciesnet_labels(["too;few;fields"])


if __name__ == "__main__":
    unittest.main()
