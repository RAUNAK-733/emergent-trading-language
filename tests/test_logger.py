"""Tests for persistent experiment logging."""

import json
import os
import tempfile
import unittest

from utils.logger import ExperimentLogger


class ExperimentLoggerTests(unittest.TestCase):
    def test_append_sorts_and_replaces_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "training_log.json")
            logger = ExperimentLogger(path)
            logger.append({"update": 1000, "efficiency": 0.1})
            logger.append({"update": 500, "efficiency": 0.05})
            logger.append({"update": 1000, "efficiency": 0.2})

            records = logger.load()

        self.assertEqual([record["update"] for record in records], [500, 1000])
        self.assertEqual(records[-1]["efficiency"], 0.2)

    def test_written_log_is_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "training_log.json")
            ExperimentLogger(path).append({"update": 500})

            with open(path, encoding="utf-8") as file:
                records = json.load(file)

        self.assertEqual(records, [{"update": 500}])

    def test_reset_removes_previous_log(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "training_log.json")
            logger = ExperimentLogger(path)
            logger.append({"update": 500})
            logger.reset()

            self.assertEqual(logger.load(), [])

    def test_update_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = ExperimentLogger(os.path.join(directory, "training_log.json"))
            with self.assertRaises(ValueError):
                logger.append({"efficiency": 0.1})
