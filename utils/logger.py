"""Small, interruption-safe JSON logger for experiment metrics."""

import json
import os


class ExperimentLogger:
    """Persist ordered training records without duplicating updates."""

    def __init__(self, path):
        self.path = path

    def load(self):
        """Load existing records, returning an empty list when none exist."""
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError(f"Expected a list of records in {self.path}.")
        return data

    def append(self, record):
        """Add or replace one update and atomically write the log."""
        if "update" not in record:
            raise ValueError("Training log records require an update field.")

        records_by_update = {
            int(existing["update"]): existing
            for existing in self.load()
            if isinstance(existing, dict) and "update" in existing
        }
        records_by_update[int(record["update"])] = record
        records = [
            records_by_update[update]
            for update in sorted(records_by_update)
        ]

        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        temporary_path = f"{self.path}.tmp"
        with open(temporary_path, "w", encoding="utf-8") as file:
            json.dump(records, file, indent=2)
            file.write("\n")
        os.replace(temporary_path, self.path)

    def reset(self):
        """Remove a previous run's metrics before fresh training."""
        if os.path.exists(self.path):
            os.remove(self.path)
