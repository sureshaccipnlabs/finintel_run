import json
from pathlib import Path


class AggregatedJSONLoader:

    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)

    def load(self) -> list[dict]:
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {self.folder_path}")

        payloads = []

        for file_path in sorted(self.folder_path.glob("*.json")):
            with open(file_path) as file_obj:
                payload = json.load(file_obj)

            payload["__source_file__"] = file_path.name
            payloads.append(payload)

        if not payloads:
            raise ValueError(f"No JSON files found in {self.folder_path}")

        return payloads
