import json
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_configs(config_dir="config"):
    config_dir_path = Path(config_dir)
    if not config_dir_path.exists():
        repo_root = Path(__file__).resolve().parent.parent.parent
        config_dir_path = repo_root / "config"

    active_config_path = config_dir_path / "active_config.json"
    active_config = load_json(active_config_path)

    active_use_case = active_config["active_use_case"]

    use_case_dir = config_dir_path / "use_cases" / active_use_case

    schema = load_json(use_case_dir / "schema.json")
    pipeline_config = load_json(use_case_dir / "pipeline_config.json")
    model_config = load_json(use_case_dir / "model_config.json")

    return schema, pipeline_config, model_config
