import os
import yaml

def load_deepseek_config():
    """
    Loads 'deepseek_config.yaml' (in this directory).
    Returns a dict with {api_key, base_url, model_name, ...}.
    """
    config_path = os.path.join(os.path.dirname(__file__), "deepseek_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg
