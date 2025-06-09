import os
import yaml
import re
import logging

# Set up a logger for configuration
logger = logging.getLogger("voxarch.config")
logger.setLevel(logging.INFO)

class Config:
    """
    Singleton configuration loader for the pipeline.
    Loads a YAML config and expands environment variables in values.
    """
    _instance = None

    def __new__(cls, config_path="voxarch/config/config.yaml"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            try:
                cls._instance._load(config_path)
                logger.info(f"Loaded config from: {config_path}")
            except Exception as e:
                logger.error(f"Failed to load config from {config_path}: {e}")
                raise
        return cls._instance

    def _load(self, path):
        """
        Loads and parses the YAML config file.
        """
        try:
            with open(path, "r") as f:
                cfg = yaml.safe_load(f)
            self._config = self._expand_env(cfg)
        except FileNotFoundError:
            logger.error(f"Config file '{path}' not found.")
            raise
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in '{path}': {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading config: {e}")
            raise

    def get(self, key_path, default=None):
        """
        Returns a config value using dot notation, e.g. get("chunking.chunk_size").
        Logs a warning if the requested key does not exist.
        """
        keys = key_path.split(".")
        val = self._config
        for k in keys:
            if val is None or not isinstance(val, dict):
                logger.warning(f"Config path '{key_path}' not found. Returning default: {default}")
                return default
            val = val.get(k)
        if val is None:
            logger.warning(f"Config value for '{key_path}' is None. Returning default: {default}")
            return default
        return val

    def _expand_env(self, obj):
        """
        Recursively expands ${VAR} in strings to environment variable values.
        """
        if isinstance(obj, dict):
            return {k: self._expand_env(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._expand_env(i) for i in obj]
        elif isinstance(obj, str):
            matches = re.findall(r"\$\{(\w+)\}", obj)
            for var in matches:
                obj = obj.replace(f"${{{var}}}", os.getenv(var, ""))
            return obj
        else:
            return obj
