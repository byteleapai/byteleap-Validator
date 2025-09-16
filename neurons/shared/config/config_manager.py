"""
Simplified Configuration Management
Basic configuration loading and management
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import bittensor as bt
import yaml


class ConfigManager:
    """Fail-fast configuration manager with strict access control"""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize configuration manager"""
        self.config = config or {}

    def get(self, path: str) -> Any:
        """
        Get required configuration value with fail-fast behavior

        Args:
            path: Key or dot-separated path (e.g., 'host' or 'database.host')

        Returns:
            Configuration value

        Raises:
            KeyError: If configuration key is missing
        """
        if "." in path:
            return self.get_nested(path)
        else:
            if path not in self.config:
                bt.logging.error(f"❌ Missing config key | key={path}")
                raise KeyError(f"Missing required configuration key: '{path}'")
            return self.config[path]

    def get_nested(self, path: str, separator: str = ".") -> Any:
        """
        Get nested configuration value with fail-fast behavior

        Args:
            path: Dot-separated path (e.g., 'database.host')
            separator: Path separator (default: '.')

        Returns:
            Configuration value

        Raises:
            KeyError: If any key in the path is missing
        """
        keys = path.split(separator)
        value = self.config

        for i, key in enumerate(keys):
            if not isinstance(value, dict) or key not in value:
                current_path = separator.join(keys[: i + 1])
                bt.logging.error(
                    f"Missing required configuration key: '{current_path}' in path '{path}'"
                )
                raise KeyError(
                    f"Missing required configuration key: '{current_path}' in path '{path}'"
                )
            value = value[key]

        return value

    def get_optional(self, path: str, default: Any = None) -> Any:
        """
        Get optional configuration value (allows None/missing values)

        Args:
            path: Key or dot-separated path (e.g., 'external_ip')
            default: Default value if path not found

        Returns:
            Configuration value or default
        """
        try:
            if "." in path:
                return self._get_nested_optional(path, default)
            else:
                return self.config.get(path, default)
        except KeyError:
            return default

    def _get_nested_optional(
        self, path: str, default: Any = None, separator: str = "."
    ) -> Any:
        """Get nested optional configuration value"""
        keys = path.split(separator)
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration"""
        self.config.update(updates)

    def get_positive_number(
        self, path: str, number_type: type = int
    ) -> Union[int, float]:
        """
        Require a positive number configuration value

        Args:
            path: Configuration key path
            number_type: int or float

        Returns:
            Positive number value

        Raises:
            KeyError: If configuration key is missing
            ValueError: If value is not a positive number
        """
        value = self.get(path)

        if not isinstance(value, number_type):
            bt.logging.error(
                f"Configuration key '{path}' must be {number_type.__name__}, got {type(value).__name__}: {value}"
            )
            raise ValueError(
                f"Configuration key '{path}' must be {number_type.__name__}, got {type(value).__name__}: {value}"
            )

        if value <= 0:
            bt.logging.error(
                f"Configuration key '{path}' must be positive, got: {value}"
            )
            raise ValueError(
                f"Configuration key '{path}' must be positive, got: {value}"
            )

        return value

    def get_non_empty_string(self, path: str) -> str:
        """
        Require a non-empty string configuration value

        Args:
            path: Configuration key path

        Returns:
            Non-empty string value

        Raises:
            KeyError: If configuration key is missing
            ValueError: If value is not a non-empty string
        """
        value = self.get(path)

        if not isinstance(value, str):
            bt.logging.error(
                f"Configuration key '{path}' must be str, got {type(value).__name__}: {value}"
            )
            raise ValueError(
                f"Configuration key '{path}' must be str, got {type(value).__name__}: {value}"
            )

        if not value.strip():
            bt.logging.error(f"❌ Config key empty | key={path}")
            raise ValueError(f"Configuration key '{path}' cannot be empty")

        return value

    def get_range(
        self, path: str, min_val: float, max_val: float, number_type: type = float
    ) -> Union[int, float]:
        """
        Get a number within a specific range

        Args:
            path: Configuration key path
            min_val: Minimum allowed value (inclusive)
            max_val: Maximum allowed value (inclusive)
            number_type: Expected numeric type (int or float)

        Returns:
            Number within specified range

        Raises:
            KeyError: If configuration key is missing
            ValueError: If value is outside range or wrong type
        """
        value = self.get(path)

        if not isinstance(value, number_type):
            bt.logging.error(
                f"Configuration key '{path}' must be {number_type.__name__}, got {type(value).__name__}: {value}"
            )
            raise ValueError(
                f"Configuration key '{path}' must be {number_type.__name__}, got {type(value).__name__}: {value}"
            )

        if not (min_val <= value <= max_val):
            bt.logging.error(
                f"Configuration key '{path}' must be between {min_val} and {max_val}, got {value}"
            )
            raise ValueError(
                f"Configuration key '{path}' must be between {min_val} and {max_val}, got {value}"
            )

        return value

    def get_list(self, path: str, min_length: int = 0) -> List[Any]:
        """
        Get a list configuration value with optional minimum length

        Args:
            path: Configuration key path
            min_length: Minimum required list length

        Returns:
            List value

        Raises:
            KeyError: If configuration key is missing
            ValueError: If value is not a list or too short
        """
        value = self.get(path)

        if not isinstance(value, list):
            bt.logging.error(
                f"Configuration key '{path}' must be list, got {type(value).__name__}: {value}"
            )
            raise ValueError(
                f"Configuration key '{path}' must be list, got {type(value).__name__}: {value}"
            )

        if len(value) < min_length:
            bt.logging.error(
                f"Configuration key '{path}' must have at least {min_length} items, got {len(value)}"
            )
            raise ValueError(
                f"Configuration key '{path}' must have at least {min_length} items, got {len(value)}"
            )

        return value

    @classmethod
    def from_yaml(cls, config_path: str) -> "ConfigManager":
        """Create config manager from YAML file"""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(path, "r") as f:
            config_data = yaml.safe_load(f) or {}

        return cls(config_data)


def load_config(config_path: str, defaults: Dict[str, Any] = None) -> Dict[str, Any]:
    """Simple function to load configuration from YAML file"""
    path = Path(config_path)
    if not path.exists():
        bt.logging.warning(
            f"Configuration file not found: {config_path}, using defaults"
        )
        return defaults or {}

    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f) or {}

        # Merge with defaults
        if defaults:
            merged = defaults.copy()
            merged.update(config)
            return merged

        return config

    except Exception as e:
        bt.logging.error(f"❌ Config load error | path={config_path} error={e}")
        return defaults or {}
