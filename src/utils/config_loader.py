"""
Configuration loader for YAML files
"""
import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """Load and manage configuration files"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
    
    def load(self, config_name: str) -> Dict[str, Any]:
        """
        Load a configuration file
        
        Args:
            config_name: Name of config file (without .yaml extension)
        
        Returns:
            Dictionary containing configuration
        """
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def save(self, config: Dict[str, Any], config_name: str):
        """
        Save a configuration file
        
        Args:
            config: Configuration dictionary
            config_name: Name of config file (without .yaml extension)
        """
        config_path = self.config_dir / f"{config_name}.yaml"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)


if __name__ == "__main__":
    # Test
    loader = ConfigLoader()
    print("ConfigLoader ready")
