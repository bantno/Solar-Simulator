# configuration.py
import yaml

def load_config(file_path):
    """
    Load configuration parameters from a YAML file.
    
    Parameters:
        file_path: Path to the YAML configuration file.
        
    Returns:
        Dictionary with configuration parameters.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config
