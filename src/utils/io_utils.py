"""
File I/O utilities
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Union, Any


def load_csv(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load CSV file into DataFrame"""
    return pd.read_csv(filepath)


def save_csv(df: pd.DataFrame, filepath: Union[str, Path]):
    """Save DataFrame to CSV file"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


def load_pickle(filepath: Union[str, Path]) -> Any:
    """Load pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_pickle(obj: Any, filepath: Union[str, Path]):
    """Save object to pickle file"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_numpy(filepath: Union[str, Path]) -> np.ndarray:
    """Load numpy array"""
    return np.load(filepath)


def save_numpy(arr: np.ndarray, filepath: Union[str, Path]):
    """Save numpy array"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    np.save(filepath, arr)


def ensure_dir(directory: Union[str, Path]):
    """Ensure directory exists"""
    Path(directory).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print("I/O utilities ready")
