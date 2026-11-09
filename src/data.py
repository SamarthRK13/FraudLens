"""Data loading helpers for FraudLens."""
from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "paysim.csv"


def load_paysim(path=DATA_PATH, frac=0.1, seed=42):
    """Load a random sample of PaySim without reading all of it into memory at once.

    frac=0.1 keeps about 10% of rows (roughly 636,000), which preserves the real fraud rate.
    Use frac=1.0 for the full dataset if your laptop has plenty of memory (8 GB or more).
    """
    chunks = pd.read_csv(path, chunksize=500_000)
    parts = [chunk.sample(frac=frac, random_state=seed) for chunk in chunks]
    return pd.concat(parts, ignore_index=True)