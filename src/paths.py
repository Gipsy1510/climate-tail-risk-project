from pathlib import Path

# Project root directory
ROOT = Path(__file__).resolve().parents[1]

# Main project folders
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
SQL = ROOT / "sql"
NOTEBOOKS = ROOT / "notebooks"
SRC = ROOT / "src"