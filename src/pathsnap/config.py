"""Project paths. Override the data location with PATHSNAP_DATA if needed."""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.environ.get("PATHSNAP_DATA", os.path.join(ROOT, "data"))
TRAIN_DIR = os.path.join(DATA, "train")
STUDENT_TEST_DIR = os.path.join(DATA, "student_test")
ARTIFACTS = os.path.join(ROOT, "artifacts")
REPORTS = os.path.join(ROOT, "reports")
FIGURES = os.path.join(REPORTS, "figures")
MAPS = os.path.join(REPORTS, "maps")
CACHE = os.path.join(ROOT, "cache")   # large intermediate tables, git-ignored

CAND_CFG = {"radius": 50.0, "fallback_radius": 120.0, "k": 12}

for _d in (ARTIFACTS, REPORTS, FIGURES, MAPS, CACHE):
    os.makedirs(_d, exist_ok=True)
