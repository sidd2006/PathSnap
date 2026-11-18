"""Build the candidate/feature tables for the train and student-test partitions.

Outputs
  cache/features_{train,student_test}.pkl   full candidate tables (git-ignored)
  reports/candidate_stats_{partition}.csv   per-trip candidate miss rate, label distances
"""
import os
import sys
import time

import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap import config  # noqa: E402
from pathsnap.features import build_trip_table  # noqa: E402
from pathsnap.io import list_trips, load_trip  # noqa: E402


def one(root, tid):
    df, _, stats = build_trip_table(load_trip(root, tid), **config.CAND_CFG)
    return df, stats


def main():
    for name, root in (("train", config.TRAIN_DIR), ("student_test", config.STUDENT_TEST_DIR)):
        t0 = time.time()
        trips = list_trips(root)
        res = Parallel(n_jobs=-1, verbose=0)(delayed(one)(root, t) for t in trips)
        df = pd.concat([r[0] for r in res], ignore_index=True)
        stats = pd.DataFrame([r[1] for r in res])
        df.to_pickle(os.path.join(config.CACHE, f"features_{name}.pkl"))
        stats.to_csv(os.path.join(config.REPORTS, f"candidate_stats_{name}.csv"), index=False)
        print(f"{name}: {len(trips)} trips, {stats.n_obs.sum()} obs, {len(df)} candidate rows, "
              f"pos rate {df.label.mean():.3f}, candidate miss rate "
              f"{1 - df.groupby(['trip_id', 'obs']).label.max().sum() / stats.n_obs.sum():.4f} "
              f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
