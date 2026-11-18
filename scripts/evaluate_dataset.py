"""Evaluate the committed pipeline without training or tuning.

Usage: python scripts/evaluate_dataset.py --data data/evaluator_test
Raw observations and reference routes are never written to the reports.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pathsnap import config
from pathsnap.hmm import HMMConfig
from pathsnap.io import list_trips, load_trip
from pathsnap.pipeline import evaluate_trip, predict_trip


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path(config.DATA) / 'evaluator_test')
    parser.add_argument('--output', type=Path, default=Path(config.REPORTS) / 'evaluator_test')
    args = parser.parse_args()
    trip_ids = list_trips(args.data)
    if not trip_ids:
        parser.error(f'No trips found in {args.data}')
    inputs = sorted(args.data.glob('*/*'))
    input_hashes = {p.relative_to(args.data).as_posix(): sha256(p) for p in inputs if p.is_file()}
    artifacts = Path(config.ARTIFACTS)
    meta = json.loads((artifacts / 'final_model_meta.json').read_text())
    decoder = json.loads((artifacts / 'hmm_config.json').read_text())
    decoder.pop('tuned_on', None)
    model = joblib.load(artifacts / 'final_model.joblib')
    hmm = HMMConfig(**decoder)
    rows = []
    for tid in trip_ids:
        trip = load_trip(str(args.data), tid, with_route=False)
        out = predict_trip(trip, model, hmm, meta.get('candidate_config', config.CAND_CFG))
        # Reference labels are loaded only after prediction and used only for scoring.
        route = load_trip(str(args.data), tid, with_route=True).route
        if route is None:
            raise ValueError(f'Trip {tid} has no reference route; cannot score this dataset')
        scores = evaluate_trip(out, route)
        row = {'trip_id': tid, 'n_obs': len(out['ctx'].xy),
               **{f'clf_{k}': v for k, v in scores['classifier'].items()},
               **{f'hmm_{k}': v for k, v in scores['hmm'].items()},
               'hmm_breaks': out['diag']['breaks'],
               'time_total_s': out['timing_s']['total']}
        rows.append(row)
        print(f"{tid}: obs={row['n_obs']} classifier={row['clf_accuracy']:.4f} "
              f"HMM={row['hmm_accuracy']:.4f} continuity={row['hmm_route_continuity']:.4f} "
              f"route_F1={row['hmm_route_f1']:.4f} ({row['time_total_s']:.2f}s)", flush=True)
    df = pd.DataFrame(rows)
    weights = df.n_obs / df.n_obs.sum()
    summary = pd.DataFrame([
        {'method': name,
         'pointwise_accuracy_obs_weighted': float((df[f'{prefix}_accuracy'] * weights).sum()),
         'pointwise_accuracy_trip_mean': float(df[f'{prefix}_accuracy'].mean()),
         **{metric: float(df[f'{prefix}_{metric}'].mean()) for metric in
            ('route_continuity', 'route_precision', 'route_recall', 'route_f1')},
         'trips_with_acc_ge_95': int((df[f'{prefix}_accuracy'] >= .95).sum())}
        for prefix, name in [('clf', 'Classifier only'), ('hmm', 'Classifier + HMM')]
    ])
    provenance = {
        'dataset': args.data.name, 'trips': len(df), 'observations': int(df.n_obs.sum()),
        'model': meta['model'], 'decoder': decoder,
        'artifact_sha256': {name: sha256(artifacts / name) for name in
                           ('final_model.joblib', 'final_model_meta.json', 'hmm_config.json')},
        'input_sha256': input_hashes,
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=config.ROOT, text=True).strip(),
        'evaluation_script_sha256': sha256(__file__),
        'python': platform.python_version(),
        'packages': {name: importlib.metadata.version(name) for name in
                     ('numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'joblib')},
        'training_or_tuning_performed': False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output / 'per_trip.csv', index=False)
    summary.to_csv(args.output / 'summary.csv', index=False)
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    report = (f'# Evaluator results: {args.data.name}\n\n'
              f'{len(df)} trips; {int(df.n_obs.sum()):,} cleaned GPS observations. '
              'The committed model and HMM configuration were used without training or tuning. '
              'Reference routes were loaded only after prediction for scoring.\n\n'
              'Reproduce from the repository root:\n\n'
              '```sh\npython scripts/evaluate_dataset.py --data data/evaluator_test\n```\n\n'
              'Accuracy below is observation-weighted; route metrics are means across trips, '
              'matching the student-test summary protocol.\n\n'
              + summary.to_markdown(index=False, floatfmt='.4f') + '\n\n'
              + df[['trip_id', 'n_obs', 'clf_accuracy', 'hmm_accuracy', 'hmm_route_continuity',
                    'hmm_route_f1', 'hmm_breaks', 'time_total_s']].round(4).to_markdown(index=False, disable_numparse=True)
              + '\n\nTimings cover candidate generation, classifier inference and HMM decoding on this laptop; '
              'they exclude file loading and reference scoring. Raw trip data is excluded from Git. '
              'Artifact/input hashes and runtime versions are recorded in `provenance.json`.\n')
    (args.output / 'RESULTS.md').write_text(report, encoding='utf-8')
    print('\n' + summary.round(4).to_string(index=False), flush=True)
    print(f'\nSaved reports to {args.output}', flush=True)


if __name__ == '__main__':
    main()
