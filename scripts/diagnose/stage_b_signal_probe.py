#!/usr/bin/env python3
"""Signal-ceiling probe for the Stage-B predictor inputs; no checkpoint needed.

The phased Stage-B run calibrates σ well, but its μ ends up near-constant within
each action (per-subset prediction_std ~1e-4). Before retraining μ harder, this
script measures how much per-sample signal the *inputs themselves* carry about
the damage target, using the exact feature vector the predictor is trained on
(``[tube_state(7), strength_feats(3), budget, step_frac]``):

* per-feature Pearson/Spearman correlation with the target (train split), with
  per-feature variance so dead columns (e.g. an unpopulated ``causal_value``)
  show up immediately;
* three val-split reference points per action subset:
    - the action-mean baseline (what a collapsed μ achieves),
    - ridge regression on step_frac+budget only (what (action, step) explains),
    - ridge regression on all 12 features (the linear signal ceiling),
    - k-NN regression on all features (a cheap nonlinear ceiling).

Interpretation: if the all-feature fits barely beat the action-mean baseline,
μ's collapse is a *feature* problem (retraining will not fix it); if they beat
it clearly with prediction_std well above 1e-4, it is an *optimisation* problem
and a longer, hotter mean phase should recover per-sample variation.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from cocf.common.config import Config
from cocf.common.logging import setup_logging, get_logger
from cocf.common.types import TUBE_STATE_FIELDS
from cocf.core.accelerator import Accelerator
from cocf.data import CounterfactualLMDBDataset, ProcessedLayout, collate_cocf_samples
from cocf.training.stage_b_losses import damage_scalar_batch, per_sample_budget

# `scripts` is not an installed package; an unrelated package may shadow it.
_training_path = Path(__file__).resolve().parents[1] / 'train' / 'train_stage_b.py'
if not _training_path.is_file():
    raise ImportError(f'Missing project training entry point: {_training_path}')
_training_spec = importlib.util.spec_from_file_location('_stage_b_probe_training_entry', _training_path)
_training_entry = importlib.util.module_from_spec(_training_spec)
_training_spec.loader.exec_module(_training_entry)
_apply_stage_a_geometry = _training_entry._apply_stage_a_geometry
_infer_dims_from_store = _training_entry._infer_dims_from_store

FEATURE_NAMES = list(TUBE_STATE_FIELDS) + ['strength_s_E', 'strength_s_A', 'strength_s_T',
                                           'budget', 'step_frac']
ACTION_NAMES = {1: 'lowfreq', 2: 'interp', 3: 'anchor'}
STEP_BUDGET_IDX = [FEATURE_NAMES.index('budget'), FEATURE_NAMES.index('step_frac')]


def collect(acc, loader):
    """Assemble (X, y, action) over one split with the training-time feature vector."""
    xs, ys, acts = [], [], []
    for batch in loader:
        budget = per_sample_budget(acc, batch, device=torch.device('cpu'))
        x = torch.cat([
            batch['tube_features'].detach().cpu().float(),
            batch['strength_features'].detach().cpu().float(),
            budget.reshape(-1, 1),
            batch['step_frac'].detach().cpu().float().reshape(-1, 1),
        ], dim=-1)
        xs.append(x)
        ys.append(damage_scalar_batch(batch['damage_label'].detach().cpu().float()))
        acts.append(batch['action'].detach().cpu().long())
    return (torch.cat(xs).numpy(), torch.cat(ys).numpy(), torch.cat(acts).numpy())


def _ranks(v):
    """Average ranks (ties share their mean rank); mergsort keeps it stable."""
    order = np.argsort(v, kind='mergesort')
    ranks = np.empty(len(v), dtype=np.float64)
    sorted_v = v[order]
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def _corr(a, b):
    if len(a) < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def ridge_fit(X, y, lam):
    xm, xs = X.mean(0), X.std(0)
    xs[xs < 1e-12] = 1.0
    Xs = (X - xm) / xs
    w = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ (y - y.mean()))
    return xm, xs, w, float(y.mean())


def ridge_predict(model, X):
    xm, xs, w, b = model
    return ((X - xm) / xs) @ w + b


def knn_predict(X_train, y_train, X_val, k):
    xm, xs = X_train.mean(0), X_train.std(0)
    xs[xs < 1e-12] = 1.0
    tr, va = (X_train - xm) / xs, (X_val - xm) / xs
    # Squared distances via the expansion; splits are ~1e3 rows so this is cheap.
    d2 = (va ** 2).sum(1, keepdims=True) + (tr ** 2).sum(1) - 2 * va @ tr.T
    neigh = np.argpartition(d2, k, axis=1)[:, :k]
    return y_train[neigh].mean(1)


def _fit_report(y_val, pred, action_means):
    return dict(
        mae=float(np.abs(pred - y_val).mean()),
        pearson=_corr(y_val, pred),
        prediction_std=float(pred.std()),
        beats_action_mean_mae=float(np.abs(action_means - y_val).mean()
                                    - np.abs(pred - y_val).mean()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--ridge-lambda', type=float, default=1.0)
    parser.add_argument('--knn-k', type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1 or args.ridge_lambda <= 0 or args.knn_k < 1:
        parser.error('--batch-size/--knn-k must be positive, --ridge-lambda > 0')
    if args.output.exists():
        parser.error('Output exists; choose a new report filename')
    setup_logging()
    log = get_logger('cocf.stage_b_signal_probe')
    layout = ProcessedLayout(args.processed_root)
    if not layout.read_stage_a_env():
        parser.error('Missing Stage A environment metadata')
    train_ids, eval_ids = layout.read_split('train'), layout.read_split(args.split)
    if not train_ids or not eval_ids or set(train_ids) & set(eval_ids):
        parser.error('Empty or overlapping train/evaluation splits')

    # The accelerator is built only for its budget scheduler — the same one
    # Stage B trains against (per_sample_budget is a scheduler lookup).
    config = Config()
    config.backbone.device = 'cpu'
    _apply_stage_a_geometry(config, layout, log)
    text_dim, visual_dim = _infer_dims_from_store(layout, log)
    acc = Accelerator.from_config(config, text_dim=text_dim, visual_dim=visual_dim)

    def loader(ids):
        dataset = CounterfactualLMDBDataset(layout.lmdb_dir, ids,
                                            text_embed_dir=layout.text_embed_dir)
        if set(dataset.keys) != set(ids):
            raise ValueError('Requested samples missing from store')
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                          num_workers=0, collate_fn=collate_cocf_samples)

    log.info('Collecting features: %d train / %d %s samples', len(train_ids),
             len(eval_ids), args.split)
    Xtr, ytr, atr = collect(acc, loader(train_ids))
    Xva, yva, ava = collect(acc, loader(eval_ids))
    if not (np.isfinite(Xtr).all() and np.isfinite(Xva).all()):
        raise ValueError('Nonfinite feature encountered; run check_labels.py first')

    feature_stats = {
        name: dict(train_std=float(Xtr[:, i].std()),
                   train_min=float(Xtr[:, i].min()),
                   train_max=float(Xtr[:, i].max()))
        for i, name in enumerate(FEATURE_NAMES)
    }
    dead = [n for n, s in feature_stats.items() if s['train_std'] < 1e-12]
    if dead:
        log.warning('Dead (zero-variance) feature columns on the train split: %s', dead)

    report = dict(processed_root=str(args.processed_root), split=args.split,
                  n_train=len(ytr), n_eval=len(yva), feature_names=FEATURE_NAMES,
                  ridge_lambda=args.ridge_lambda, knn_k=args.knn_k,
                  feature_stats=feature_stats, dead_features=dead, subsets={})

    subsets = [('nonfull', atr != 0, ava != 0)]
    subsets += [(name, atr == a, ava == a) for a, name in ACTION_NAMES.items()]
    for name, tr_mask, va_mask in subsets:
        Xt, yt = Xtr[tr_mask], ytr[tr_mask]
        Xv, yv = Xva[va_mask], yva[va_mask]
        if len(yt) < 10 or len(yv) < 5:
            report['subsets'][name] = dict(n_train=len(yt), n_eval=len(yv), skipped=True)
            continue
        action_mean = float(yt.mean())
        means_va = np.full(len(yv), action_mean)
        entry = dict(
            n_train=len(yt), n_eval=len(yv),
            target_mean=float(yv.mean()), target_std=float(yv.std()),
            action_mean_mae=float(np.abs(means_va - yv).mean()),
            zero_mae=float(np.abs(yv).mean()),
            feature_correlation={
                FEATURE_NAMES[i]: dict(pearson=_corr(Xt[:, i], yt),
                                       spearman=_corr(_ranks(Xt[:, i]), _ranks(yt)))
                for i in range(Xt.shape[1])
            },
            ridge_step_budget=_fit_report(
                yv, ridge_predict(ridge_fit(Xt[:, STEP_BUDGET_IDX], yt, args.ridge_lambda),
                                  Xv[:, STEP_BUDGET_IDX]), means_va),
            ridge_all=_fit_report(
                yv, ridge_predict(ridge_fit(Xt, yt, args.ridge_lambda), Xv), means_va),
            knn_all=_fit_report(
                yv, knn_predict(Xt, yt, Xv, min(args.knn_k, len(yt))), means_va),
        )
        report['subsets'][name] = entry
        log.info(
            '%-8s n=%d  action_mean_mae=%.5f | ridge(step+budget) mae=%.5f r=%s | '
            'ridge(all) mae=%.5f r=%s std=%.5f | knn(all) mae=%.5f r=%s std=%.5f',
            name, len(yv), entry['action_mean_mae'],
            entry['ridge_step_budget']['mae'], _fmt_r(entry['ridge_step_budget']['pearson']),
            entry['ridge_all']['mae'], _fmt_r(entry['ridge_all']['pearson']),
            entry['ridge_all']['prediction_std'],
            entry['knn_all']['mae'], _fmt_r(entry['knn_all']['pearson']),
            entry['knn_all']['prediction_std'],
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    log.info('Wrote %s', args.output)


def _fmt_r(value):
    return f'{value:+.3f}' if value is not None else ' n/a '


if __name__ == '__main__':
    main()
