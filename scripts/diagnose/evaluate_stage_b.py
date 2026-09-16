#!/usr/bin/env python3
"""Read-only checkpoint evaluation on cached Stage A features; no real backbone."""
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
from cocf.core.accelerator import Accelerator
from cocf.data import CounterfactualLMDBDataset, ProcessedLayout, collate_cocf_samples
from cocf.lcocf.damage import DEFAULT_DAMAGE_WEIGHTS
from cocf.lcocf.predictor import build_predictor_input_batch
from cocf.training.checkpoint import load_checkpoint
from cocf.training.stage_b_losses import (
    damage_scalar_batch, per_sample_budget, batch_float, _local_cmsc_violation,
)

# `scripts` is not an installed package; an unrelated package may shadow it.
_training_path = Path(__file__).resolve().parents[1] / 'train' / 'train_stage_b.py'
if not _training_path.is_file():
    raise ImportError(f'Missing project training entry point: {_training_path}')
_training_spec = importlib.util.spec_from_file_location('_stage_b_eval_training_entry', _training_path)
_training_entry = importlib.util.module_from_spec(_training_spec)
_training_spec.loader.exec_module(_training_entry)
_apply_stage_a_geometry = _training_entry._apply_stage_a_geometry
_infer_dims_from_store = _training_entry._infer_dims_from_store


def summarize(rows, train_targets):
    result = {}
    for name, selected in [('all', rows), ('nonfull', [r for r in rows if r['action'] != 0])] + [
        (name, [r for r in rows if r['action'] == a])
        for a, name in enumerate(('full', 'lowfreq', 'interp', 'anchor'))
    ]:
        if not selected:
            result[name] = {'n': 0}
            continue
        y = np.array([r['target'] for r in selected])
        mu = np.array([r['mu'] for r in selected])
        sigma = np.array([r['sigma'] for r in selected])
        cert = np.array([r['certificate'] for r in selected])
        means = np.array([np.mean(train_targets[r['action']]) for r in selected])
        medians = np.array([np.median(train_targets[r['action']]) for r in selected])
        result[name] = dict(
            n=len(y), mae=float(np.abs(mu-y).mean()), zero_mae=float(np.abs(y).mean()),
            action_mean_mae=float(np.abs(means-y).mean()),
            action_median_mae=float(np.abs(medians-y).mean()),
            target_mean=float(y.mean()), target_std=float(y.std()),
            prediction_mean=float(mu.mean()), prediction_std=float(mu.std()),
            pearson=float(np.corrcoef(y, mu)[0, 1]) if y.std() > 1e-10 and mu.std() > 1e-10 else None,
            sigma_mean=float(sigma.mean()), sigma_min=float(sigma.min()),
            within_2sigma=float((np.abs(y-mu) <= 2*sigma).mean()),
            certificate_violation=float((cert < y).mean()),
            certificate_slack_mean=float((cert-y).mean()),
            absolute_error_p95=float(np.quantile(np.abs(y-mu), .95)),
        )
    return result


def collect_inputs(acc, loader):
    """Keep only compact predictor inputs, not video/text tensors."""
    chunks = {k: [] for k in ('tube_features', 'strength_features', 'step_frac', 'budget', 'action')}
    for batch in loader:
        for key in chunks:
            value = per_sample_budget(acc, batch, device=torch.device('cpu')) if key == 'budget' else batch[key]
            chunks[key].append(value.detach().cpu())
    return {k: torch.cat(v) for k, v in chunks.items()}


def audit_fields(dataset):
    fields = ('tube_features', 'strength_features', 'step_frac', 'damage_label',
              'text_embed', 'tube_visual_embed_full', 'tube_visual_embed_cf')
    missing = dict.fromkeys(fields, 0)
    nonfinite = dict.fromkeys(fields, 0)
    for sample in dataset:
        for key in fields:
            value = sample.get(key)
            if value is None or np.asarray(value).size == 0:
                missing[key] += 1
            elif not np.isfinite(np.asarray(value)).all():
                nonfinite[key] += 1
    return dict(missing_or_empty_samples=missing, nonfinite_samples=nonfinite)


def input_statistics(pool):
    stats = {}
    for key, tensor in pool.items():
        if key == 'action':
            continue
        values = tensor.double().reshape(len(tensor), -1)
        columns = []
        for i in range(values.shape[1]):
            col = values[:, i]
            finite = col[torch.isfinite(col)]
            columns.append(dict(index=i, nonfinite=int((~torch.isfinite(col)).sum()),
                mean=float(finite.mean()) if len(finite) else None,
                std=float(finite.std(unbiased=False)) if len(finite) else None,
                min=float(finite.min()) if len(finite) else None,
                max=float(finite.max()) if len(finite) else None,
                zero_fraction=float((finite == 0).double().mean()) if len(finite) else None))
        stats[key] = columns
    return stats


def shuffle_inputs(pool, seed):
    """Shuffle joint input rows within each action across the entire split."""
    generator = torch.Generator().manual_seed(seed)
    order = torch.arange(len(pool['action']))
    for action in torch.unique(pool['action']):
        indices = torch.where(pool['action'] == action)[0]
        order[indices] = indices[torch.randperm(len(indices), generator=generator)]
    return {k: v[order] if k != 'action' else v for k, v in pool.items()}


@torch.inference_mode()
def predict(acc, loader, device, *, inputs=None, zero_state=False):
    rows = []
    offset = 0
    for batch in loader:
        states = batch['tube_features'].to(device).float()
        features = batch['strength_features'].to(device).float()
        actions = batch['action'].to(device).long()
        step = batch['step_frac'].to(device).float()
        budget = per_sample_budget(acc, batch, device=device)
        if inputs is not None:
            end = offset + len(actions)
            if not torch.equal(inputs['action'][offset:end].to(device), actions):
                raise ValueError('Perturbation input actions/order do not match evaluation batch')
            states = inputs['tube_features'][offset:end].to(device).float()
            features = inputs['strength_features'][offset:end].to(device).float()
            step = inputs['step_frac'][offset:end].to(device).float()
            budget = inputs['budget'][offset:end].to(device).float()
        if zero_state:
            states = torch.zeros_like(states)
            features = torch.zeros_like(features)
        offset += len(actions)
        inp = build_predictor_input_batch(
            states=states, strength_feats=features, strength=acc.lcocf.strength_field(features),
            budget=budget,
            step_frac=step,
            step_embed_dim=acc.config.lcocf.predictor.context_dim,
        )
        pred = acc.lcocf.predictor(inp)
        idx = actions[:, None]
        mu = pred.mu.gather(-1, idx).squeeze(-1)
        sigma = pred.sigma.gather(-1, idx).squeeze(-1)
        # Perturbed rows are a predictor sensitivity probe, not certificate calibration.
        if inputs is not None or zero_state:
            cert = torch.full_like(mu, 0)
        else:
            cert = acc.raec.certificate.value(
                mu, sigma, residual=batch_float(batch, 'skip_residual', mu),
                boundary=states[:, 3], anchor_age=states[:, 6],
                local_cmsc=_local_cmsc_violation(acc, batch, mu),
            )
        target = damage_scalar_batch(batch['damage_label'].to(device))
        values = torch.stack((actions, target, mu, sigma, cert), dim=1).cpu().tolist()
        for action, y, m, s, c in values:
            if not np.isfinite([y, m, s, c]).all():
                raise ValueError('Nonfinite label or prediction encountered')
            rows.append(dict(action=int(action), target=y, mu=m, sigma=s, certificate=c))
    return rows


def canonical_split(name):
    return 'test_hard' if name == 'test' else name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--compare-checkpoint', type=Path, help='Optional final checkpoint to compare')
    parser.add_argument('--diagnostics', action='store_true', help='Also evaluate train and input sensitivity')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--split', type=canonical_split, choices=['val', 'test_hard'],
                        default='val', help='Evaluation split; test is an alias for test_hard')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if args.compare_checkpoint and not args.diagnostics:
        parser.error('--compare-checkpoint requires --diagnostics')
    for path in [args.checkpoint, args.compare_checkpoint]:
        if path is not None and not path.is_file():
            parser.error(f'Checkpoint not found: {path}')
    if args.output.exists():
        parser.error('Output exists; choose a new report filename')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; use --device cpu')
    setup_logging()
    log = get_logger('cocf.eval_stage_b')
    layout = ProcessedLayout(args.processed_root)
    if not layout.read_stage_a_env():
        parser.error('Missing Stage A environment metadata')
    indexed = {r['sample_id'] for r in layout.read_sample_index()}
    train_ids = layout.read_split('train')
    eval_ids = layout.read_split(args.split)
    if not train_ids or not eval_ids or set(train_ids) & set(eval_ids):
        parser.error('Empty or overlapping train/evaluation splits')
    if len(set(train_ids)) != len(train_ids) or len(set(eval_ids)) != len(eval_ids):
        parser.error('Duplicate sample ids in split')
    if not (set(train_ids) | set(eval_ids)) <= indexed:
        parser.error('Split contains samples absent from retained sample_index')
    def loader(ids):
        dataset = CounterfactualLMDBDataset(layout.lmdb_dir, ids, text_embed_dir=layout.text_embed_dir)
        if set(dataset.keys) != set(ids):
            raise ValueError('Requested samples missing from store')
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                          num_workers=0, collate_fn=collate_cocf_samples)
    targets = {a: [] for a in range(4)}
    for batch in loader(train_ids):
        for a, y in zip(batch['action'].tolist(), damage_scalar_batch(batch['damage_label']).tolist()):
            targets[a].append(y)
    if any(not v or not np.isfinite(v).all() for v in targets.values()):
        parser.error('Training baseline has missing actions or nonfinite targets')
    config = Config()
    config.backbone.device = args.device
    _apply_stage_a_geometry(config, layout, log)
    text_dim, visual_dim = _infer_dims_from_store(layout, log)
    acc = Accelerator.from_config(config, text_dim=text_dim, visual_dim=visual_dim)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    load_checkpoint(acc, checkpoint, allow_incomplete=True)
    acc.to(args.device).eval()
    rows = predict(acc, loader(eval_ids), torch.device(args.device))
    if len(rows) != len(eval_ids):
        raise ValueError('Evaluation sample count mismatch')
    report = dict(checkpoint=str(args.checkpoint), split=args.split,
                  damage_weights=DEFAULT_DAMAGE_WEIGHTS,
                  baseline_source='train split only', metrics=summarize(rows, targets),
                  predictions=[dict(sample_id=sid, **row) for sid, row in zip(eval_ids, rows)])
    if args.diagnostics:
        report['seed'] = args.seed
        report['field_audit'] = {}
        for split, ids in [('train', train_ids), (args.split, eval_ids)]:
            log.info('Auditing stored fields: %s', split)
            report['field_audit'][split] = audit_fields(loader(ids).dataset)
        report['diagnostics'] = {}
        report['diagnostic_notes'] = (
            'Train metrics are in-sample, not generalization estimates. Shuffle jointly permutes '
            'state/strength/step/budget within each action over the whole split. Zero-state keeps '
            'step/budget unchanged and may be out of distribution. Perturbation certificate metrics '
            'are intentionally omitted. Zero-valued features do not prove missing data.'
        )
        for label, path in [('best', args.checkpoint)] + ([('compare', args.compare_checkpoint)] if args.compare_checkpoint else []):
            log.info('Diagnostics checkpoint=%s path=%s', label, path)
            load_checkpoint(acc, torch.load(path, map_location='cpu', weights_only=False),
                            allow_incomplete=True)
            acc.eval()
            splits = {}
            for split, ids in [('train', train_ids), (args.split, eval_ids)]:
                log.info('Evaluating %s: %d samples', split, len(ids))
                pool = collect_inputs(acc, loader(ids))
                original = predict(acc, loader(ids), torch.device(args.device))
                probes = {}
                for mode in ('shuffle', 'zero_state'):
                    changed = predict(acc, loader(ids), torch.device(args.device),
                        inputs=shuffle_inputs(pool, args.seed) if mode == 'shuffle' else None,
                        zero_state=mode == 'zero_state')
                    metrics = summarize(changed, targets)
                    for group in metrics.values():
                        group.pop('certificate_violation', None)
                        group.pop('certificate_slack_mean', None)
                    deltas = np.array([abs(a['mu']-b['mu']) for a, b in zip(original, changed)])
                    probes[mode] = dict(metrics=metrics, mean_abs_prediction_change=float(deltas.mean()),
                                        max_abs_prediction_change=float(deltas.max()))
                splits[split] = dict(metrics=summarize(original, targets),
                                     input_statistics=input_statistics(pool), sensitivity=probes)
            report['diagnostics'][label] = dict(checkpoint=str(path), splits=splits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(report['metrics'], indent=2, allow_nan=False))
    print(f'Report: {args.output}')


if __name__ == '__main__':
    main()
