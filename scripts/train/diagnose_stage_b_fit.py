#!/usr/bin/env python3
"""Run small-data predictor-only fitting probe."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from torch.utils.data import DataLoader

from cocf.common.config import Config
from cocf.common.logging import setup_logging, get_logger
from cocf.core.accelerator import Accelerator
from cocf.data import ProcessedLayout, CounterfactualLMDBDataset, collate_cocf_samples
from cocf.lcocf.predictor import build_predictor_input_batch
from cocf.training.stage_b_losses import damage_scalar_batch, gaussian_nll, per_sample_budget

spec = importlib.util.spec_from_file_location('_stage_b_evaluation', ROOT / 'scripts/diagnose/evaluate_stage_b.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def regression_loss(target, mu, sigma, actions, objective, scale=100.0):
    """Compute regression loss for selected actions."""
    selected = actions != 0
    if not selected.any():
        return None
    y, m, s = target[selected], mu[selected], sigma[selected]
    if objective == 'nll':
        return gaussian_nll(y, m, s)
    if objective == 'mse':
        return torch.nn.functional.mse_loss(m * scale, y * scale)
    return torch.nn.functional.smooth_l1_loss(m * scale, y * scale)


def main():
    """Run predictor-only fit experiment."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--processed-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--objective', choices=['nll', 'mse', 'huber'], default='mse')
    p.add_argument('--train-videos', type=int, default=8)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--target-scale', type=float, default=100.0)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = p.parse_args()
    if min(args.train_videos, args.epochs, args.batch_size, args.lr, args.target_scale) <= 0:
        p.error('Counts, learning rate and scale must be positive')
    if args.output_dir.exists():
        p.error('Use a new output directory; existing results are never overwritten')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable')
    torch.manual_seed(args.seed)
    setup_logging()
    log = get_logger('cocf.fit_probe')
    layout = ProcessedLayout(args.processed_root)
    index = {r['sample_id']: r for r in layout.read_sample_index()}
    train = layout.read_split('train')
    val = layout.read_split('val')
    if not train or not val or set(train) & set(val):
        p.error('Missing or overlapping splits')
    if not (set(train) | set(val)) <= index.keys():
        p.error('Split contains unindexed samples')
    videos = sorted({index[s]['video_id'] for s in train})
    order = torch.randperm(len(videos), generator=torch.Generator().manual_seed(args.seed)).tolist()
    selected_videos = {videos[i] for i in order[:args.train_videos]}
    ids = [s for s in train if index[s]['video_id'] in selected_videos]
    def loader(keys, shuffle=False):
        """Build dataloader for given keys."""
        ds = CounterfactualLMDBDataset(layout.lmdb_dir, keys, text_embed_dir=layout.text_embed_dir)
        if list(ds.keys) != list(keys):
            raise ValueError('Missing/reordered dataset samples')
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=0,
                          generator=torch.Generator().manual_seed(args.seed), collate_fn=collate_cocf_samples)
    config = Config()
    config.backbone.device = args.device
    if not layout.read_stage_a_env():
        p.error('Missing Stage A geometry metadata')
    evaluation._apply_stage_a_geometry(config, layout, log)
    text_dim, visual_dim = evaluation._infer_dims_from_store(layout, log)
    acc = Accelerator.from_config(config, text_dim=text_dim, visual_dim=visual_dim).to(args.device)
    for param in acc.parameters():
        param.requires_grad_(False)
    predictor = acc.lcocf.predictor
    for name, param in predictor.named_parameters():
        param.requires_grad_(args.objective == 'nll' or not name.startswith('var_head.'))
    optimizer = torch.optim.AdamW([x for x in predictor.parameters() if x.requires_grad],
                                 lr=args.lr, weight_decay=0)
    train_loader = loader(ids, True)
    baseline = {a: [] for a in range(4)}
    for batch in loader(ids):
        for a, y in zip(batch['action'].tolist(), damage_scalar_batch(batch['damage_label']).tolist()):
            baseline[a].append(y)
    if any(not values for values in baseline.values()):
        p.error('Selected videos must cover all four actions')
    args.output_dir.mkdir(parents=True)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    settings.update(selected_videos=sorted(selected_videos), train_samples=len(ids),
                    note='Predictor-only diagnostic; no warmup, early stopping or auxiliary losses. '
                         'MSE/Huber use scaled targets; sigma is frozen for these objectives.')
    (args.output_dir / 'settings.json').write_text(json.dumps(settings, indent=2))
    initial = {k: v.detach().clone() for k, v in predictor.named_parameters()}
    log.info('Probe settings: %s', settings)
    with (args.output_dir / 'history.jsonl').open('x') as history:
        for epoch in range(args.epochs + 1):
            acc.eval()
            grad = {}
            losses = []
            if epoch:
                for batch in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    states = batch['tube_features'].to(args.device).float()
                    feats = batch['strength_features'].to(args.device).float()
                    actions = batch['action'].to(args.device).long()
                    inp = build_predictor_input_batch(states, feats, acc.lcocf.strength_field(feats),
                        per_sample_budget(acc, batch, device=args.device),
                        batch['step_frac'].to(args.device).float(), config.lcocf.predictor.context_dim)
                    pred = predictor(inp)
                    mu = pred.mu.gather(1, actions[:, None]).squeeze(1)
                    sigma = pred.sigma.gather(1, actions[:, None]).squeeze(1)
                    loss = regression_loss(damage_scalar_batch(batch['damage_label'].to(args.device)),
                                           mu, sigma, actions, args.objective, args.target_scale)
                    if loss is None:
                        continue
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite loss')
                    loss.backward()
                    for name, param in predictor.named_parameters():
                        if param.grad is not None:
                            if not torch.isfinite(param.grad).all():
                                raise ValueError(f'Nonfinite gradient: {name}')
                            grad[name] = max(grad.get(name, 0.), float(param.grad.norm()))
                    torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.)
                    optimizer.step()
                    losses.append(float(loss.detach()))
            record = dict(epoch=epoch, mean_batch_loss=sum(losses)/len(losses) if losses else None,
                max_preclip_grad_norm=grad,
                parameter_delta={k: float((v.detach()-initial[k]).norm()) for k, v in predictor.named_parameters()})
            for split, keys in [('train', ids), ('val', val)]:
                rows = evaluation.predict(acc, loader(keys), torch.device(args.device))
                metrics = evaluation.summarize(rows, baseline)
                for group in metrics.values():
                    for key in ('certificate_violation', 'certificate_slack_mean', 'within_2sigma'):
                        group.pop(key, None)
                record[split] = metrics
            history.write(json.dumps(record, allow_nan=False) + '\n')
            history.flush()
            log.info('epoch=%d train_nonfull_mae=%.7g val_nonfull_mae=%.7g', epoch,
                      record['train']['nonfull']['mae'], record['val']['nonfull']['mae'])
    torch.save({'predictor_only': predictor.state_dict(), 'diagnostic_only': True, 'settings': settings},
               args.output_dir / 'predictor_probe.pt')
    log.info('Done: %s/history.jsonl (diagnostic only)', args.output_dir)


if __name__ == '__main__':
    main()
