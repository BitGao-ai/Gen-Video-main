import subprocess
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.diagnose.evaluate_stage_b import (
    predict, summarize, collect_inputs, shuffle_inputs, input_statistics, audit_fields,
    canonical_split,
)
from cocf.common.config import Config
from cocf.common.types import TokenGrid
from cocf.core.accelerator import Accelerator
from cocf.data.cocf_batch import collate_cocf_samples
from cocf.training.checkpoint import build_checkpoint, load_checkpoint
from cocf.training.stage_a_data_gen import DataGenerationStage, StageAConfig


class EvaluationTests(unittest.TestCase):
    def test_evaluation_split_alias_matches_store(self):
        import argparse
        from cocf.data.processed_layout import ProcessedLayout
        parser = argparse.ArgumentParser()
        parser.add_argument('--split', type=canonical_split, choices=['val', 'test_hard'])
        with tempfile.TemporaryDirectory() as tmp:
            layout = ProcessedLayout(Path(tmp))
            layout.write_splits(['train_sample'], ['val_sample'], ['heldout_sample'])
            for name, expected in [('val', 'val_sample'), ('test', 'heldout_sample'),
                                   ('test_hard', 'heldout_sample')]:
                split = parser.parse_args(['--split', name]).split
                self.assertEqual(layout.read_split(split), [expected])
                self.assertNotEqual(split, 'test')

    def test_training_only_baselines(self):
        rows = [dict(action=1, target=1., mu=.5, sigma=.2, certificate=1.1)]
        result = summarize(rows, {1: [0., 0.]})['all']
        self.assertEqual(result['action_mean_mae'], 1.)
        self.assertEqual(result['action_median_mae'], 1.)
        self.assertEqual(result['mae'], .5)
        self.assertIsNone(result['pearson'])

    def test_mock_checkpoint_inference_is_read_only(self):
        config = Config()
        config.backbone.name = 'mock'
        config.backbone.device = 'cpu'
        config.teacher.num_inference_steps = 3
        config.teacher.seeds_per_prompt = 2
        acc = Accelerator.from_config(config)
        with tempfile.TemporaryDirectory() as directory:
            stage = DataGenerationStage(StageAConfig(openvid_csvs=[], processed_root=Path(directory),
                config=config, device=torch.device('cpu')), acc.backbone, acc.metric_extractor, acc)
            trajectory = stage.teacher_runner.run('eval', 'a person running', grid=TokenGrid(t=3, h=8, w=8))
            samples = stage.data_generator.generate(trajectory, acc.backbone, acc.transition,
                                                    max_tubes=1, max_samples=4)
            batch = collate_cocf_samples([s.to_dict() for s in samples])
            checkpoint = build_checkpoint(acc)
            load_checkpoint(acc, checkpoint)
            acc.eval()
            before = {k: v.clone() for k, v in acc.state_dict().items()}
            rows = predict(acc, [batch], torch.device('cpu'))
            self.assertEqual(len(rows), len(samples))
            pool = collect_inputs(acc, [batch])
            for altered in (predict(acc, [batch], torch.device('cpu'), inputs=shuffle_inputs(pool, 42)),
                            predict(acc, [batch], torch.device('cpu'), zero_state=True)):
                self.assertEqual([(r['action'], r['target']) for r in rows],
                                 [(r['action'], r['target']) for r in altered])
            for key, value in acc.state_dict().items():
                torch.testing.assert_close(value, before[key])
            self.assertTrue(all(p.grad is None for p in acc.parameters()))

    def test_shuffle_preserves_actions_and_joint_rows(self):
        pool = dict(action=torch.tensor([0]*10+[1]*10),
                    tube_features=torch.arange(20)[:, None],
                    strength_features=torch.arange(20)[:, None]+100)
        shuffled = shuffle_inputs(pool, 7)
        torch.testing.assert_close(shuffled['action'], pool['action'])
        torch.testing.assert_close(shuffled['strength_features'], shuffled['tube_features']+100)
        self.assertFalse(torch.equal(shuffled['tube_features'], pool['tube_features']))
        for key in pool:
            torch.testing.assert_close(shuffled[key], shuffle_inputs(pool, 7)[key])

    def test_statistics_and_missing_fields(self):
        stats = input_statistics({'x': torch.tensor([[0., 1.], [0., float('nan')]])})
        self.assertEqual(stats['x'][0]['std'], 0)
        self.assertEqual(stats['x'][1]['nonfinite'], 1)
        audit = audit_fields([{'tube_features': [0., 1.]}])
        self.assertEqual(audit['missing_or_empty_samples']['tube_features'], 0)
        self.assertEqual(audit['missing_or_empty_samples']['text_embed'], 1)

    def test_cli_help(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/diagnose/evaluate_stage_b.py'
        result = subprocess.run([sys.executable, str(script), '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_with_conflicting_scripts_package(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/diagnose/evaluate_stage_b.py'
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / 'scripts'
            package.mkdir()
            (package / '__init__.py').write_text('raise RuntimeError("Wrong scripts package loaded")\n')
            env = dict(os.environ, PYTHONPATH=directory)
            result = subprocess.run([sys.executable, str(script), '--help'],
                                    env=env, cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
