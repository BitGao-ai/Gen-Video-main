import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from cocf.common.config import Config, DataConfig, _build_dataclass
from cocf.common.types import Action, AllocationDecision, TokenGrid
from cocf.common.vram import apply_wan_variant
from cocf.core.accelerator import Accelerator
from cocf.engine import InferenceEngine
from cocf.lcocf.data import COCFDataGenerator, _FullStepCache
from cocf.training.checkpoint import build_checkpoint
from cocf.training.lora import LoRALinear


class ReviewRegressions(unittest.TestCase):
    def test_stage_a_to_stage_b(self):
        from cocf.training.stage_a_data_gen import DataGenerationStage, StageAConfig
        from cocf.data.cocf_batch import collate_cocf_samples
        from cocf.training.stage_b_losses import compute_joint_loss
        from cocf.training.stage_b_joint import JointTrainingStage
        config = Config()
        config.backbone.name = "mock"
        config.backbone.device = "cpu"
        config.teacher.num_inference_steps = 3
        config.teacher.seeds_per_prompt = 2
        torch.manual_seed(7)
        acc = Accelerator.from_config(config)
        with tempfile.TemporaryDirectory() as directory:
            stage = DataGenerationStage(StageAConfig(openvid_csvs=[], processed_root=Path(directory),
                config=config, device=torch.device("cpu")), acc.backbone, acc.metric_extractor, acc)
            traj = stage.teacher_runner.run("smoke", "a person running", grid=TokenGrid(t=3, h=8, w=8))
            self.assertIsNotNone(traj)
            samples = stage.data_generator.generate(traj, acc.backbone, acc.transition,
                                                     max_tubes=1, max_samples=4)
            batch = collate_cocf_samples([s.to_dict() for s in samples])
            loss, _ = compute_joint_loss(acc, batch)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            trainer = object.__new__(JointTrainingStage)
            trainer.accelerator, trainer.device = acc, "cpu"
            trainer._reduce = lambda x: x
            metrics = trainer._validate([batch])
            self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))

    def test_nested_tuple_and_extra_preservation(self):
        data = _build_dataclass(DataConfig, {"resolution_buckets": [[49, 384, 640]]})
        self.assertEqual(data.resolution_buckets, ((49, 384, 640),))
        config = Config()
        config.backbone.extra = {"custom": 17}
        apply_wan_variant(config, SimpleNamespace(backbone="wan22", wan_variant="a14b-t2v"))
        self.assertEqual(config.backbone.extra["custom"], 17)

    def test_one_tube_selection(self):
        generator = object.__new__(COCFDataGenerator)
        traj = SimpleNamespace(tubes=[SimpleNamespace(tube_id=i) for i in range(3)],
            strength_feats={i: SimpleNamespace(s_E=i, s_A=i, s_T=i) for i in range(3)})
        self.assertEqual(generator._select_tubes(traj, 1), [2])
        with self.assertRaises(ValueError):
            generator._select_tubes(traj, 0)

    def test_flush_survives_abrupt_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            code = '''import os, sys
from cocf.data.sample_store import CounterfactualSampleWriter
w = CounterfactualSampleWriter(sys.argv[1], force_fallback=True, shard_size=256)
w.put("sample", {"value": 7})
w.flush()
os._exit(0)
'''
            subprocess.run([sys.executable, "-c", code, directory], check=True,
                           env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
            files = list(Path(directory).glob("*.pt"))
            self.assertEqual(len(files), 1)
            self.assertEqual(torch.load(files[0], weights_only=False)[0]["payload"]["value"], 7)

    def test_flow_and_anchor_age_in_real_engine_loop(self):
        config = Config()
        config.backbone.name = "mock"
        config.backbone.device = "cpu"
        config.engine.num_inference_steps = 3
        config.engine.tube_build_step = 0
        config.engine.tube_refresh_every = 0
        config.engine.risk_control_enabled = False
        config.engine.cf_check_enabled = False
        acc = Accelerator.from_config(config)
        engine = InferenceEngine(acc, config.engine, config.trigger)
        records = []
        def allocate(tubes, predictions, **kw):
            action = Action.FULL if kw["step"] == 0 else Action.ANCHOR
            return AllocationDecision(kw["step"], {t.tube_id: action for t in tubes}, 0.0, 1.0)
        grid = TokenGrid(t=3, h=8, w=8)
        with patch.object(acc.allocator, "allocate", side_effect=allocate), \
             patch.object(acc.raec, "certify", return_value=SimpleNamespace(value=0.0)), torch.no_grad():
            engine.generate(["person running"], torch.randn(1, grid.num_tokens, acc.token_dim),
                grid, acc.backbone.encode_text(["person running"]), acc.backbone,
                record_sink=lambda **kw: records.append([(s.motion_phase, s.anchor_age) for s in kw["tube_states"].values()]))
        self.assertEqual(len(records), 3)
        self.assertTrue(any(motion > 0 for motion, _ in records[0]))
        self.assertTrue(all(age == 2 for _, age in records[2]))

    def test_checkpoint_uses_actual_lora_geometry(self):
        block = torch.nn.Sequential(LoRALinear(torch.nn.Linear(2, 2), rank=2, alpha=6))
        backbone = SimpleNamespace(lora_roots=lambda: [("transformer", block)],
            dit_blocks=lambda: [block], lora_target_blocks=lambda n: [block])
        acc = SimpleNamespace(backbone=backbone, state_dict=lambda: {},
            config=SimpleNamespace(backbone=SimpleNamespace(name="test")),
            token_dim=2, text_dim=2, visual_dim=2)
        ckpt = build_checkpoint(acc, rank=8, alpha=16, last_n_blocks=4)
        self.assertEqual(ckpt["lora_config"], {"rank": 2, "alpha": 6.0, "last_n_blocks": 1})

    def test_seed_reference_cached_per_seed(self):
        traj = SimpleNamespace(prompt="test")
        cache = _FullStepCache(traj, None, 0.02)
        cache.get = lambda step, seed: (torch.tensor(seed), torch.tensor(seed))
        calls = []
        generator = SimpleNamespace(_rollout=lambda before, after, *args: (calls.append(int(before)) or after, 0))
        damage = SimpleNamespace(reference_features=lambda video, prompt, **kw: int(video))
        tube = SimpleNamespace(tube_id=0)
        first = cache.reference(0, 1, generator, tube, None, damage, None)
        second = cache.reference(0, 1, generator, tube, None, damage, None)
        self.assertEqual(first[1], 1)
        self.assertEqual(second[1], 1)
        self.assertEqual(calls, [1])


if __name__ == "__main__":
    unittest.main()
