import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/data/run_stage_a_multigpu.sh'


class StageALauncherTests(unittest.TestCase):
    def preview(self, **settings):
        env = dict(os.environ)
        for name in ('CUDA_VISIBLE_DEVICES', 'GPU_IDS', 'NUM_GPUS', 'NUM_SHARDS',
                     'MAX_CONCURRENT', 'SHARD_IDS', 'FINALIZE_PARTIAL'):
            env.pop(name, None)
        env.update(DRY_RUN='1', GPU_IDS='0', NUM_GPUS='1', PYTHON_BIN=sys.executable)
        env.update(settings)
        return subprocess.run(['bash', str(SCRIPT)], env=env, text=True,
                              capture_output=True)

    def test_eight_shards_one_gpu(self):
        result = self.preview(NUM_SHARDS='8')
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [line for line in result.stdout.splitlines()
                    if line.startswith('CUDA_VISIBLE_DEVICES=')]
        self.assertEqual(len(commands), 8)
        for index, command in enumerate(commands):
            self.assertIn('CUDA_VISIBLE_DEVICES=0 ', command)
            self.assertIn('--num-shards 8 --shard-index ' + str(index), command)

    def test_selected_shards_batched(self):
        result = self.preview(GPU_IDS='2,3', NUM_GPUS='2', NUM_SHARDS='8',
                              SHARD_IDS='0,1,2')
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [line for line in result.stdout.splitlines()
                    if line.startswith('CUDA_VISIBLE_DEVICES=')]
        self.assertEqual(len(commands), 3)
        for command, gpu in zip(commands, (2, 3, 2)):
            self.assertIn(f'CUDA_VISIBLE_DEVICES={gpu} ', command)

    def test_invalid_selection(self):
        for ids in ('0,0', '8', '', '0,', '-1', '08'):
            with self.subTest(ids=ids):
                self.assertNotEqual(self.preview(NUM_SHARDS='8', SHARD_IDS=ids).returncode, 0)

    def test_concurrency_cannot_exceed_gpus(self):
        self.assertNotEqual(self.preview(MAX_CONCURRENT='2').returncode, 0)

    def test_execution_and_failure_gate(self):
        for fail in ('', '1'):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                runner = root / 'python_runner'
                runner.write_text(
                    f'#!{sys.executable}\n'
                    'import os, sys\n'
                    'if "generate_counterfactual_data.py" not in " ".join(sys.argv):\n'
                    '    os.execv(sys.executable, [sys.executable] + sys.argv[1:])\n'
                    'args = sys.argv[1:]\n'
                    'shard = args[args.index("--shard-index") + 1]\n'
                    'final = "--finalize-only" in args\n'
                    'with open(os.environ["CALLS"], "a") as f:\n'
                    '    f.write(("final" if final else shard) + "\\n")\n'
                    'sys.exit(7 if not final and shard == os.environ["FAIL_SHARD"] else 0)\n'
                )
                runner.chmod(0o755)
                # The integration test isolates launcher logic, not OS flock.
                flock = root / 'flock'
                flock.write_text('#!/bin/sh\nexit 0\n')
                flock.chmod(0o755)
                csv = root / 'input.csv'
                csv.touch()
                env = dict(os.environ)
                for name in ('CUDA_VISIBLE_DEVICES', 'SHARD_IDS'):
                    env.pop(name, None)
                env.update(
                    PATH=str(root) + os.pathsep + env['PATH'],
                    PYTHON_BIN=str(runner), GPU_IDS='0,1', NUM_GPUS='2',
                    NUM_SHARDS='3', MAX_CONCURRENT='2', DRY_RUN='0',
                    FINALIZE_PARTIAL='0', RAM_PER_WORKER_GIB='1',
                    OPENVID_CSV=str(csv), DATA_ROOT=tmp, VIDEO_SUBDIR='.',
                    MODEL_PATH=tmp, SAM_MODEL=tmp, DINO_MODEL=tmp,
                    CLIP_MODEL=tmp, RAFT_WEIGHTS=tmp,
                    PROCESSED_ROOT=str(root / 'processed'), LOG_DIR=str(root / 'logs'),
                    CALLS=str(root / 'calls'), FAIL_SHARD=fail,
                )
                result = subprocess.run(['bash', str(SCRIPT)], env=env, text=True,
                                        capture_output=True)
                calls = (root / 'calls').read_text().splitlines()
                if fail:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(sorted(calls), ['0', '1'])
                else:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(sorted(calls[:2]), ['0', '1'])
                    self.assertEqual(calls[2:], ['2', 'final'])


if __name__ == '__main__':
    unittest.main()
