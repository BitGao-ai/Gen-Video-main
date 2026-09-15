import subprocess
import sys
import unittest
from pathlib import Path


class CheckLabelsCliTests(unittest.TestCase):
    def test_help_imports_current_store_api(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/diagnose/check_labels.py'
        result = subprocess.run(
            [sys.executable, str(script), '--help'],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--limit', result.stdout)
