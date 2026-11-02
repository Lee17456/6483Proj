"""Check that the suite assigns independent processes to distinct GPU masks."""

import sys
import tempfile
import unittest
from pathlib import Path

from run_suite import run_parallel_gpu_jobs


class MultiGpuRunnerTests(unittest.TestCase):
    def test_each_gpu_sees_only_its_assigned_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = []
            for index in range(4):
                command = [sys.executable, "-c",
                           "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])"]
                jobs.append((f"job{index}", root / f"job{index}", command))
            run_parallel_gpu_jobs(jobs, [0, 1], poll_interval=0.01)
            observed = [(root / f"job{index}" / "console.log").read_text().strip()
                        for index in range(4)]
            self.assertEqual(observed, ["0", "1", "0", "1"])


if __name__ == "__main__":
    unittest.main()
