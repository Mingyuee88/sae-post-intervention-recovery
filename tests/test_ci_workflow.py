"""Checks that GitHub Actions only invokes scripts committed to the repo."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


class CiWorkflowTest(unittest.TestCase):
    def test_python_script_references_exist(self):
        workflow_text = WORKFLOW.read_text(encoding="utf-8")
        script_paths = re.findall(r"\bpython\s+([^\s]+\.py)\b", workflow_text)

        self.assertTrue(script_paths, "CI workflow does not invoke any Python scripts")
        missing = [path for path in script_paths if not (ROOT / path).is_file()]
        self.assertFalse(missing, f"CI workflow references missing scripts: {missing}")


if __name__ == "__main__":
    unittest.main()
