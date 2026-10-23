"""Exercise the offline mobile client with Node's built-in browser-state seams."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_phone_web_client_state_and_delivery():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed only for mobile JavaScript regression checks")
    result = subprocess.run(
        [node, str(Path(__file__).with_name("phone_web.test.cjs"))],
        text=True, capture_output=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
