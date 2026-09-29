"""End-to-end manipulation: the Menagerie Panda grasps and lifts randomized
cubes; a gripper that never closes must lift nothing (negative control)."""

import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANDA = os.path.join(ROOT, "third_party/mujoco_menagerie/franka_emika_panda/panda.xml")
pytestmark = pytest.mark.skipif(not os.path.exists(PANDA), reason="run scripts/fetch_assets.sh")


def run(*args):
    out = subprocess.run([sys.executable, "examples/panda_grasp.py", "--envs", "8", *args],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    line = [l for l in out.splitlines() if l.startswith("success:")][0]
    return int(line.split()[1].split("/")[0])


def test_grasp_lifts_every_cube():
    assert run("--seed", "5") == 8


def test_open_gripper_lifts_nothing():
    assert run("--keep-open") == 0
