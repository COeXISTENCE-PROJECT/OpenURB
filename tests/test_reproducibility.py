import os
import pandas as pd
import pytest
import shutil
import subprocess

from pathlib import Path

SCRIPTS_DIR = Path("scripts")
RESULTS_DIR = Path("results")
python_script = [SCRIPTS_DIR / "cond_open_iql.py"]

@pytest.fixture(scope="session", autouse=True)
def check_sumo_installed():
    sumo_executable = shutil.which("sumo")
    if sumo_executable is None:
        pytest.exit("[SUMO ERROR] SUMO is not installed or not in PATH.")
    else:
        try:
            result = subprocess.run(
                ["sumo", "--version"], capture_output=True, text=True, check=True
            )
            print(f"[DEBUG] SUMO version: {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            pytest.exit(f"[SUMO ERROR] Failed to get SUMO version: {e.stderr}")


@pytest.mark.parametrize("script_path", python_script)
def test_python_script_execution(script_path):
    script_filename = script_path.name
    id1 = f"test_{script_filename}1"
    id2 = f"test_{script_filename}2"
    
    try:
        result = subprocess.run(
            ["python", script_filename,
             "--id", id1,
             "--alg-conf", "test",
             "--env-conf", "test",
             "--task-conf", "dynamic_test",
             "--net", "saint_arnoult",
             "--env-seed", 0,
             "--torch-seed", 0],
            capture_output=True, text=True, check=True, cwd=script_path.parent
        )
        print(f"[DEBUG] Successfully executed {script_path} for experiment {id1}")
    except subprocess.CalledProcessError as e:
        pytest.fail(f"[FAIL] {script_path} failed for experiment {id1}: {e.stderr}")
        
        
    try:
        result = subprocess.run(
            ["python", script_filename,
             "--id", id2,
             "--alg-conf", "test",
             "--env-conf", "test",
             "--task-conf", "dynamic_test",
             "--net", "saint_arnoult",
             "--env-seed", "0",
             "--torch-seed", "0"],
            capture_output=True, text=True, check=True, cwd=script_path.parent
        )
        print(f"[DEBUG] Successfully executed {script_path} for experiment {id2}")
    except subprocess.CalledProcessError as e:
        pytest.fail(f"[FAIL] {script_path} failed for experiment {id2}: {e.stderr}")
        
    # Compare the results of the two runs
    last_ep = max(int(file.stem[2:]) for file in (RESULTS_DIR / id1 / "episodes").glob("ep*.csv"))
    ep1_path = RESULTS_DIR / id1 / "episodes" / f"ep{last_ep}.csv"
    ep2_path = RESULTS_DIR / id2 / "episodes" / f"ep{last_ep}.csv"
    
    ep1_df = pd.read_csv(ep1_path)
    ep2_df = pd.read_csv(ep2_path)
    
    if not ep1_df.equals(ep2_df):
        pytest.fail(f"[FAIL] The episode files {ep1_path} and {ep2_path} are not the same.")