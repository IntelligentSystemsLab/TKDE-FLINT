import os
import subprocess

def run_script(script_path):
    try:
        subprocess.run(['python', script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing {script_path}: {e}")
    except FileNotFoundError:
        print(f"Script not found: {script_path}")

path = None
run_script(path)