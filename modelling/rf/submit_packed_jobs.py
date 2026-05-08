import os
import subprocess
from datetime import datetime

# --- Configuration ---
LAUNCHER_SCRIPT = '/u/dansp/egnn/errorbar_modelling/modelling/rf/run_all_on_node.py'
VENV_PATH = '/u/dansp/egnn/py12_venv/bin/activate'
ROOT_DIR = '/u/dansp/egnn/rf_results'

SLURM_TEMPLATE = """#!/bin/bash -l
#SBATCH -J rf_master_sweep
#SBATCH -o {log_dir}/master.out
#SBATCH -e {log_dir}/master.err
#SBATCH -D {log_dir}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --mem=0

source {venv_path}

echo "Running Master Launcher..."
python3 {launcher_script}
"""

def main():
    date_str = datetime.now().strftime('%Y_%m_%d')
    base_output = os.path.join(ROOT_DIR, f"rf_sweep_{date_str}_packed")
    os.makedirs(base_output, exist_ok=True)
    
    print(f"Preparing single node submission in: {base_output}")

    slurm_content = SLURM_TEMPLATE.format(
        log_dir=base_output,
        venv_path=VENV_PATH,
        launcher_script=LAUNCHER_SCRIPT
    )
    
    slurm_path = os.path.join(base_output, "run_master.slurm")
    with open(slurm_path, "w") as f:
        f.write(slurm_content)
        
    print(f"Submitting Master Job...")
    # Using --exclusive is redundant with the #SBATCH line but good practice for ensuring general partition
    subprocess.run(["sbatch", "--partition=general", slurm_path])

if __name__ == "__main__":
    main()
