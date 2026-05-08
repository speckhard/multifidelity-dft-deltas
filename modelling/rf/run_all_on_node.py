import subprocess
import os
import sys

# Configuration
PYTHON_EXEC = sys.executable 
SCRIPT_PATH = '/u/dansp/egnn/errorbar_modelling/modelling/rf/rf_trainer.py'
DATA_FILE = '/u/dansp/egnn/relaxation_data_with_kpoints/csv_data/delta_combined_relaxations_23_22__12_1_2026_kpoints_included_no_duplicates.csv'
BASE_OUTPUT_DIR = '/u/dansp/egnn/rf_results/rf_sweep_packed_v2_02_01_2026'

TARGETS = ['energy', 'bandgap', 'volume', 'geometry', 'all_scalar']
METRICS = ['smape', 'asinh', 'mae']


def main():
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    # --- CRITICAL ENVIRONMENT SETUP ---
    # We copy the current environment (with VENV) but force low-level threads to 1.
    # We rely ENTIRELY on sklearn's n_jobs for parallelism.
    # This prevents 15 jobs from spawning 72 threads each (which would crash the node).
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'
    env['OPENBLAS_NUM_THREADS'] = '1'

    procs = []
    print(f"Starting batch run on node: {os.environ.get('SLURMD_NODENAME', 'Unknown')}")

    for target in TARGETS:
        for metric in METRICS:
            
            job_name = f"{target}_{metric}"
            job_dir = os.path.join(BASE_OUTPUT_DIR, job_name)
            os.makedirs(job_dir, exist_ok=True)
            
            cmd = [
                PYTHON_EXEC, SCRIPT_PATH,
                f"--data_file={DATA_FILE}",
                f"--output_dir={job_dir}",
                f"--target_key={target}",
                f"--metric_key={metric}",
                "--n_iter=100",
                "--n_jobs=4"  # 15 jobs * 4 cores = 60 cores usage
            ]
            
            print(f"Launching {job_name} -> Logs in {job_dir}")
            
            # Redirect specific job output to its own folder
            with open(f"{job_dir}/run.out", "w") as out, open(f"{job_dir}/run.err", "w") as err:
                p = subprocess.Popen(cmd, stdout=out, stderr=err, env=env)
                procs.append(p)

    print(f"All {len(procs)} jobs launched. Waiting for completion...")
    
    for p in procs:
        p.wait()
        
    print("All jobs completed successfully.")

if __name__ == "__main__":
    main()
