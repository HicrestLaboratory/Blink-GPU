# Run a pbs script to execute the alltoall occl benchmark on aurora with different env variables
#!/usr/bin/env python3

import argparse
import subprocess
import os

# Map collective → algorithms
#TODO: allreduce ring algorith: reduce_scatter + allgather ring. Use CCL_RS_CHUNK_COUNT and CCL_RS_MIN_CHUNK_SIZE to control pipelining on reduce_scatter phase
algorithms_map = {
    # "a2a": ["direct", "naive", "scatter"],
    "ar": ["direct", "nreduce", "rabenseifner", "ring", "double_tree", "recursive_doubling"],
    
    "a2a": ["direct"], 
}

# scaleup_modes = [0, 1] # 0: use copy engines, 1: use GPU compute kernel
scaleup_modes = [0] # 0: use copy engines, 1: use GPU compute kernel

def str2bool(v):
    return v.lower() in ("true", "1", "yes")

def main():
    parser = argparse.ArgumentParser(description="Submit PBS jobs for collectives")
    parser.add_argument("--pbs", required=True, help="Path to PBS script")
    parser.add_argument("--exe", required=True, help="Path to executable")
    parser.add_argument("--csv-log", required=True, help="CSV output directory")
    parser.add_argument("--power-log", required=True, help="Power output directory")
    parser.add_argument("--coll", required=True, help="Collective name [a2a,ar ...]")
    parser.add_argument("--is-single-node", required=True, type=str2bool)

    args = parser.parse_args()

   

    if args.coll not in algorithms_map:
        raise ValueError(f"Unknown collective: {args.coll}")

    algorithms = algorithms_map[args.coll]

    for scaleup_mode in scaleup_modes: # iterate over scaleup modes 
        if args.is_single_node:
            submit_job(
                pbs=args.pbs,
                exe=args.exe,
                csv_log=args.csv_log,
                power_log=args.power_log,
                coll=args.coll,
                scaleup_mode=scaleup_mode,
                alg_name="none"
            )
        else:
            for alg in algorithms:
                submit_job(
                    pbs=args.pbs,
                    exe=args.exe,
                    csv_log=args.csv_log,
                    power_log=args.power_log,
                    coll=args.coll,
                    scaleup_mode=scaleup_mode,
                    alg_name=alg
                )

def submit_job(pbs, exe, csv_log, power_log, coll, scaleup_mode, alg_name):
    env_vars = {
        "EXE_PATH": exe,
        "CSV_LOG_DIR": csv_log,
        "POWER_LOG_DIR": power_log,
        "COLLECTIVE": coll,
        "SCALEUP_MODE": scaleup_mode,
        "ALG_NAME": alg_name,
    }

    env_string = ",".join(f"{k}={v}" for k, v in env_vars.items())

    cmd = ["qsub", "-v", env_string, pbs]

    print("Submitting:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
