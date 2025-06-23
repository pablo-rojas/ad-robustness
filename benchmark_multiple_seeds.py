#!/usr/bin/env python3
import os
import argparse
import numpy as np
from tabulate import tabulate
from benchmark import main as bench_main
from src.misc_utils import load_config

def aggregate_seeds(config_path, num_seeds):
    base_cfg = load_config(config_path)
    all_results = {}

    for seed in range(num_seeds):
        cfg = base_cfg.copy()
        cfg['seed'] = seed
        cfg['experiment_name'] = f"{base_cfg['experiment_name']}_{seed}"
        # if you have per-seed model paths, set them here
        cfg['uninformed_students_path'] = f"models/{base_cfg['dataset']}_us_{seed}_best"

        print(f"\n--- Running seed {seed} ---")
        headers, rows = bench_main(cfg)

        # rows: [ ["Uninformed Students", p1, p2, ...],
        #         ["ACGAN", p1, p2, ...] ]
        for row in rows:
            key = row[0]
            metrics = []
            for x in row[1:]:
                try:
                    metrics.append(float(x))
                except ValueError:
                    metrics.append(x)
            all_results.setdefault(key, []).append(metrics)

    # now aggregate across seeds
    aggregated = []
    for detector, seed_lists in all_results.items():
        n_metrics = len(seed_lists[0])
        agg_row = [detector]
        for i in range(n_metrics):
            col = [sl[i] for sl in seed_lists]
            if all(isinstance(v, (int, float)) for v in col):
                arr = np.array(col, dtype=float)
                m, s = arr.mean(), arr.std()
                hdr = headers[1 + i].lower()
                agg_row.append(f"{m:.5f}±{s:.6f}")
            else:
                agg_row.append(col[0])
        aggregated.append(agg_row)

    return headers, aggregated

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark results over multiple seeds."
    )
    parser.add_argument(
        '--config', type=str, default='cfg/imagenet_benchmark.json',
        help='Path to the configuration file.'
    )
    parser.add_argument(
        '--num_seeds', type=int, default=1,
        help='Number of random seeds to run.'
    )
    args = parser.parse_args()

    headers, agg_rows = aggregate_seeds(args.config, args.num_seeds)

    # Transpose the data for better formatting
    data = [headers] + agg_rows
    transposed = list(map(list, zip(*data)))
    new_headers, *new_rows = transposed

    print("\nAggregated Results (mean ± std over seeds):\n")
    table_str = tabulate(
        agg_rows,
        headers=headers,
        tablefmt='latex_raw',
        stralign='center'
    )
    print(table_str)

    print("\n\n--- Full Results ---\n")
    table_str = tabulate(
        new_rows,
        headers=new_headers,
        tablefmt='latex_raw',
        stralign='center'
    )
    print(table_str)

    # save to disk
    base = os.path.splitext(args.config)[0]
    out_fname = f"results{base[3:]}_multi_seeds.txt"
    with open(out_fname, 'w') as f:
        f.write(table_str + "\n")
    print(f"\nSaved aggregated results to {out_fname}")

if __name__ == "__main__":
    main()
