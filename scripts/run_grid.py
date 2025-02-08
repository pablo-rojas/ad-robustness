import os
import json

def save_config(experiment_name, dataset, epsilon):
    config = {
        "experiment_name": experiment_name,
        "dataset": dataset,
        "patch_size": 17,
        "num_students": 10,
        "train": {
            "steps": 100000
        },
        "test": {
            "attacker": {
                "epsilon": epsilon,
                "constraint": "inf",
                "targeted": False,
                "iterations": 100,
                "step_size": 0.01
            },
            "samples": 5000,
            "save": False
        }
    }
    
    with open("cfg/temp_config.json", "w") as json_file:
        json.dump(config, json_file, indent=4)
        
datasets = ["cifar"]
epsilons = [0.05, 0.1, 0.3]

total_experiments = len(datasets) * len(epsilons)
i = 0
for dataset in datasets:
    for epsilon in epsilons:
        experiment_name = f"{dataset}v2_epsilon_{epsilon}"

        print (f"Running experiment {experiment_name}, experiment number {i} out of {total_experiments}") 
        save_config(experiment_name, dataset, epsilon)
        os.system ("python train.py --config cfg/temp_config.json")
        os.system ("python test.py --config cfg/temp_config.json")
        i +=1
