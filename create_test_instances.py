import json
import os 
import numpy as np 

np.random.seed(42)

instacne_dir = os.path.join("textmapworld", "textmapworld_main", "in")

with open(os.path.join(instacne_dir, "instances.json"), 'r') as f:
    instance_data = json.load(f)

exps = instance_data["experiments"]

test_exps = []
for exp in exps:
    exp_dict = {}
    exp_dict["name"] = exp["name"]
    test_game_instance = []

    test_game_instance.append(exp["game_instances"][np.random.choice(len(exp["game_instances"]))])
    test_game_instance[0]["game_id"] = 0
    exp_dict["game_instances"] = test_game_instance
    test_exps.append(exp_dict)

test_data = {
    "experiments": test_exps
}

with open(os.path.join(instacne_dir, "test.json"), 'w') as f:
    json.dump(test_data, f, indent=4)
