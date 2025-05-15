import pandas as pd
import numpy as np
import random
import json

random.seed(42)

REAL_FOLDER = "/home/alexandru/Desktop/Master-IA/An1-sem2/proiect-SSL-NLP/deep-dfa-ggnn/bigvul"
NUM_SAMPLES = 1000
TRAIN_RAP = 0.8
NONVULN_RAP = 0.9


def main():
    df_nodes = pd.read_csv(f"{REAL_FOLDER}/nodes.csv")
    df_edges = pd.read_csv(f"{REAL_FOLDER}/edges.csv")

    nonvuln_graph_ids = list(map(int, set(df_nodes[df_nodes["vuln"] == 0]["graph_id"])))
    vuln_graph_ids = list(map(int, set(df_nodes[df_nodes["vuln"] == 1]["graph_id"])))

    print(f"{len(nonvuln_graph_ids) = }, {len(vuln_graph_ids) = }", flush = True)

    train_test_ids = {"train": [], "test": []}
    selected_ids = set()
    for _ in range(NUM_SAMPLES):
        nonvuln = random.random() < NONVULN_RAP
        graph_ids = nonvuln_graph_ids if nonvuln else vuln_graph_ids
        
        z = random.choice(graph_ids)
        while z in selected_ids:
            z = random.choice(graph_ids)
        selected_ids.add(z)

        train_test_ids["train" if random.random() < TRAIN_RAP else "test"].append((z, "nonvuln" if nonvuln else "vuln"))
        
    df_nodes[df_nodes["graph_id"].isin(selected_ids)].to_csv(f"{REAL_FOLDER}/nodes_{NUM_SAMPLES}_samples.csv", index = False)
    df_edges[df_edges["graph_id"].isin(selected_ids)].to_csv(f"{REAL_FOLDER}/edges_{NUM_SAMPLES}_samples.csv", index = False)

    with open(f"{REAL_FOLDER}/train_test_split_{NUM_SAMPLES}_samples_new.json", "w") as fout:
        json.dump(train_test_ids, fout, indent = 4)


if __name__ == "__main__":
    main()
