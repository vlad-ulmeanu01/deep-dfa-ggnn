import torch.nn.functional as F
import numpy as np
import random
import torch
import time
import json
import tqdm

import loader
import design
import utils
from sklearn.metrics import precision_score, recall_score, f1_score

random.seed(utils.DEFAULT_SEED)

def run_net(agg, upd, gap, mlp, dsets, criterion, optimizer, metrics, epoch, dset_type):
    print(f"Started {epoch = }, {dset_type = }.")

    with torch.set_grad_enabled(dset_type == "train"), torch.autograd.set_detect_anomaly(True):
        all_graph_indices = list(range(len(dsets[dset_type])))
        random.shuffle(all_graph_indices)

        all_graph_losses, y_true, y_pred = [], [], []

        for bs_ind in tqdm.tqdm(all_graph_indices):
            graph_id = dsets[dset_type].graph_ids[bs_ind]
            graph_embeddings = dsets[dset_type][bs_ind]

            for ggnn_iter in range(1, utils.GGNN_NUM_ITERATIONS + 1):
                parents_in_batch = []

                for nod in range(dsets[dset_type].cnt_nodes_per_graph[graph_id]):
                    parents_in = torch.mean(torch.stack([graph_embeddings[pa] for pa in dsets[dset_type].graphs_parent_list[graph_id][nod]]), dim = 0)
                    parents_in_batch.append(parents_in)

                # [node_count (variabil de la un graf la altul), TOP_K_PROP_NAMES * CNT_CATS].
                parents_in_batch = torch.stack(parents_in_batch)

                graph_embeddings = upd(agg(parents_in_batch)) # rescriem direct output-ul din UPD peste graph_embeddings.

            mean_embedding = gap(graph_embeddings) # vrem graph_embeddings dupa GGNN_NUM_ITERATIONS runde de msg passing = parents_out, care ramane din for.
            vuln_score = mlp(mean_embedding)
            
            want_vuln_score = torch.ones(1, device = utils.DEVICE) if dsets[dset_type].vuln_verdict[graph_id] == "vuln" else torch.zeros(1, device = utils.DEVICE)

            loss = criterion(torch.cat([vuln_score, 1 - vuln_score]), torch.cat([want_vuln_score, 1 - want_vuln_score]))

            if dset_type == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            all_graph_losses.append(loss.item())
            y_true.append(int(dsets[dset_type].vuln_verdict[graph_id] == "vuln"))
            y_pred.append(int(vuln_score.round().item()))

    metrics[dset_type]["loss"].append(np.mean(all_graph_losses))
    metrics[dset_type]["f1"].append(f1_score(y_true, y_pred, zero_division = 0.0))
    metrics[dset_type]["precision"].append(precision_score(y_true, y_pred, zero_division = 0.0))
    metrics[dset_type]["recall"].append(recall_score(y_true, y_pred, zero_division = 0.0))


def main():
    runid = int(time.time())

    dsets = {dset_type: loader.Dataset(dset_type) for dset_type in utils.DSET_TYPES}

    agg, upd, gap, mlp = design.Aggregator(), design.Updater(), design.GlobalAttnPool(), design.MLP()

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(list(agg.parameters()) + list(upd.parameters()) + list(gap.parameters()) + list(mlp.parameters()))

    metrics = {dset_type: {"loss": [], "f1": [], "precision": [], "recall": []} for dset_type in utils.DSET_TYPES}

    for epoch in range(1, utils.GGNN_NUM_EPOCHS + 1):
        for dset_type in utils.DSET_TYPES:
            run_net(agg, upd, gap, mlp, dsets, criterion, optimizer, metrics, epoch, dset_type)

        if epoch % utils.GGNN_DEBUG_SAVE_EVERY == 0:
            torch.save(agg.state_dict(), f"../ggnn_saves/agg_{runid}_{epoch}.pt")
            torch.save(upd.state_dict(), f"../ggnn_saves/upd_{runid}_{epoch}.pt")
            torch.save(gap.state_dict(), f"../ggnn_saves/gap_{runid}_{epoch}.pt")
            torch.save(mlp.state_dict(), f"../ggnn_saves/mlp_{runid}_{epoch}.pt")

            for dset_type in utils.DSET_TYPES:
                with open(f"../ggnn_logs/metrics_log_{runid}_{dset_type}_{epoch}.json", "w") as fout:
                    json.dump(metrics[dset_type], fout, indent = 4)


if __name__ == "__main__":
    main()
