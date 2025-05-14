import torch.nn.functional as F
import numpy as np
import random
import torch
import time
import json

import loader
import design
import utils

random.seed(utils.DEFAULT_SEED)

def run_net(agg, upd, gap, mlp, dsets, criterion, optimizer, metrics, dset_type):
    with torch.set_grad_enabled(dset_type == "train"), torch.autograd.set_detect_anomaly(True):
        all_graph_indices = list(range(len(dsets[dset_type])))
        random.shuffle(all_graph_indices)

        all_graph_losses = []
        for bs_ind in all_graph_indices:
            graph_id = dsets[dset_type].graph_ids[bs_ind]
            graph_embeddings = dsets[dset_type][bs_ind]

            print(f"{graph_id = }")

            for ggnn_iter in range(1, utils.GGNN_NUM_ITERATIONS + 1):
                parents_in_batch = []

                for nod in range(dsets[dset_type].cnt_nodes_per_graph[graph_id]):
                    parents_in = torch.mean(torch.stack([graph_embeddings[pa] for pa in dsets[dset_type].graphs_parent_list[graph_id][nod] + [nod]]), dim = 0)
                    parents_in_batch.append(parents_in)

                # [node_count (variabil de la un graf la altul), TOP_K_PROP_NAMES * CNT_CATS].
                parents_in_batch = torch.stack(parents_in_batch)

                parents_out_agg_batch = agg(parents_in_batch)
                parents_out_batch = upd(parents_out_agg_batch)

                for nod in range(dsets[dset_type].cnt_nodes_per_graph[graph_id]):
                    graph_embeddings[nod] = parents_out_batch[nod]

                print(f"finished {ggnn_iter = }")

            mean_embedding = gap(parents_out_batch) # vrem graph_embeddings dupa GGNN_NUM_ITERATIONS runde de msg passing = parents_out, care ramane din for.
            vuln_score = mlp(mean_embedding)
            
            want_vuln_score = torch.ones(1) if dsets[dset_type].vuln_verdict[graph_id] == "vuln" else torch.zeros(1)

            loss = criterion(torch.cat([vuln_score, 1 - vuln_score]), torch.cat([want_vuln_score, 1 - want_vuln_score]))

            if dset_type == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            all_graph_losses.append(loss.item())

    metrics[dset_type]["loss"].append(np.mean(all_graph_losses))


def main():
    runid = int(time.time())

    dsets = {dset_type: loader.Dataset(dset_type) for dset_type in utils.DSET_TYPES}

    agg, upd, gap, mlp = design.Aggregator(), design.Updater(), design.GlobalAttnPool(), design.MLP()

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(list(agg.parameters()) + list(upd.parameters()) + list(gap.parameters()) + list(mlp.parameters()))

    metrics = {dset_type: {"loss": []} for dset_type in utils.DSET_TYPES} # , "accuracy": [], "f1": []

    for epoch in range(1, utils.GGNN_NUM_EPOCHS + 1):
        for dset_type in utils.DSET_TYPES:
            run_net(agg, upd, gap, mlp, dsets, criterion, optimizer, metrics, dset_type)

        if epoch % utils.GGNN_DEBUG_SAVE_EVERY == 0 or epoch == utils.GGNN_EPOCH_CNT:
            torch.save(agg.state_dict(), f"../ggnn_saves/agg_{runid}_{epoch}.pt")
            torch.save(upd.state_dict(), f"../ggnn_saves/upd_{runid}_{epoch}.pt")
            torch.save(gap.state_dict(), f"../ggnn_saves/gap_{runid}_{epoch}.pt")
            torch.save(mlp.state_dict(), f"../ggnn_saves/mlp_{runid}_{epoch}.pt")

            for dset_type in utils.DSET_TYPES:
                with open(f"../ggnn_logs/metrics_log_{runid}_{dset_type}_{epoch}.json", "w") as fout:
                    json.dump(metrics[dset_type], fout, indent = 4)


if __name__ == "__main__":
    main()
