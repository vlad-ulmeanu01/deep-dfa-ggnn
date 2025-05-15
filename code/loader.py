import torch.nn.functional as F
import pandas as pd
import numpy as np
import itertools
import torch
import json
import time

import utils


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dset_type: str):
        t_start = time.time()

        self.dset_type = dset_type

        with open(f"{utils.TEST_FOLDER}/train_test_split_1000_samples.json") as fin:
            tmp_ht = json.load(fin)

            ht_tt_split_graph_ids = {dset_type: [graph_id for graph_id, _ in tmp_ht[dset_type]] for dset_type in tmp_ht}
            tmp_vuln_verdict = {dset_type: [vuln for _, vuln in tmp_ht[dset_type]] for dset_type in tmp_ht}

            self.vuln_verdict = {graph_id: vuln for dset_type in utils.DSET_TYPES for graph_id, vuln in zip(ht_tt_split_graph_ids[dset_type], tmp_vuln_verdict[dset_type])}

        # {dgl_id}, [node_id], graph_id
        self.df_nodes = pd.read_csv(f"{utils.TEST_FOLDER}/nodes_1000_samples.csv")
        self.df_nodes = self.df_nodes[self.df_nodes["graph_id"].isin(ht_tt_split_graph_ids[dset_type])]

        # {innode}, {outnode}, [id_x], [id_y], graph_id
        self.df_edges = pd.read_csv(f"{utils.TEST_FOLDER}/edges_1000_samples.csv")
        self.df_edges = self.df_edges[self.df_edges["graph_id"].isin(ht_tt_split_graph_ids[dset_type])]

        # graph_id, [node_id], hash
        self.df_tokens = pd.read_csv(f"{utils.REAL_FOLDER}/abstract_dataflow_hash_api_datatype_literal_operator.csv")
        self.df_tokens = self.df_tokens[self.df_tokens["graph_id"].isin(ht_tt_split_graph_ids[dset_type])]

        # tine top K prop names pentru "api", "datatype", "literal", "operator" (aici sunt mult mai putin de 1000).
        with open(f"{utils.TEST_FOLDER}/used_property_names.json", "r") as fin:
            self.ht_used_property_names = json.load(fin)
        for cat in self.ht_used_property_names:
            self.ht_used_property_names[cat] = {prop: i for prop, i in zip(self.ht_used_property_names[cat], itertools.count())}

        print(f"({dset_type = }) Loaded df_nodes, df_edges, df_tokens, used_property_names. {round(time.time() - t_start, 3)} s passed.", flush = True)
        # utils.print_used_memory()

        self.cnt_nodes_per_graph = {graph_id: 0 for graph_id in ht_tt_split_graph_ids[dset_type]}
        graph_node_dgi_map = {graph_id: {} for graph_id in ht_tt_split_graph_ids[dset_type]}
        self.graph_ids = [graph_id for graph_id in ht_tt_split_graph_ids[dset_type]]

        for dgl_id, node_id, graph_id in zip(list(self.df_nodes["dgl_id"]), list(self.df_nodes["node_id"]), list(self.df_nodes["graph_id"])):
            self.cnt_nodes_per_graph[graph_id] += 1
            graph_node_dgi_map[graph_id][node_id] = dgl_id
        
        self.ht_nodes_sparse_rep = {
            graph_id: [
                {cat: [] for cat in self.ht_used_property_names} for dgl_id in range(self.cnt_nodes_per_graph[graph_id])
            ]
            for graph_id in self.graph_ids
        }

        for graph_id, node_id, hh in zip(list(self.df_tokens["graph_id"]), list(self.df_tokens["node_id"]), list(self.df_tokens["hash"])):
            dgl_id = graph_node_dgi_map[graph_id][node_id]
            hh = eval(hh)

            for cat in self.ht_used_property_names:
                for prop in hh[cat]:
                    if prop in self.ht_used_property_names[cat]:
                        i = self.ht_used_property_names[cat][prop]
                        self.ht_nodes_sparse_rep[graph_id][dgl_id][cat].append(i)
        
        self.graphs_parent_list = {
            graph_id: [[] for _ in range(self.cnt_nodes_per_graph[graph_id])]
            for graph_id in self.graph_ids
        }

        # print(self.df_edges)
        for graph_id, innode, outnode in zip(list(self.df_edges["graph_id"]), list(self.df_edges["innode"]), list(self.df_edges["outnode"])):
            try:
                innode, outnode = int(innode), int(outnode)
            except:
                continue
            self.graphs_parent_list[graph_id][outnode].append(innode)

        print(f"({dset_type = }) Computed ht_nodes_sparse_rep, graphs_parent_list. {round(time.time() - t_start, 3)} s passed.", flush = True)
        # utils.print_used_memory()


    def __len__(self):
        return len(self.graph_ids)

    @torch.no_grad()
    def __getitem__(self, ind: int):
        graph_id = self.graph_ids[ind]

        rep = [torch.zeros(utils.TOP_K_PROP_NAMES * len(self.ht_used_property_names)) for _ in range(self.cnt_nodes_per_graph[graph_id])]

        self.ht_nodes_sparse_rep[graph_id]

        for dgl_id in range(self.cnt_nodes_per_graph[graph_id]):
            for cat_ind, cat in zip(itertools.count(), self.ht_used_property_names):
                for prop_ind in self.ht_nodes_sparse_rep[graph_id][dgl_id][cat]:
                    rep[dgl_id][cat_ind * utils.TOP_K_PROP_NAMES + prop_ind] = 1

        return rep
