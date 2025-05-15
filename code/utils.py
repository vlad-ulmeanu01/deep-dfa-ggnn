import torch.nn.functional as F
import numpy as np
import itertools
import psutil
import torch
import json
import os

DEFAULT_SEED = 42
DSET_TYPES = ["train", "test"]
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
NODENAME = os.uname().nodename
NODENAME = "grid.pub.ro" if NODENAME.endswith("grid.pub.ro") else NODENAME


TEST_FOLDER = {
    "vlad-TM1701": "/home/vlad/Desktop/Probleme/deep_dfa_local/local_implementation/dummy_data",
    "alexandru": "/home/alexandru/Desktop/Master-IA/An1-sem2/proiect-SSL-NLP/deep-dfa-ggnn/dummy_data",
    "grid.pub.ro": "/export/home/acs/stud/v/vlad_adrian.ulmeanu/Probleme/deep-dfa-ggnn/dummy_data"
}[NODENAME]

REAL_FOLDER = {
    "vlad-TM1701": "/home/vlad/Desktop/Probleme/deep_dfa_local/DDFA/storage/processed/bigvul",
    "alexandru": "/home/alexandru/Desktop/Master-IA/An1-sem2/proiect-SSL-NLP/deep-dfa-ggnn/bigvul",
    "grid.pub.ro": "/export/home/acs/stud/v/vlad_adrian.ulmeanu/Probleme/deep_dfa/DDFA/storage/processed/bigvul"
}[NODENAME]


LOADER_NUM_SAMPLES = 1000
NONVULN_VULN_RAP = 3

TOP_K_PROP_NAMES = 1000 # cate API/datatype/literal/operator sa tinem (top cele mai frecvente)
CNT_CATS = 4 # API/datatype/literal/operator

AGG_HIDDEN_SIZE = 256
UPD_HIDDEN_SIZE = 32
GGNN_NUM_ITERATIONS = 5 # pasi de message passing per fiecare graf intr-un train loop.
GGNN_NUM_EPOCHS = 100
BATCH_SIZE = 128

GGNN_DEBUG_SAVE_EVERY = 5


def print_used_memory():
    print(f"Host memory used: {round(psutil.Process().memory_info().rss / (2 ** 30), 3)} GB.")
    if NODENAME != "alexandru":
        free, total = torch.cuda.mem_get_info(DEVICE)
        print(f"Device memory used: {round((total - free) / (2 ** 30), 3)} GB.", flush = True)

def focal_loss(y_pred: torch.tensor, y_truth: torch.tensor):
    return -(y_truth * torch.log(y_pred + 1e-10) * (1-y_pred) ** 2 + (1 - y_truth) * torch.log(1-y_pred + 1e-10) * y_pred ** 2).mean()

