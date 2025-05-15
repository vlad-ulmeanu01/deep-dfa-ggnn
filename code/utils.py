import torch.nn.functional as F
import numpy as np
import itertools
import psutil
import torch
import json
import os

DEFAULT_SEED = 42

# BASE_SERVER_FOLDER = (
#     "/home/vlad/Desktop/Probleme/deep_dfa_local/" if os.uname().nodename == "vlad-TM1701"
#     else "/export/home/acs/stud/v/vlad_adrian.ulmeanu/Probleme/deep_dfa/"
# )

DSET_TYPES = ["train", "test"]

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

TEST_FOLDER = "/home/alexandru/Desktop/Master-IA/An1-sem2/proiect-SSL-NLP/deep-dfa-ggnn/dummy_data"
REAL_FOLDER = "/home/alexandru/Desktop/Master-IA/An1-sem2/proiect-SSL-NLP/deep-dfa-ggnn/bigvul"

TOP_K_PROP_NAMES = 1000 # cate API/datatype/literal/operator sa tinem (top cele mai frecvente)
CNT_CATS = 4 # API/datatype/literal/operator

AGG_HIDDEN_SIZE = 256
UPD_HIDDEN_SIZE = 32
GGNN_NUM_ITERATIONS = 5 # pasi de message passing per fiecare graf intr-un train loop.
GGNN_NUM_EPOCHS = 10 # 100

GGNN_DEBUG_SAVE_EVERY = 5


def print_used_memory():
    free, total = torch.cuda.mem_get_info(DEVICE)
    print(f"Host memory used: {round(psutil.Process().memory_info().rss / (2 ** 30), 3)} GB.\nDevice memory used: {round((total - free) / (2 ** 30), 3)} GB.", flush = True)

