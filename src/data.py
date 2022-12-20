import numpy as np
import torch
import torch.nn.functional as F
from pandas import read_csv
from sklearn.model_selection import train_test_split

# Encoding of FASTA characters to integers. Encoding starts at 1.
residues = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]

mapping = dict(zip(residues, range(1, len(residues) + 1)))


def string_to_ints(s, l, padding='post'):
    """Build a list of integers from a string which is then trimmed or padded with 0 to have length `l`."""

    a = [*map(mapping.get, s)]
    n = len(a)

    if n < l:
        if padding == 'pre':
            a = [0] * ((l - n)) + a
        elif padding == 'mid':
            a = [0] * ((l - n) // 2) + a + [0] * ((l - n) // 2)
            if len(a) < l: #odd length case
                a += [0]
        elif padding == 'post':
            a += [0] * ((l - n))
    else:
        a = a[:l]
        
    return np.array(a)


def load_data(data_path, device, trim_length):
    df = read_csv(data_path)
    y = df.solubility.values
    x = np.stack(df.fasta.map(lambda s: string_to_ints(s, trim_length, 'mid')).values)
    ty = torch.tensor(y).to(device)
    tx = torch.tensor(x).to(device)
    return ty, tx


def encode_one_hot(x):
    x = F.one_hot(x.to(torch.int64)).permute(0, 2, 1).float()

    return x[:, 1:, :]


def init_data(data_path, device, config):
    y, x = load_data(data_path, device, config["max_chain_length"])
    x = encode_one_hot(x)

    d = {}
    d["x_train"], d["x_test"], d["y_train"], d["y_test"] = train_test_split(
        x, y, test_size=0.2
    )

    d["neg_pos_ratio"] = (y == 0).sum() / y.sum()

    return d
