import numpy as np
import torch
import torch.nn.functional as F
from pandas import read_csv


class Encoding:
    """Encoding of FASTA characters to integers. Encoding starts at 1."""

    residues = [
        "",
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

    mapping = dict(zip(residues, range(len(residues))))


def string_to_ints(s, l):
    """Build a list of integers from a string which is then trimmed or padded with 0 to have length `l`."""

    a = [*map(Encoding.mapping.get, s)]
    n = len(a)
    if n < l:
        a += [0] * (l - n)
    else:
        a = a[:l]
    return np.array(a)


def load_data(data_path, trim_length):
    df = read_csv(data_path)
    y = df.solubility.values
    x = np.stack(df.fasta.map(lambda s: string_to_ints(s, trim_length)).values)

    return y, x


def encode_data(y, x, trim_length):
    y, x = torch.tensor(y), torch.tensor(x)
    x = F.one_hot(x.to(torch.int64))

    x_t = torch.zeros(y.shape[0], len(Encoding.residues), trim_length)
    for i in range(y.shape[0]):
        x_t[i] = x[i].T

    return y, x_t[:, 1:, :]

