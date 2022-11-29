import numpy as np
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
    x = np.stack(df.fasta.map(lambda st: string_to_ints(st, trim_length)).values)
    return y, x
