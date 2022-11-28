from pandas import read_csv
import numpy as np

# Encoding of FASTA letters into integers
# Starts at 0
class Encoding:
    residues = ['A', 'C', 'D', 'E', 'F',
        'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
        'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    mapping = dict(zip(residues, range(len(residues))))

# Build a list of integers from string
# The resulting list is trimmed/padded with 0
# so that it has length `l`
def string_to_ints(s, l=1024):
    a = [*map(Encoding.mapping.get, s)]
    n = len(a)
    if (n < l):
        a += [-1]*(l - n)
    else:
        a = a[:l]
    return np.array(a)

def load_data(data_path):
    df = read_csv(data_path)
    y = df.solubility.values
    x = np.stack(df.fasta.map(lambda st: string_to_ints(st, 400)).values)
    return (y, x)

