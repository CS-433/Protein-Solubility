from pandas import read_csv

residues = ['A', 'C', 'D', 'E', 'F',
 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

# Build a list of integers from string
# The resulting list is trimmed/padded with 0
# so that it has length `l`
def string_to_ints(s, l=1024):
    a = list(map(ord, s))
    n = len(a)
    if (n < l):
        a += [0]*(l - n)
    else:
        a = a[:l]
    return torch.tensor(a)

def load_data(data_path):
    df = read_csv(data_path)
    y = df.solubility.values
    x = np.stack(df.fasta.map(lambda st: transform(st, 400)).values)
    return (y, x)

