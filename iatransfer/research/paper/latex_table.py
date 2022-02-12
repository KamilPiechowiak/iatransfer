import pandas as pd
import argparse

def get_name(s: str):
    names_to_skip = ["Transfer", "Matching", "Standardization", "Score"]
    for name in names_to_skip:
        s = s.replace(name, "")
    c = s.split("-")
    if len(c) == 2:
        return "(Graph, Argmax, AutoEncoder, Clip)"
    elif len(c) == 3:
        return f"({c[2]}, {c[1]}, Shape, {c[0]})"
    else:
        return f"({c[2]}, {c[1]}, {c[3]}, {c[0]})"

def get_models(s: str):
    if s == "mean":
        res = s
    elif s.startswith("mean_"):
        res = "mean'"
    else:
        c = s.split("_from_")
        res = f"({c[1]}, {c[0]})"
    return f"\\rotatebox[origin=c]{{-90}}{{{res}}}"

def get_cell_value(x, x_max):
    if abs(x-x_max) < 0.01:
        return f"\\textbf{{{x}}}"
    else:
        return str(x)

def generate_table(data: pd.DataFrame):
    # print(data.index)
    # print(data)
    data.sort_values(by='mean', ascending=False, inplace=True)
    data = round(data, 2)
    columns_max = data.max()
    print("\\hline\n")
    print(" & ".join([""] + [get_models(s) for s in data.columns]), end="\\\\\n")
    print("\\hline\n")
    for row in data.itertuples():
        print(get_name(row[0]), end=" & ")
        print(" & ".join([get_cell_value(val, max_val[1]) for val, max_val in zip(row[1:], columns_max.iteritems())]), end="\\\\\n\\hline\n")

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', required=True,
    help="Path to csv")
args = parser.parse_args()
path = args.path
data = pd.read_csv(path, index_col=0)
generate_table(data)