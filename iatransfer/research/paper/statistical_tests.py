import argparse
import pandas as pd
import numpy as np
from scipy.stats import t, f

def run_correlation_tests(x: np.array, y: np.array):
    assert len(x) == len(y)
    n = len(x)
    y_mean = np.mean(y)
    r = np.corrcoef(x, y)[1,0]
    print(f"n: {n}")
    print(f"corr: {r}")
    T = r/(1-r**2)**0.5*(n-2)**0.5
    print(f"T: {T}, p-value: {1-t.cdf(T, n-2)}")
    a = np.std(y)*r/np.std(x)
    b = y_mean - np.mean(x)*a
    y_pred = a*x+b
    SSE = np.sum((y_pred-y)**2)
    SST = np.sum((y_mean-y)**2)
    SSR = np.sum((y_pred-y_mean)**2)
    print(SSE, SSR, SST)
    R_2 = SSR/SST
    print(f"R^2 {R_2}")
    F = SSR/SSE*(n-2)
    print(f"F: {F}, p-value: {1-f.cdf(F, 1, n-2)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', required=True,
        help='Path to csv file')
    args = parser.parse_args()
    df = pd.read_csv(args.path)
    run_correlation_tests(df['sim'], df['acc_ratio'])
