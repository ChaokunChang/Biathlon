import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import math

from tap import Tap

PJNAME = "Biathlon"


class EvalArgs(Tap):
    home_dir: str = "./cache"
    filename: str = "evals.csv"
    loading_mode: int = 0
    ncores: int = 1
    beta_of_all: bool = False


def load_df(args: EvalArgs) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(args.home_dir, args.filename))
    df['BD:Others'] = df['avg_latency'] - df['BD:AFC'] - df['BD:AMI'] - df['BD:Sobol']

    # special handling for profiling results
    def handler_soboltime(df: pd.DataFrame) -> pd.DataFrame:
        # AMI and Sobol share some computation
        # update BD:Sobol as max(BD:Sobol - BD:AMI, 0)
        df["BD:Sobol"] = df["BD:Sobol"] - df["BD:AMI"]
        df["BD:Sobol"] = df["BD:Sobol"].apply(lambda x: max(x, 0))
        old_lat = df["avg_latency"]
        df["avg_latency"] = df['BD:AFC'] + df['BD:AMI'] + df['BD:Sobol'] + df['BD:Others']
        df['speedup'] = (old_lat * df['speedup']) / df['avg_latency']
        return df

    def handler_filter_ncfgs(df: pd.DataFrame) -> pd.DataFrame:
        # delte rows with ncfgs = 2
        df = df[df["ncfgs"] != 2]
        # df = df[df["ncfgs"] != 50]
        return df

    def handler_loading_mode(df: pd.DataFrame) -> pd.DataFrame:
        # keep only the rows with the specified mode
        loading_mode = args.loading_mode
        df = df[df["loading_mode"] == loading_mode]
        return df

    # df = handler_soboltime(df)
    df = handler_filter_ncfgs(df)
    df = handler_loading_mode(df)
    return df
