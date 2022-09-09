import os

import numpy as np
import pandas as pd


def get_df(path):
    dfs = {}
    for df in os.listdir(path):
        try:
            dfs[df.split('.')[0]] = pd.read_csv(path + f'/{df}', sep='|')
        except:
            continue
    return dfs

