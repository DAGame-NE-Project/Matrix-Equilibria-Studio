from cmath import nan
from ntpath import join
import numpy as np
import pandas as pd
import json
import os
import re

labels = ['algo', 'game', 'size','last_eps', 'last_ws_eps',
          'before_adjust_eps', 'before_adjust_ws_eps']
data = []
result_path = "../results"


def handle_single(filename: str) -> None:
    game, size, algo = re.match(
        r"([a-zA-Z]+)([0-9]+)_([\w]+)", filename).groups()
    print(f"handle_single on game: {game}, size: {size}, algo: {algo}")
    last_eps = np.array([])
    last_ws_eps = np.array([])
    before_adjust_eps = np.array([])
    before_adjust_ws_eps = np.array([])
    # parse every data point
    for single_file in os.listdir(os.path.join(result_path, filename)):
        if single_file.endswith(".json"):
            with open(os.path.join(result_path, filename, single_file)) as f:
                json_data = json.load(f)
                last_eps = np.append(last_eps, json_data["last_eps"][0][0])
                last_ws_eps = np.append(last_ws_eps, json_data["last_ws_eps"][0][0])
                if "before_adjust_eps" in json_data:
                    before_adjust_eps = np.append(
                        before_adjust_eps, json_data["before_adjust_eps"][0][0])
                    before_adjust_ws_eps = np.append(
                        before_adjust_ws_eps, json_data["before_adjust_ws_eps"][0][0])
    #  take the avg of these data points
    last_eps = np.mean(last_eps)
    last_ws_eps = np.mean(last_ws_eps)
    if len(before_adjust_eps) > 0:
        before_adjust_eps = np.mean(before_adjust_eps)
        before_adjust_ws_eps = np.mean(before_adjust_ws_eps)
    else:
        before_adjust_eps = None
        before_adjust_ws_eps = None
    # store the result in data
    data.append([algo, game, size, last_eps, last_ws_eps,
                before_adjust_eps, before_adjust_ws_eps])

def handle_all() -> None:
    for single_file in os.listdir(result_path):
        if os.path.isdir(os.path.join(result_path, single_file)):
            handle_single(single_file)
    df = pd.DataFrame(data, columns=labels).sort_values(by=['algo', 'game', 'size'])
    df.to_csv("data.csv", index=False)

if __name__ == '__main__':
    handle_all()