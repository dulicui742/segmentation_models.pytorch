import glom
import json
import pandas as pd

from typing import Any, Callable, Dict, Iterable, List, OrderedDict, Tuple


def parse_jsonline_file(infile: str) -> List[Dict]:
    info = []
    with open(infile) as IN:
        for n, line in enumerate(IN):
            temp = line.strip()
            if not temp:
                continue
            load = glom.glom(temp, lambda x: json.loads(x))
            info.append(load)
    return info

def mean_of_epoch(df_src):
    epochs = df_src["epoch"].unique()
    
    columns=df_src.columns
    df_dst = pd.DataFrame(columns=columns)
    for epoch in epochs:
        df_dst.loc[epoch] = df_src[df_src["epoch"] == epoch].mean(numeric_only=True)
    
    return df_dst


def cal_global_steps(global_step, cur_step):
#     g_s = epoch * steps_per_epoch + step
    g_s = global_step + cur_step
    return int(g_s)


if __name__ == "__main__":
    log_files = [
        ".\\output\\logs\\efficientnet-b4\\efficientnet-b4_logs_0316_174217.json",
    ]
    #####  按照epoch来绘图
    df_list = []
    for file in log_files:
        info = parse_jsonline_file(file)
        df = pd.DataFrame.from_records(info)
        df_train = df[df["status"] == "train"]
        df_list.append(df_train)
    
    df_epoch_list = []
    ex_keys = []
    for idx, df in enumerate(df_list):
        df_dst = mean_of_epoch(df)
        key = 'trail{}'. format(idx)
        df_dst.loc[:, "key"] = key
        ex_keys.append(key)
        df_epoch_list.append(df_dst)

    DF = pd.concat(df_epoch_list, keys=ex_keys)
    DFGroup = DF.groupby(['epoch','key'])

    for metric in ["bce_loss", "fscore", "recall"]: #, "fscore", "recall", "precision", "accuracy"]:
        DFGPlot = DFGroup.sum().unstack('key').plot(kind='line', y=metric, title =metric)
    

    #####  按照step来绘图
    steps_per_epoch = 396 ##根据数据及batch-size计算出来的
    df_list = []
    for file in log_files:
        info = parse_jsonline_file(file)
        df = pd.DataFrame.from_records(info)
        df_train = df[df["status"] == "train"]
        df_list.append(df_train)
    
    df_epoch_list = []
    ex_keys = []
    for idx, df in enumerate(df_list):
        df = df[df["step"] % 50 == 0]
        df = df.reset_index()
        df["total_step"] = df["step"]
        df["total_step"] = df.apply(
            lambda x: myfunction(x["epoch"]*steps_per_epoch,  x['step']), 
            axis=1
        )
        df_dst = df
        # df_dst = mean_of_epoch(df)
        key = 'trail{}'. format(idx)
        df_dst.loc[:, "key"] = key
        ex_keys.append(key)
        df_epoch_list.append(df_dst)

    DF = pd.concat(df_epoch_list, keys=ex_keys)
    DFGroup = DF.groupby(['total_step','key'])

    for metric in ["bce_loss", "fscore", "recall"]: #, "fscore", "recall", "precision", "accuracy"]:
        DFGPlot = DFGroup.sum().unstack('key').plot(kind='line', y=metric, title =metric)
    print("Happy Ending!")


