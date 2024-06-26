

import numpy as np
import toml
import os
import pathlib
import pandas as pd
import analysis.plot_paths_v9_batch256 as baseline_info
from analysis.log2data import extract_from_single_run

def get_lbl_name(baseline_name, datasetname):
    bn = ""
    if baseline_name == "td3_bc":
        bn = "TD3BC"
    elif baseline_name == "cqlsac_offline":
        bn = "CQL"
    elif baseline_name == "awac_offline":
        bn = "AWAC"
    elif baseline_name == "iql_offline":
        bn = "IQL"

    if datasetname == "expert" or datasetname == "medium":
        return bn + "_" + datasetname

    if datasetname == "medexp":
        return bn + "_" + "medium_expert"
    
    if datasetname == "medrep":
        return bn + "_" + "medium_replay"

    raise "datasetname not found"

def get_baseline_param(baseline_name, envname, datasetname):
    bn_lbl = get_lbl_name(baseline_name, datasetname)
    info = None
    print(bn_lbl)
    if envname == "Ant":
        info = next(v for v in baseline_info.at if v["label"] == bn_lbl)
    elif envname == "HalfCheetah":
        info = next(v for v in baseline_info.hc if v["label"] == bn_lbl)
    elif envname == "Hopper":
        info = next(v for v in baseline_info.hp if v["label"] == bn_lbl)
    elif envname == "Walker2d":
        info = next(v for v in baseline_info.wk2d if v["label"] == bn_lbl)
    else:
        raise "Env name not found"

    return info["control"][1][0]

def get_baseline_data(baseline_name, envname, datasetname):
    dir_name = "data/baselines/{}/{}/data_{}/sweep".format(envname.lower(),
                                                           baseline_name,
                                                           datasetname)

    param_setting = get_baseline_param(
        baseline_name, envname, datasetname)

    return extract_baseline_data(dir_name, param_setting)

def extract_baseline_data(dir_name, param_setting):
    data = [extract_from_single_run(
        dir_name + "/" + "{}_run/{}_param_setting/log".format(
            r, param_setting),
        "normalized_return") for r in range(5)]
    return data

def get_configs_and_data(dir_name):
    param_fldrs = os.listdir(dir_name)
    pd = {}
    for fldr in param_fldrs:
        runs = filter(
            lambda f: pathlib.Path(f).suffix != ".toml",
            os.listdir(pathlib.Path(dir_name, fldr)))
        for r in runs:
            # get params
            p = pathlib.Path(dir_name, fldr, r)
            p_eval = pathlib.Path(p, "evaluations.npy")
            p_toml = pathlib.Path(dir_name, fldr, r + ".toml")
            params[p] = {}
            params[p]["params"] = toml.load(p_toml)
            try:
                params[p]["data"] = np.load(p_eval)
            except FileNotFoundError:
                params[p]["data"] = None
    return params


def compress_configs_and_data(config_and_data):
    ks = list(config_and_data.keys())
    # l = []
    diff_ks = set()
    for idx, k1 in enumerate(ks):
        for k2 in ks[(idx+1):]:
            d = dict(set(config_and_data[k1]["params"].items()) ^
                     set(config_and_data[k2]["params"].items()))
            for k in d.keys():
                diff_ks.add(k)
    diff_dict = {k: set() for k in diff_ks}
    for k, v in config_and_data.items():
        for p in diff_ks:
            diff_dict[p].add(v["params"][p])
    return diff_dict


def analyze_data(dir_name, subset_func=None):
    param_fldrs = os.listdir(dir_name)
    p_a_ds = []
    for fldr in param_fldrs:
        runs = filter(
            lambda f: pathlib.Path(f).suffix != ".toml",
            os.listdir(pathlib.Path(dir_name, fldr)))
        p_toml = pathlib.Path(dir_name, fldr, "run-0" + ".toml")
        row = toml.load(p_toml)
        row.pop("run", "seed")
        row["runs"] = []
        row["seeds"] = []
        # row["data"] = []
        data = []
        kldiv_data = []
        for r in runs:
            pth = pathlib.Path(dir_name, fldr, r)
            pth_eval = pathlib.Path(pth, "evaluations.npy")
            pth_kldiv = pathlib.Path(pth, "kldiv_eval.npy")
            pth_toml = pathlib.Path(dir_name, fldr, r + ".toml")
            prm = toml.load(pth_toml)
            row["runs"].append(prm["run"])
            row["seeds"].append(prm["seed"])
            # try:
            data.append(np.load(pth_eval))
            kldiv_data.append(np.load(pth_kldiv))
            # except FileNotFoundError:
            #     data.append(None)

        if any(d is not None for d in data):
            _data = [d for d in data if d is not None]
            row["arr"] = _data
            row["kldiv"] = kldiv_data
            row["mean_arr"] = np.mean(_data, axis=0)
            row["var_arr"] = np.var(_data, axis=0)
            row["stderr_arr"] = np.sqrt(row["var_arr"]/len(_data))
            row["end_max"] = np.max([np.mean(d[51:]) for d in _data])
            row["end_min"] = np.min([np.mean(d[51:]) for d in _data])
            row["end_mean"] = np.mean(row["mean_arr"][51:])
            row["end_var"] = np.var([np.mean(d[51:]) for d in _data])
            row["end_stderr"] = np.sqrt(row["end_var"]/len(_data))
            row["all_max"] = np.max([np.mean(d) for d in _data])
            row["all_min"] = np.min([np.mean(d) for d in _data])
            row["all_mean"] = np.mean([np.mean(d) for d in _data])
            row["all_var"] = np.var([np.mean(d) for d in _data])
            row["all_stderr"] = np.sqrt(row["all_var"]/len(_data))
            row["beg_mean"] = np.mean([np.mean(d[0:10]) for d in _data])
            row["beg_var"] = np.var([np.mean(d[0:10]) for d in _data])
            row["beg_stderr"] = np.sqrt(row["beg_var"]/len(_data))
            row["beg_50_mean"] = np.mean([np.mean(d[0:50]) for d in _data])
            row["beg_50_var"] = np.var([np.mean(d[0:50]) for d in _data])
            row["beg_50_stderr"] = np.sqrt(row["beg_50_var"]/len(_data))
        p_a_ds.append(row)

    df = pd.DataFrame(p_a_ds)
    if subset_func is not None:
        df = subset_func(df)
    return df

def sensitivity_curve(dir_name,
                      sort_perf_by,
                      group_by,  # this is the sensitivity parameters
                      ):

    df = analyze_data(dir_name)
    sdf = transform_best_over(df, sort_perf_by, group_by)
    return sdf


def transform_best_over(df, sort_perf_by, group_by=None):
    if group_by is None:
        sdf = df.sort_values([sort_perf_by]).tail(1)
    else:
        sdf = df.sort_values([sort_perf_by]).groupby(group_by).tail(1).sort_values(group_by)
    return sdf

def transform_to_json(dir_name, csv_save_name, sort_perf_by, group_by):
    # Honestly it would be really nice to grab data and compress it.
    # param_fldrs = os.listdir(dir_name)
    df = analyze_data(dir_name)
    sdf = df.sort_values([sort_perf_by]).groupby(group_by).tail(1).sort_values(group_by)
    sdf.to_json(csv_save_name)

def transform_to_csv(dir_name, csv_save_name, sort_perf_by, group_by):
    # Honestly it would be really nice to grab data and compress it.
    # param_fldrs = os.listdir(dir_name)
    df = analyze_data(dir_name)
    sdf = df.sort_values([sort_perf_by]).groupby(group_by).tail(1).sort_values(group_by)
    sdf.to_csv(csv_save_name, index=False)




