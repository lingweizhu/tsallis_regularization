import os

import analysis.data_analysis as da
from analysis.log2data import extract_from_single_run
import matplotlib.pyplot as plt
import numpy as np
import tol_colors as tc


def plot_data(hans_dir, hans_param_setting, my_data, show=False, save=None, title=""):
    hans_hc_y, hans_hc_error = prep_hans_data(hans_dir, hans_param_setting)
    y, error = prep_my_data(my_data)

    colorset = tc.colorsets["bright"]
    x = range(len(y))

    plt.plot(x, y, color=colorset[1], label="Tsallis")
    plt.fill_between(x, y-error, y+error, color=colorset[1], alpha=0.4)

    plt.plot(range(len(hans_hc_y)), hans_hc_y, color=colorset[2], label="InAC")
    plt.fill_between(range(len(hans_hc_y)),
                     hans_hc_y - hans_hc_error,
                     hans_hc_y + hans_hc_error, color=colorset[2], alpha=0.4)
    plt.legend()
    plt.title(title)
    if save != None:
        plt.savefig(save)
        plt.clf()
    if show:
        plt.show()

def plot_kldiv(my_data, show=False, save=None, title=""):
    colorset = tc.colorsets["bright"]
    y, error = prep_my_data(my_data)
    x = range(len(y))
    plt.plot(x, y, color=colorset[1], label="Tsallis")
    plt.fill_between(x, y-error, y+error, color=colorset[1], alpha=0.4)

    plt.legend()
    plt.title(title)
    if save is not None:
        plt.savefig(save)
        plt.clf()
    if show:
        plt.show()


def get_dirname(base_dir, envname, dataset):
    return os.path.join(
        base_dir,
        "env_name-{}".format(envname),
        "dataset-{}".format(dataset))


def plot_line(y, error, color, label=""):
    x = range(len(y))
    plt.plot(x, y, color=color, label=label)
    plt.fill_between(x, y-error, y+error, color=color, alpha=0.4)


def plot_hans_line(dir_name, param_setting, color, label="InAC"):
    y, error = prep_hans_data(dir_name, param_setting)
    x = range(len(y))
    plot_line(y, error, color=color, label=label)


def prep_hans_data(dir_name, param_setting):
    hans_hc = [extract_from_single_run(
        dir_name + "/" + "{}_run/{}_param_setting/log".format(r, param_setting),
        "normalized_return") for r in range(5)]
    print(extract_from_single_run(dir_name + "/" + "0_run/{}_param_setting/log".format(param_setting), "tau"))
    hans_smoothed_hc_y = [np.mean(
        np.lib.stride_tricks.sliding_window_view(d, 10),
        axis=1) for d in hans_hc]
    y = np.mean(hans_smoothed_hc_y, axis=0)
    error = 1.97*(np.sqrt(np.var(hans_smoothed_hc_y, axis=0))/4)
    return y, error


def plot_mydata_line(dirname, color, label):
    df_tkl = da.analyze_data(dirname)
    sdf_tkl = da.transform_best_over(df_tkl, "end_mean")
    y, error = prep_my_data(sdf_tkl[["arr"]].iloc[0][0][0:5])
    plot_line(y, error, color, label=label)


def prep_my_data(my_data):
    smoothed_y = [np.mean(np.lib.stride_tricks.sliding_window_view(d, 10), axis=1) for d in my_data]
    y, error = np.mean(smoothed_y, axis=0), 1.97*(np.sqrt(np.var(smoothed_y, axis=0))/4)
    return y, error


def plot_eval_env(base_tkl_dir, base_hans_dir, hans_param_setting, envname, dataset, tau=0.1, save=None):
    tkl_dirname = os.path.join(
        base_tkl_dir, "env_name-{}".format(envname), "dataset-{}".format(dataset))
    df_tkl = da.analyze_data(tkl_dirname)
    sdf_tkl = da.transform_best_over(df_tkl, "end_mean")
    print(sdf_tkl["tau"])
    
    plt_data = sdf_tkl[["arr"]].iloc[0]

    hans_dir = base_hans_dir.format(envname, dataset)
    plot_data(hans_dir, hans_param_setting, plt_data[0][0:5], save=save, title=envname + " " + dataset)

def plot_eval_everything(save_dir):
    dataset_name = "medexp"
    for (envname, hps) in [("Hopper", 4), ("HalfCheetah", 2), ("Walker2d", 2), ("Ant", 4)]:
        plot_eval_env("data/tsallis_klinac/",
                      "data/hans_data/after_fix/{}/in_sample_ac/data_{}/sweep/",
                      hps, envname, dataset_name,
                      save=os.path.join(save_dir, envname + "_" + dataset_name + ".pdf"))

    dataset_name = "medrep"
    for (envname, hps) in [("Hopper", 1), ("HalfCheetah", 1), ("Walker2d", 1), ("Ant", 1)]:
        plot_eval_env("data/tsallis_klinac/",
                      "data/hans_data/after_fix/{}/in_sample_ac/data_{}/sweep/",
                      hps, envname, dataset_name,
                      save=os.path.join(save_dir, envname + "_" + dataset_name + ".pdf"))


def plot_eval_env_multiple(base_tkl_dirs_names, base_hans_dir, hans_param_setting, envname, dataset, save=None):

    colorset = tc.colorsets["bright"]
    i = 0
    for (base_tkl_dir, name) in base_tkl_dirs_names:
        plot_mydata_line(
            get_dirname(base_tkl_dir, envname, dataset),
            color=colorset[i], label=name
        )
        i += 1

    plt.legend()
    plt.title(envname + " " + dataset)

    plt.show()


def plot_kldiv_env(base_tkl_dir, envname, dataset, tau=0.1, save=None):
    tkl_dirname = os.path.join(
        base_tkl_dir, "env_name-{}".format(envname), "dataset-{}".format(dataset))
    df_tkl = da.analyze_data(tkl_dirname)
    sdf_tkl = da.transform_best_over(df_tkl, "end_mean")
    print(sdf_tkl["tau"])

    plt_data = sdf_tkl[["kldiv"]].iloc[0]
    plot_kldiv(plt_data[0][0:5], save=save, title=envname + " " + dataset)
