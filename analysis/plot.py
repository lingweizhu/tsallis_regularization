import os

import analysis.data_analysis as da
from analysis.log2data import extract_from_single_run
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tol_colors as tc


def get_color(label):
    colorset = tc.colorsets["bright"]
    color_dict = {
        "TKL": colorset[4],
        "Tsallis": colorset[0],
        "InAC": colorset[1],
        "td3_bc": colorset[3],
        "iql_offline": colorset[2],
        "awac_offline": colorset[5],
        "q=2": colorset[4],
        "q=3": colorset[5],
        "q=5": colorset[6],
        "q=10": colorset[7],
    }
    return color_dict[label]

def get_baseline_label(label):
    label_dict = {
        "td3_bc": "TD3BC",
        "iql_offline": "IQL",
        "awac_offline": "AWAC"
    }
    if label in label_dict.keys():
        return label_dict[label]
    return label

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


def plot_sens_line(x, y, error, color, label=""):
    plt.plot(x, y, color=color, label=label)
    plt.errorbar(x, y, yerr=error, color=color)
    # plt.fill_between(x, y-error, y+error, color=color, alpha=0.4)


def plot_line(y, error, color, label="", ax=plt):
    x = range(len(y))
    ax.plot(x, y, color=color, label=label)
    ax.fill_between(x, y-error, y+error, color=color, alpha=0.4)


def plot_hans_line(dir_name, param_setting, color, label="InAC", ax=plt):
    y, error = prep_hans_data(dir_name, param_setting)
    x = range(len(y))
    plot_line(y, error, color=color, label=label, ax=ax)


def prep_hans_data(dir_name, param_setting):
    hans_hc = [extract_from_single_run(
        dir_name + "/" + "{}_run/{}_param_setting/log".format(r, param_setting),
        "normalized_return") for r in range(5)]
    print(extract_from_single_run(dir_name + "/" + "0_run/{}_param_setting/log".format(param_setting), "tau"))
    hans_smoothed_hc_y = [np.mean(
        np.lib.stride_tricks.sliding_window_view(d, 10),
        axis=1) for d in hans_hc]
    y = np.mean(hans_smoothed_hc_y, axis=0)
    error = (np.sqrt(np.var(hans_smoothed_hc_y, axis=0)/len(hans_hc)))
    return y, error


def plot_mydata_line(dirname, color, label, subset_func=None, arr_name="arr", flip=False, ax=plt):
    mod = 1
    if flip:
        mod = -1
    df_tkl = da.analyze_data(dirname, subset_func)
    sdf_tkl = da.transform_best_over(df_tkl, "end_mean")
    y, error = prep_my_data(sdf_tkl[[arr_name]].iloc[0][0][0:5], flip=flip)
    plot_line(mod*y, error, color, label=label, ax=ax)

def prep_my_data(my_data, flip=False):
    smoothed_y = [np.mean(np.lib.stride_tricks.sliding_window_view(d, 10), axis=1) for d in my_data]
    # print(len(my_data))
    y, error = np.mean(smoothed_y, axis=0), (np.sqrt(np.var(smoothed_y, axis=0)/len(my_data)))
    return y, error

def plot_baseline_data(alg, envname, dataset, color, label, ax=plt):
    baseline_data = da.get_baseline_data(alg, envname, dataset)
    y, error = prep_my_data(baseline_data)
    plot_line(y, error, color, label=get_baseline_label(label), ax=ax)


def plot_eval_env(base_tkl_dir, base_hans_dir, hans_param_setting, envname, dataset, tau=0.1, save=None):
    tkl_dirname = os.path.join(
        base_tkl_dir, "env_name-{}".format(envname), "dataset-{}".format(dataset))
    df_tkl = da.analyze_data(tkl_dirname)
    sdf_tkl = da.transform_best_over(df_tkl, "end_mean")
    print(sdf_tkl["tau"])
    
    plt_data = sdf_tkl[["arr"]].iloc[0]

    hans_dir = base_hans_dir.format(envname, dataset)
    plot_data(hans_dir, hans_param_setting, plt_data[0][0:5], save=save, title=envname + " " + dataset)

def plot_eval_everything(save_dir, legend):

    params = {'legend.fontsize': 18,
              'axes.labelsize': 'x-large',
              'axes.titlesize': 'x-large',
              'xtick.labelsize': 22,
              'ytick.labelsize': 22}
    plt.rcParams.update(params)

    dataset_name = "expert"
    for (envname, hps) in [("Hopper", 9), ("HalfCheetah", 4), ("Walker2d", 7), ("Ant", 4)]:
        plt.clf()
        plot_eval_env_multiple([("data/fixed_tsallis_inac/", "Tsallis", None),
                                ("data/tkl_policy/", "TKL", None)],
                               [("InAC", hps), ("td3_bc", None), ("iql_offline", None), ("awac_offline", None)],
                               envname, dataset_name, arr_name="arr",
                               save=os.path.join(save_dir, envname + "_" + dataset_name + ".pdf"),
                               legend=legend, ylims=(0.0, 1.4))

    dataset_name = "medexp"
    for (envname, hps) in [("Hopper", 9), ("HalfCheetah", 9), ("Walker2d", 2), ("Ant", 4)]:
        plt.clf()
        plot_eval_env_multiple([("data/fixed_tsallis_inac/", "Tsallis", None),
                                ("data/tkl_policy/", "TKL", None)],
                               [("InAC", hps), ("td3_bc", None), ("iql_offline", None), ("awac_offline", None)],
                               envname, dataset_name, arr_name="arr",
                               save=os.path.join(save_dir, envname + "_" + dataset_name + ".pdf"),
                               legend=legend, ylims=(0.0, 1.4))

    dataset_name = "medrep"
    for (envname, hps) in [("Ant", 1), ("HalfCheetah", 6), ("Hopper", 1), ("Walker2d", 1)]: #, ("Ant", 1)]:
        plt.clf()
        plot_eval_env_multiple([("data/fixed_tsallis_inac/", "Tsallis", None),
                                ("data/tkl_policy/", "TKL", None)],
                               [("InAC", hps), ("td3_bc", None), ("iql_offline", None), ("awac_offline", None)],
                               envname, dataset_name, arr_name="arr",
                               save=os.path.join(save_dir, envname + "_" + dataset_name + ".pdf"),
                               legend=legend, ylims=(0.0, 1.4))        

def plot_kldiv_everything(save_dir):
    
    dataset_name = "expert"
    for (envname, hps) in [("Hopper", 9), ("HalfCheetah", 4)]:
        plt.clf()
        plot_eval_env_multiple([("data/fixed_tsallis_inac/", "Tsallis", None),
                                ("data/tkl_policy/", "TKL", None),
                                ("data/insample/", "InAC", None)],
                               None,
                               envname, dataset_name, arr_name="kldiv",
                               save=os.path.join(save_dir, envname + "_" + dataset_name + "_kldiv.pdf"), flip=True)

    dataset_name = "medexp"
    for (envname, hps) in [("Hopper", 9), ("HalfCheetah", 9), ("Walker2d", 2), ("Ant", 4)]:
        plt.clf()
        plot_eval_env_multiple([("data/fixed_tsallis_inac/", "Tsallis", None),
                                ("data/tkl_policy/", "TKL", None)],
                               None,
                               envname, dataset_name, arr_name="kldiv",
                               save=os.path.join(save_dir, envname + "_" + dataset_name + "_kldiv.pdf"),
                               flip=True)


def plot_eval_env_multiple(base_tkl_dirs_names,
                           hans_alg_dir_setting,
                           envname,
                           dataset,
                           show=False,
                           save=None,
                           arr_name="arr",
                           ylims=None, xlims=None, legend=False,
                           title_extra="",
                           flip=False,
                           ax=plt):

    i = 0
    for (base_tkl_dir, name, subset_func) in base_tkl_dirs_names:
        plot_mydata_line(
            get_dirname(base_tkl_dir, envname, dataset),
            color=get_color(name), label=name, arr_name=arr_name,
            subset_func=subset_func, flip=flip, ax=ax
        )
        i += 1

    if arr_name == "arr" and hans_alg_dir_setting is not None:
        for (alg, setting) in hans_alg_dir_setting:
            if alg == "InAC":
                base_hans_dir = "data/lr1e-3/{}/in_sample_ac/data_{}/sweep/".format(
                    envname, dataset)
                plot_hans_line(base_hans_dir,
                               setting,
                               get_color(alg),
                               label="InAC",
                               ax=ax)
            else:
                plot_baseline_data(
                    alg, envname, dataset, get_color(alg), label=alg, ax=ax)

            i += 1

    if ylims is not None:
        ax.ylim(ylims)
    if xlims is not None:
        ax.xlim(xlims)
    if legend:
        ax.legend()
    # ax.title(envname + " " + dataset + " " + title_extra)

    if show:
        ax.show()

    if save is not None:
        ax.savefig(save)

def plot_sensitivity_multiple_tau(base_tkl_dirs_names,
                                  envname,
                                  dataset,
                                  data_keys=("end_mean", "end_stderr"),
                                  show=False,
                                  save=None, legend=True):
    colorset = tc.colorsets["bright"]
    i = 0
    for (base_tkl_dir, name, subset_func) in base_tkl_dirs_names:
        ddn = get_dirname(base_tkl_dir, envname, dataset)
        sdf = da.sensitivity_curve(ddn, "end_mean", ["tau"])
        tau = sdf["tau"].to_numpy()
        y = sdf[data_keys[0]].to_numpy()
        err = sdf[data_keys[1]].to_numpy()

        plot_sens_line(tau, y, err, get_color(name), label=name)
        i += 1

    if legend:
        plt.legend()
    plt.title(envname + " " + dataset)

    if show:
        plt.show()
        # plt.clf()

    if save is not None:
        plt.savefig(save)
        plt.clf()


def plot_all_sens():

    params = {'legend.fontsize': 'large',
              'axes.labelsize': 'x-large',
              'axes.titlesize': 'x-large',
              'xtick.labelsize': 18,
              'ytick.labelsize': 18}
    plt.rcParams.update(params)
    
    plt.clf()
    plot_sensitivity_multiple_tau(
        [("data/tkl_q3/", "q=3", None),
         ("data/tkl_policy/", "q=2", None),
         ("data/tkl_q5/", "q=5", None),
         ("data/tkl_q10/", "q=10", None),
         ("data/fixed_tsallis_inac/", "Tsallis", None)],
        "Hopper", "expert", save="plots/sens/tau_sens_hopper_expert.pdf")

    plt.clf()
    plot_sensitivity_multiple_tau(
        [("data/tkl_q3/", "q=3", None),
         ("data/tkl_policy/", "q=2", None),
         ("data/tkl_q5/", "q=5", None),
         ("data/tkl_q10/", "q=10", None),
         ("data/fixed_tsallis_inac/", "Tsallis", None)],

        "Walker2d", "expert", save="plots/sens/tau_sens_walker_expert.pdf")

    plt.clf()
    plot_eval_env_multiple(
        [("data/tkl_policy/", "q=2", None),
         ("data/tkl_q3/", "q=3", None),
         ("data/tkl_q5/", "q=5", None),
         ("data/tkl_q10/", "q=10", None),
         ("data/fixed_tsallis_inac/", "Tsallis", None)],
        None,
        "Hopper", "expert", arr_name="kldiv",
        save="plots/sens/qsweep_kldiv_hopper_expert.pdf")

    plt.clf()
    plot_sensitivity_multiple_tau(
        [("data/tkl_policy/", "TKL", None),
         ("data/fixed_tsallis_inac/", "Tsallis", None)],
        "Hopper", "medrep", save="plots/sens/tau_sens_hopper_medrep.pdf")
    plt.clf()
    plot_sensitivity_multiple_tau(
        [("data/tkl_policy/", "TKL", None),
         ("data/fixed_tsallis_inac/", "Tsallis", None)],
        "Walker2d", "medrep",
        save="plots/sens/tau_sens_walker_medrep.pdf")

    for tau in [0.01, 0.1, 0.33, 0.5, 1.0]:
        plt.clf()
        plot_eval_env_multiple(
            [("data/tkl_policy/", "q=2", lambda df: df[df["tau"] == tau]),
             ("data/tkl_q3/", "q=3", lambda df: df[df["tau"] == tau]),
             ("data/tkl_q5/", "q=5", lambda df: df[df["tau"] == tau]),
             ("data/tkl_q10/", "q=10", lambda df: df[df["tau"] == tau]),
             ("data/fixed_tsallis_inac/", "Tsallis", lambda df: df[df["tau"] == tau])],
            None,
            "Hopper", "expert", arr_name="arr",
            save="plots/sens/qsweep_eval_hopper_expert_tau_{}.pdf".format(tau),
            title_extra="tau={}".format(tau))

    for tau in [0.01, 0.1, 0.33, 0.5, 1.0]:
        plt.clf()
        plot_eval_env_multiple(
            [("data/tkl_policy/", "q=2", lambda df: df[df["tau"] == tau]),
             ("data/tkl_q3/", "q=3", lambda df: df[df["tau"] == tau]),
             ("data/tkl_q5/", "q=5", lambda df: df[df["tau"] == tau]),
             ("data/tkl_q10/", "q=10", lambda df: df[df["tau"] == tau]),
             ("data/fixed_tsallis_inac/", "Tsallis", lambda df: df[df["tau"] == tau])],
            None,
            "Walker2d", "expert", arr_name="arr",
            save="plots/sens/qsweep_eval_walker_expert_tau_{}.pdf".format(tau),
            title_extra="tau={}".format(tau))

    plt.clf()
    plot_eval_env_multiple(
        [("data/tkl_policy/", "q=2", None),
         ("data/tkl_q3/", "q=3", None),
         ("data/tkl_q5/", "q=5", None),
         ("data/tkl_q10/", "q=10", None),
         ("data/fixed_tsallis_inac/", "Tsallis", None)],
        None,
        "Hopper", "expert", arr_name="kldiv", ylims=(-0.25, 1.5), flip=True,
        save="plots/sens/SMALL_qsweep_kldiv_hopper_expert.pdf", legend=True)

    plt.clf()
    plot_eval_env_multiple(
        [("data/tkl_policy/", "q=2", None),
         ("data/tkl_q3/", "q=3", None),
         ("data/tkl_q5/", "q=5", None),
         ("data/tkl_q10/", "q=10", None),
         ("data/fixed_tsallis_inac/", "Tsallis", None)],
        None,
        "Hopper", "expert", arr_name="kldiv",
        xlims=(0, 10), flip=True,
        save="plots/sens/SMALLXLIM_qsweep_kldiv_hopper_expert.pdf")

    plt.clf()
    plot_eval_env_multiple(
        [("data/tkl_policy/", "q=2", None),
         ("data/tkl_q3/", "q=3", None),
         ("data/tkl_q5/", "q=5", None),
         ("data/tkl_q10/", "q=10", None),
         ("data/fixed_tsallis_inac/", "Tsallis", None)],
        None,
        "Walker2d", "expert", arr_name="kldiv", flip=True,
        save="plots/sens/qsweep_kldiv_walker_expert.pdf")

    plt.clf()
    plot_eval_env_multiple(
        [("data/tkl_policy/", "q=2", None),
         ("data/tkl_q3/", "q=3", None),
         ("data/tkl_q5/", "q=5", None),
         ("data/tkl_q10/", "q=10", None),
         ("data/fixed_tsallis_inac/", "Tsallis", None)],
        None,
        "Walker2d", "expert", flip=True, arr_name="kldiv", ylims=(0.0, 1.5),
        save="plots/sens/SMALL_qsweep_kldiv_walker_expert.pdf", legend=True)


def plot_qsweep_grid(save):

    envname = "Walker2d"
    dataset = "expert"
    
    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(5, 4)

    j = 0
    for tau in [0.01, 0.1, 0.33, 0.5, 1.0]:
        ax = fig.add_subplot(gs[j, 0])
        plot_eval_env_multiple(
            [("data/tkl_policy", "q=2", lambda df: df[df["tau"] == tau])],
            None,
            envname, dataset, arr_name="arr", ax=ax)

        i = 1
        for q in [3, 5, 10]:
            ax = fig.add_subplot(gs[j, i])
            plot_eval_env_multiple(
                [("data/tkl_q{}".format(q),
                  "q={}".format(q),
                  lambda df: df[df["tau"] == 0.01])],
                None,
                envname, dataset, ax=ax, arr_name="arr")
            i += 1
        j += 1
    return fig

def plot_qsweep_heatmap(envname, dataset, key="end_mean"):

    params = {'legend.fontsize': 'large',
              'axes.labelsize': 20,
              'axes.titlesize': 'x-large',
              'xtick.labelsize': 18,
              'ytick.labelsize': 18}
    plt.rcParams.update(params)
    # envname = "Walker2d"
    # dataset = "expert"
    mat = np.zeros((4, 5))

    ddn = get_dirname("data/tkl_policy", envname, dataset)
    sdf = da.sensitivity_curve(ddn, "end_mean", ["tau"])
    tau = sdf["tau"].to_numpy()
    y = sdf[key].to_numpy()
    err = sdf["end_stderr"].to_numpy()
    mat[0, :] = y

    i = 1
    for q in [3, 5, 10]:
        ddn = get_dirname("data/tkl_q{}".format(q), envname, dataset)
        sdf = da.sensitivity_curve(ddn, "end_mean", ["tau"])
        y = sdf[key].to_numpy()
        mat[i, :] = y
        i += 1

    plt.clf()
    
    heatmap(mat, [2, 3, 5, 10], tau, xlabel=r"$\tau$", ylabel="q")


def plot_kldiv_env(base_tkl_dir, envname, dataset, tau=0.1, save=None):
    tkl_dirname = os.path.join(
        base_tkl_dir,
        "env_name-{}".format(envname),
        "dataset-{}".format(dataset))
    df_tkl = da.analyze_data(tkl_dirname)
    sdf_tkl = da.transform_best_over(df_tkl, "end_mean")
    print(sdf_tkl["tau"])

    plt_data = sdf_tkl[["kldiv"]].iloc[0]
    plot_kldiv(plt_data[0][0:5], save=save, title=envname + " " + dataset)



def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", xlabel="", ylabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, cmap="magma", **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar    
