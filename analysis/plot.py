from analysis.log2data import extract_from_single_run
import matplotlib.pyplot as plt
import numpy as np
import tol_colors as tc


def plot_data(hans_dir, hans_param_setting, my_data):
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
    plt.show()


def prep_hans_data(dir_name, param_setting):
    hans_hc = [extract_from_single_run(
        dir_name + "/" + "{}_run/{}_param_setting/log".format(r, param_setting),
        "normalized_return") for r in range(4)]
    hans_smoothed_hc_y = [np.mean(
        np.lib.stride_tricks.sliding_window_view(d, 10),
        axis=1) for d in hans_hc]
    y = np.mean(hans_smoothed_hc_y, axis=0)
    error = 1.97*(np.sqrt(np.var(hans_smoothed_hc_y, axis=0))/4)
    return y, error

def prep_my_data(my_data):
    smoothed_y = [np.mean(np.lib.stride_tricks.sliding_window_view(d, 10), axis=1) for d in my_data]
    y, error = np.mean(smoothed_y, axis=0), 1.97*(np.sqrt(np.var(smoothed_y, axis=0))/4)
    return y, error
