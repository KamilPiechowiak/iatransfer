import functools
import operator

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from glob import glob
from scipy.signal import savgol_filter

try:
    # if you are hacker "dark_background"
    plt.style.use(["science", "ieee", "high-vis"])
except:
    print("try `pip install SciencePlots`")



def normalize_data(all_arr):
    loss_train_all = []
    loss_train_all2 = []
    for arr in all_arr:
        lx = len(arr)
        loss_train_all.append(arr)
        yhat = savgol_filter(arr, 2 + 2 + lx // 24 - ((lx // 24) % 2 == 0), 2)
        loss_train_all2.append(yhat)
    loss_train_all = np.array(loss_train_all)
    loss_train_all2 = np.array(loss_train_all2)
    lx = loss_train_all.shape[1]
    ylog = savgol_filter(
        np.log(loss_train_all.mean(axis=0)), lx // 2 - ((lx // 2) % 2 == 0), 2
    )
    yhat = savgol_filter(loss_train_all.mean(axis=0), lx // 2 - ((lx // 2) % 2 == 0), 2)
    return lx, loss_train_all, loss_train_all2, ylog, yhat


def plot_key(
    results,
    key,
    log=False,
    xlabel="x",
    ylabel="y",
    title=None,
    subtitle=None,
    prefix=None,
    ncol=3,
    ylim=None,
    legend="lower",
):
    print(f"[plot_loss] generation `{key}`")

    untex = lambda x: x.replace(" ", "\ ").replace("_", "\_")

    plt.rcParams.update({"font.size": 14})

    plt.clf()
    ax = plt.gca()
    ax.autoscale(tight=True)
    fig = plt.gcf()
    fig.set_size_inches(3.25, 5)  # (6, 2.5)
    ax.set_title(untex(title), loc="left")

    if log:
        norm = lambda x: np.log(x)
    else:
        norm = lambda x: x

    results_norm = []
    for _, result in results.groupby(results["ghash"]):
        lx, arr_all, arr_p50, arr_p90log, arr_p90 = normalize_data(result[key])
        results_norm.append(
            [result.iloc[0]["meta"], lx, arr_all, arr_p50, arr_p90log, arr_p90]
        )

    for _result in results_norm:
        meta, lx, arr_all, _, _, _ = _result

        shift = meta["shift"] if "shift" in meta else 0
        for i in range(arr_all.shape[0]):
            plt.plot(
                np.array(range(lx)) + shift,
                norm(arr_all[i]),
                color=meta["color"],
                alpha=0.05,
                linestyle="-",
            )

    for _result in results_norm:
        meta, lx, _, arr_p50, _, _ = _result

        shift = meta["shift"] if "shift" in meta else 0
        plt.fill_between(
            np.array(range(lx)) + shift,
            norm(arr_p50.mean(axis=0) - arr_p50.std(axis=0)),
            norm(arr_p50.mean(axis=0) + arr_p50.std(axis=0)),
            facecolor=meta["color"],
            alpha=0.4,
        )

    for _result in results_norm:
        meta, lx, _, _, arr_p90log, arr_p90 = _result

        label = untex(meta["name"])
        linestyle = meta["linestyle"] if "linestyle" in meta else "-"
        arr_fit = arr_p90log if log else arr_p90

        shift = meta["shift"] if "shift" in meta else 0
        plt.plot(
            np.array(range(lx)) + shift,
            arr_fit,
            color=meta["color"],
            alpha=1,
            linestyle=linestyle,
            label=label,
            linewidth=3,
        )

    if log:
        ylabel = f"log({ylabel})"

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    # labels, handles = zip(labels, handles)

    # FIXME: dodac opcje / lub naprawic finalnie
    # plt.legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=2)
    # ax.legend(handles, labels, loc="lower right",
    #          ncol=ncol)
    # ax.legend(handles, labels, loc="lower left",
    #           bbox_to_anchor=(0, -0.5), ncol=ncol)
    # ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), ncol=ncol)
    # ax.legend(handles, labels, loc="lower center", ncol=ncol)
    # ax.legend(handles, labels, ncol=ncol, loc='upper center',
    #           bbox_to_anchor=(0.5, 1.30), fancybox=True, shadow=True,
    #           prop={'size': 4})
    if legend == "lower":
        ax.legend(handles, labels, ncol=ncol, loc='lower right',
                    fancybox=True, shadow=False,
                    prop={'size': 7}, framealpha=1, frameon=True)
    elif legend == "upper":
        ax.legend(handles, labels, ncol=ncol, loc='upper right',
                    fancybox=True, shadow=False,
                    prop={'size': 7}, framealpha=1, frameon=True)

    if subtitle:
        ax.text(
            1,
            1.025,
            untex(subtitle),
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=ax.transAxes,
        )

    if ylim:
        ax.set_ylim(*ylim)

    prefix = f"{prefix}_" if prefix else ""
    plt.savefig(f"results/{prefix}_{key}.pdf")  # FIXME: add flag
    # FIXME: add pdf compression as flag to `run.py`

################################################################################

# automatic
def get_colormap(arr, name='gist_rainbow', x=0, y=1):
    from matplotlib.colors import to_hex
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(name)
    # if not for_edges:
    #     arr = g1.cluster_map.keys()
    # else:
    #     arr = range(len(remap.keys()))

    colors = cmap(np.linspace(x, y, len(arr)))
    colors_map = {}  # FIXME: sorted?
    for (i, color) in zip(arr, colors):
        colors_map[i] = to_hex(color)
    return colors_map

###################

def prepare_data(df_raw, meta):
    df_ext = []
    for index, row in df_raw.iterrows():
        name = row[0]
        values = row[1:].tolist()
        ghash = name # all seperate?
        _meta = {
            "name": name,
            "color": "gray"
        }
        print(name)
        if name in meta:
            _meta.update(meta[name])
            print("\tmeta >>", meta[name])
        df_blob = {
            "ghash": ghash,
            "name": name,
            "values": values,
            "meta": _meta
        }
        df_ext.append(df_blob)
    df_plot = pd.DataFrame(df_ext)
    return df_plot

from copy import copy
def plot_data(path, meta, plot_meta):
    print(f"\x1b[6;30;42m [[{path}]] \x1b[0m")

    _meta = copy(meta)
    if path in plot_meta and "meta" in plot_meta[path]:
        _meta.update(plot_meta[path]["meta"])

    df_raw = pd.read_csv(path)
    df_plot = prepare_data(df_raw, meta=_meta)

    _plot_meta = {
        "xlabel": "epochs",
        "ylabel": "test accuracy",
        "title": "XXX",
        "subtitle": "YYY",
        "prefix": "plot_none",
        "ncol": 2,
        "ylim": [0, 1],
        "legend": "lower"
    }

    # if df_plot.shape[0] == 4:
    #    _plot_meta["ncol"] = 2

    if path in plot_meta:
        _plot_meta.update(plot_meta[path])

    plot_key(
        df_plot,
        key="values",
        log=False,
        xlabel=_plot_meta['xlabel'],
        ylabel=_plot_meta['ylabel'],
        title=_plot_meta['title'],
        subtitle=_plot_meta['subtitle'],
        prefix=_plot_meta['prefix'],
        ncol=_plot_meta['ncol'],
        ylim=_plot_meta['ylim'],
        legend=_plot_meta['legend'],
    )

if __name__ == "__main__":
    print("[FIGURE]")

    CE = get_colormap(['10', '20', '30', '40', '50'],
                            name='Reds', x=0.25)

    CB = get_colormap(['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2'],
                            name='winter')

    CR = get_colormap(['resnet_14', 'resnet_20', 'resnet_26', 'resnet_32'],
            name='bwr') # cool, Greens x=0.5

    CA = get_colormap(['regnety_004', 'mixnet_s', 'tf_mobilenetv3_large_minimal_100',
     'semnasnet_100', 'mnasnet_100'],
                      name='winter')
                      #name='gist_ncar', x=0.25, y=0.9)
                      #name='Reds', x=0.25)
    print(CE)
    meta = {
        #'resnet_narrow_14': {'color': 'red'},
        #'resnet_narrow_20': {'color': 'darkred'},
        #'resnet_wide_14': {'color': 'orange'},
        #'resnet_wide_20': {'color': 'darkorange'},

        'mixnet_s': {'color': CA['mixnet_s']},
        'mnasnet_100': {'color': CA['mnasnet_100']},
        'regnety_004': {'color': CA['regnety_004']},
        'semnasnet_100': {'color': CA['semnasnet_100']},
        'tf_mobilenetv3_large_minimal_100': {'color':
                                             CA['tf_mobilenetv3_large_minimal_100']},

        'baseline': {'color': 'black', 'linestyle': '--'},
        'resnet_14': {'color':  CR['resnet_14']},
        'resnet_wide_14': {'color':  CR['resnet_14'], 'linestyle': 'dotted'},
        'resnet_narrow_14': {'color':  CR['resnet_14'], 'linestyle': 'dashdot'},
        'resnet_20': {'color':  CR['resnet_20']},
        'resnet_wide_20': {'color':  CR['resnet_20'], 'linestyle': 'dotted'},
        'resnet_narrow_20': {'color':  CR['resnet_20'], 'linestyle': 'dashdot'},
        'resnet_26': {'color': CR['resnet_26']},
        'resnet_32': {'color': CR['resnet_32']},

        '10': {'color': CE['10']},
        '20': {'color': CE['20']},
        '30': {'color': CE['30']},
        '40': {'color': CE['40']},
        '50': {'color': CE['50']},
        'efficientnet-b0': {'color': CB['efficientnet-b0']},
        'efficientnet-b1': {'color': CB['efficientnet-b1']},
        'efficientnet-b2': {'color': CB['efficientnet-b2']},
    }

    plot_meta = {
        'resnet_14_CIFAR10.csv': {"ylim": [0.4, 0.9],
            "meta": {
                "resnet_14": {"color": "gray"}
            }
        },
        'resnet_20_CIFAR10.csv': {"ylim": [0.4, 0.9],
            "meta": {
                "resnet_20": {"color": "gray"}
            }
        },
        'resnet_32_CIFAR10.csv': {"ylim": [0.4, 0.9],
            "meta": {
                "resnet_32": {"color": "gray"}
            }
        },

        'resnet_14_CIFAR10_width.csv': {"ylim": [0.4, 0.9]},
        'resnet_20_CIFAR10_width.csv': {"ylim": [0.4, 0.9]},

        'resnet_narrow_14_CIFAR10.csv': {"ylim": [0.3, 0.9]},
        'resnet_wide_14_CIFAR10.csv': {"ylim": [0.3, 0.9]},

        'efficientnet-b0_CIFAR100_epochs.csv':
            {"ylim": [0, 0.7], "legend": "lower", "meta": {
                "efficientnet-b0": {"color": "gray"}
            }},
        "resnet_20_CIFAR100_epochs.csv":
            {"ylim": [0, 0.7], "legend": "lower", "meta": {
                "resnet_20": {"color": "gray"}
            }},

        "efficientnet-b0_CIFAR10_similarity_transfer.csv":
            {"legend": "lower", "ncol":1, "meta": {
                "efficientnet-b0": {"color": "gray"}
            }},

        "efficientnet-b0_CIFAR100_similarity_transfer.csv":
            {"legend": "upper", "ncol":1, "meta": {
                "efficientnet-b0": {"color": "gray"}
            }},

        # 3x2
       "efficientnet-b0_CIFAR10.csv":
        {"ylim": [0.4, 1], "meta": {
                "efficientnet-b0": {"color": "gray"}
            }},
       "efficientnet-b1_CIFAR10.csv":
            {"ylim": [0.4, 1], "meta": {
                "efficientnet-b1": {"color": "gray"}
            }},
       "efficientnet-b2_CIFAR10.csv":
            {"ylim": [0.4, 1], "meta": {
                "efficientnet-b2": {"color": "gray"}
            }},

       "efficientnet-b0_CIFAR100.csv":
            {"ylim": [0, 0.7], "meta": {
                "efficientnet-b0": {"color": "gray"}
            }},
       "efficientnet-b1_CIFAR100.csv":
            {"ylim": [0, 0.7], "meta": {
                "efficientnet-b1": {"color": "gray"}
            }},
       "efficientnet-b2_CIFAR100.csv":
            {"ylim": [0, 0.7], "meta": {
                "efficientnet-b2": {"color": "gray"}
            }},

        # 'resnet_14_CIFAR10.csv': {"ncol": 2}
    }

    for path in glob("figures/csv/*.csv"):
        if path not in plot_meta:
            plot_meta[path] = {}
        plot_meta[path]['prefix'] = path.replace('.csv', '').replace('csv', 'plots')
        plot_data(path, meta, plot_meta)
