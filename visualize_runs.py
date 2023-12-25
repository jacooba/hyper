'''
A script for plotting tensorboard data with multiple runs of a model. Assumes that the dirs
written out for each run were formatted by maze_runner.py. Originally from AMRL paper
Github: https://github.com/jacooba/AMRL-ICLR2020/
Paper: https://iclr.cc/virtual_2020/poster_Bkl7bREtDr.html
'''

from tensorboard.backend.event_processing import event_accumulator
from collections import defaultdict, OrderedDict
from argparse import ArgumentParser
from multiprocessing import Pool
from glob import glob

import os
import json
import zlib
import random

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams.update({'font.weight': 'bold'})
import seaborn as sns
sns.set()

# Define Model Order of Legend
MODEL_ORDER = [
        "VI",
        "VI+HN",
        "TI_Naive",
        "TI",
        "TI++",
        "TI+HN",
        "TI++HN",
        "BI++HN",
        "BIL++HN",
        "RNN",
        "RNN+S",
        "RNN+HN",
        "Multi-Task",
        "RNN+HN_Kaiming",
        "TI++HN_0updates",
        "TI++HN_100updates",
        "TI++HN_300updates",
        "TI++HN_500updates",
        "TI++HN_1000updates",
        "Multi+HN",
        "RNN+HN+BI++",
        "RNN+HN+BI10p++",
        "RNN+HN+BI50p++",
        "RNN+HN+TI_Naive",
        "RNN+HN+TI10p_Naive",
    ]
# Define colors for plotting
blue, orange, green, red, purple, brown, pink, grey, yellow = sns.color_palette(n_colors=9)
black = (0,0,0)
COLOR_PALETTE = defaultdict(lambda: black, {
    "RNN": grey,
    "RNN+S": red,
    "RNN+HN": blue,
    "Multi-Task": purple,
    "VI+HN":orange,
    "TI++HN":green,
    "VI": black,
    "BI++HN": grey,
    "BIL++HN": red,
    "TI+HN": pink,
    "TI": brown,
    "TI++": (blue[0]*1.5, blue[1]*1.5, blue[2]*1.1),
    "TI_Naive": yellow,
    "RNN+HN_Kaiming": (blue[0]*1.5, blue[1]*1.5, blue[2]*1.1),
    "RNN+HN+BI++": black,
    "RNN+HN+TI_Naive": (blue[0]*1.7, blue[1]*1.6, blue[2]*1.6),
    "RNN+S+TI": (green[0]*1.6, green[1]*1.6, green[2]*1.3),
    "RNN+HN+BI50p++": yellow,
})
for key, value in list(COLOR_PALETTE.items()):
    COLOR_PALETTE[key] = tuple(np.clip(np.array(value), 0, 1))


# Define some sets of experiments for easy plotting
HYPER_LIST = ["RL2", "Varibad", "MultiActivation", "MultiLargeEmbed", "MultiNet", "MultiHead", 
              "HyperNet_no_init", "HyperNet_init_b", "RL2_HyperNet_init_b_avg", "FiLM"] # "HyperNet_init_wb", "HyperNet_init_w",  "HyperHead", "HyperNet"
HYPER_SET = {h for h in HYPER_LIST}
MODEL_ORDER += HYPER_LIST + ["NO_TASK"]
SPECIAL_SETS = {"HYPER_SET":HYPER_SET,}


# REPO_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

def simple_name(model_name):
    # optionally simplify the long names of models for plotting:

    model_name = model_name.replace("smallPol_smallEm", "small").replace("nonVar", "N").replace("_FullTaskChance0", "").replace("_TaskChance0", "").replace("_pol_XL", "")

    return model_name

def get_name_events_info(args, directory):
    # e.g. logs/Avg_ST_20__03:12_22:56:12
    dirname = os.path.basename(os.path.normpath(directory)) # => Avg_ST_20__03:12_22:56:12
    model_name_and_seed, date = dirname.split("__")    # => Avg_ST_20     and   03:12_22:56:12
    model_name = "_".join(model_name_and_seed.split("_")[:-1]) # Avg_ST
    if not args.full_names:
        model_name = simple_name(model_name)
    run_num = model_name_and_seed.split("_")[-1] # 20
    run_info = {"run": run_num}
    # get lr from config
    config_path = os.path.join(directory, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as json_file:
            conf = json.load(json_file)
            lr_info = tuple([conf[k] for k in args.lr_keys]) # tuple of all lrs
            run_info["lr"] = lr_info
    else:
        print("Warning: no config found for", directory)
    #
    # Note: "Optimizer" key also supported in to_return later
    events = event_accumulator.EventAccumulator(directory)
    events.Reload()
    # Deal with getting scalars given tag
    scalars = get_scalars(args.tag, events)
    if scalars is None:
        return (None, None, None)
    if args.metric_tag is None:
        run_info["metric_scalars"] = scalars
    else:
        metric_scalars = get_scalars(args.metric_tag, events)
        run_info["metric_scalars"] = metric_scalars
        if metric_scalars is None:
            return (None, None, None)
    return (model_name, scalars, run_info)

def get_scalars(t, events):
    tags = events.Tags()
    if t not in tags["scalars"]:
        replaced_spaces = t.replace(" ", "_") # fix needed for walker / older mujoco
        replaced_underscore = t.replace("_", " ") # fix needed to parse list of tags
        if replaced_spaces in tags["scalars"]:
            t = replaced_spaces
        elif replaced_underscore in tags["scalars"]:
            t = replaced_underscore
        else:
            print("Warning: run without given tag ({}). Available tags:".format(t) + str(tags["scalars"]))
            return None
    return events.Scalars(t)


def get_model_name_2_runs(args):
    # Make dict of name to [(run_events, run_info), ...] (multiple runs). Note multiple runs share the same name, given multiple seeds.
    dirs = glob(os.path.normpath(args.dir) + "/*/")
    dirs = [d for d in dirs if args.ignore not in d]           # exclude models we want to ignore
    print("Num dirs found:", len(dirs))
    print("e.g.", dirs[0], "and", dirs[-1])
    # load in all data for each dir using process pool
    with Pool(processes=len(dirs)) as p_pool:
        loaded_data = p_pool.starmap(get_name_events_info, [(args, d) for d in dirs])
    # add this data to model_name_2_runs
    model_name_2_runs = defaultdict(list)
    for model_name, scalars, run_info in loaded_data:
        if model_name is None: # happens if no scalars found
            continue
        model_name_2_runs[model_name].append((scalars, run_info))
    return model_name_2_runs

def num_warmups(full_name):
    # Get number of warmups (pre-compute), if relevant for this model
    if "warmups" in full_name.lower():
        str_after_warm = full_name.lower().split("warmups")[1]
        if str_after_warm[:4] == "full": # remove "full" string to get number after
            str_after_warm = str_after_warm[4:]
        num_warmups = int(str_after_warm.split("_")[0])
    else:
        return None
    return num_warmups

def get_clean_data(model_name_2_runs, args):
    global MODEL_ORDER
    global COLOR_PALETTE
    model_to_fullname = {} # fullnames will be made that include some run info
    data = []
    model_2_best_performances = {} # Get metric for best performing learning rate (LR)
    model_2_best_LR = {} # Get best lrs
    #
    # set up model exclusions
    exclude_mode = None
    if "," in args.exclude:
        exclude_set = set([s.lower() for s in args.exclude.split(",") if s != ""])
        exclude_mode = "set"
    elif args.exclude != "":
        exclude_mode = "substr"
    # set up model inclusions
    include_mode = None # the mode for args.only 
    if args.only in SPECIAL_SETS: # args.only specifies a pre-defined set
        include_set = SPECIAL_SETS[args.only]
        include_mode = "set"
    elif "," in args.only: # args.only is a list
        include_set = set([s.lower() for s in args.only.split(",") if s != ""])
        include_mode = "set"
    elif args.only != "": # args.only is a substring of model names 
        include_mode = "substr"
    #
    for model_name, runs in model_name_2_runs.items():
        fixed_version_exists = (model_name+"_Fixed") in model_name_2_runs
        model_name = model_name.replace("_Fixed", "") # If this model fixes another
        legend_name = "Model"
        # skips for old version
        if fixed_version_exists:
            continue
        # skips for excludes
        if exclude_mode is not None:
            if exclude_mode == "set" and model_name.lower() in exclude_set:
                continue
            if exclude_mode == "substr" and args.exclude.lower() in model_name.lower():
                continue
        # skips for includes
        if include_mode is not None:
            if include_mode == "set" and model_name.lower() not in include_set:
                continue
            if include_mode == "substr" and args.only.lower() not in model_name.lower():
                continue
        if model_name not in MODEL_ORDER:
            print("Warning: model name,", model_name, "not in MODEL_ORDER list. It will be added at the end.")
            MODEL_ORDER.append(model_name)
        if model_name not in COLOR_PALETTE:
            print("Warning: model name,", model_name, "not in COLOR_PALETTE. It will be added with a random color.")
            h = zlib.adler32(model_name.encode()) # cannot use python hash(), since it is stochastic
            r = np.random.default_rng(seed=abs(h)) # use hash of model name as seed so colors are reproducible between runs
            COLOR_PALETTE[model_name] = tuple(.75 * r.uniform(size=(3,)))
        num_runs = str(len(runs))
        full_name = (model_name + "_" + str(num_runs) + "runs") if args.nr else model_name # Add number of runs to name in legend
        model_to_fullname[model_name] = full_name
        # Note: if you use a tag that is not the episode length or reward, you should be able to do:
            # y_label = args.tag.split("/")[-1]
        # but it will not be capitalized properly
        y_label = "Return"
        x_label = "Frames (k)"
        if "episode_len" in args.tag:
                y_label = "Episode Length"
        if "success" in args.tag.lower() and "test" in args.tag.lower():
                y_label = "Success Rate (test)"
        elif "norm" in args.tag:
                y_label = "Norm"
                x_label = "Updates"
        # Find best lr if lr search and calculate run lengths to make sure no events are missing
        max_run_len = 0
        prev_run_info = None
        lr_2_metrics = defaultdict(list) # If multiple lr runs, get final rewards for each run, to find best
        for scalars, run_info in runs: # runs contains repreats as well as different lr specified in run_info
            m_scalars = run_info["metric_scalars"]
            if args.step is not None:
                m_scalars =[s for s in m_scalars if s.step < args.step*1000]
            max_run_len = max(max_run_len, len(m_scalars))
            lr_or_None = run_info["lr"] if "lr" in run_info else None
            if args.shift_warmups:
                num_frames_warm = None if num_warmups(full_name) is None else num_warmups(full_name) * args.frames_per_update
                m_scalars = m_scalars if num_warmups(full_name) is None else [s for s in m_scalars if s.step >= num_frames_warm]
            if args.final_r:
                # metric = scalars[-1].value # (to not smooth first)
                metric = smooth_ys(m_scalars, args)[-1] # smooth first. # Faster than below and works
                # Note, the following is slower but safer:
                    # events = scalars
                    # sorted_events = sorted(events, key=lambda event: event.step)
                    # final_r = sorted_events[-1].value
                    # assert metric == final_r, (metric, final_r)
            elif args.AUC: # Area under the curve
                rs = [event.value for event in m_scalars]
                steps =[event.step for event in m_scalars]
                metric = np.trapz(y=rs,x=steps)
            else:
                metric = np.mean([event.value for event in m_scalars])
            lr_2_metrics[lr_or_None].append(metric)
        best_lr, best_metric = None, None
        for lr, metrics in lr_2_metrics.items():
            avg_metric = np.mean(metrics)
            if best_metric == None or avg_metric > best_metric:
                best_lr = lr
                best_metric = avg_metric
                model_2_best_performances[full_name] = metrics
        print("\n(Best over lr) Metric for {} is: {}".format(full_name, best_metric))
        print("  with lr =", str(tuple(args.lr_keys)),"=", best_lr,"\n")
        print("  and individual runs:", model_2_best_performances[full_name])
        model_2_best_LR[full_name] = best_lr
        # Save data for plotting
        for scalars, run_info in runs:
            if args.step is not None:
                scalars =[s for s in scalars if s.step < args.step*1000]
            run_len = len(scalars)
            if run_len < max_run_len:
                print("Warning, found events missing for a run with {}, dir {}".format(full_name, args.dir))
                print("  Last step: {}".format(scalars[-1].step))
            if run_len == 0:
                continue
            if args.best and "lr" in run_info and run_info["lr"] != best_lr:
                continue
            zipped_events = list(zip(scalars, smooth_ys(scalars, args)))
            if args.max_events is not None and len(zipped_events) > args.max_events:
                scale_down_factor = len(zipped_events)/args.max_events 
                zipped_events = [zipped_events[int(round(i*scale_down_factor))] for i in range(0, args.max_events)]
            for scalar_event, smoothed_y in zipped_events:
                if args.step is None or (scalar_event.step <= (args.step*1000)):
                    x_step = scalar_event.step/1000
                    if args.shift_warmups and ("warmups" in full_name.lower()):
                        num_frames_warm = num_warmups(full_name) * args.frames_per_update
                        x_step -= num_frames_warm/1000
                    event_dict = {legend_name: full_name, 
                                  x_label: x_step, 
                                  y_label: smoothed_y, 
                                 }
                    event_dict.update(run_info)
                    if "lr" in run_info: # for now, since numberical lr is not displayed correctly
                        clean_lrs = [format(k, ".0e") for k in run_info["lr"]]
                        event_dict["lr"] = ",".join(clean_lrs)
                    data.append(event_dict)
    if not data:
        print("No data found")
        exit()

    return data, model_2_best_performances, model_2_best_LR, model_to_fullname, x_label, y_label, legend_name

def smooth_ys(scalars, args):
    y_vals = [se.value for se in scalars]
    if args.smoothing_sz is None:
        smoothed_y_vals = y_vals
    else:
        smooth_window = np.ones(max(2,int(len(y_vals)*args.smoothing_sz)))
        smoothed_y_vals = np.convolve(np.pad(y_vals,(len(smooth_window)-1,0),mode="edge"), smooth_window/len(smooth_window), mode='valid')
    assert len(smoothed_y_vals)==len(y_vals), (len(smoothed_y_vals),len(y_vals))
    return smoothed_y_vals

def plot_runs(data, model_2_best_performances, model_2_best_LR, model_to_fullname, x_label, y_label, legend_name, args):
    fig = plt.figure()
    fullname_model_order = [model_to_fullname[m] for m in MODEL_ORDER if m in model_to_fullname]
    estimator = None if args.indiv else "mean"
    ci = None if args.indiv else args.ci
    units = "run" if args.indiv else None
    # Note: To visualize the opimizer used, you may be able to do the following:
        # style = "Optimizer" if "Optimizer" in data[0] else None
        # size = "lr" if "lr" in data[0] else None
    # However, This created issues displaying the lr and AMRL only used Adam optimizer, so we do this instead:
    style = "lr" if ("lr" in data[0]) and (args.lr or not args.best) else None
    size = None
    # dataf[y_label] = dataf[y_label].rolling(max(1,int(len(data)*smoothing_sz))).mean()
    plot = sns.lineplot(x=x_label, y=y_label, estimator=estimator, units=units, markers=True, 
        style=style, size=size, hue=legend_name, hue_order=fullname_model_order,
        data=pd.DataFrame(data), ci=ci, palette=COLOR_PALETTE)

    if args.log_scale:
        plot.set_yscale('log')

    if args.no_legend:
        plot.get_legend().remove()

    if args.no_ylabel:
        plt.ylabel("")

    if args.lower_ylim is not None:
        plot.set_ylim(bottom=args.lower_ylim)
    if args.upper_ylim is not None:
        plot.set_ylim(top=args.upper_ylim)
    
    # save and show plot
    # uncomment for title
    # if args.title: 
    #     fig.suptitle(args.tag, fontsize=16) # Or set directory name as title if not tag
    fig.set_size_inches(7, 4.5)
    if args.out is None:
        OUT_PATH_PNG = '/mnt/jabeckstorage/ray_results/current_plot.png'
        OUT_PATH_CSV = '/mnt/jabeckstorage/ray_results/current_plot.csv'
    else:
        OUT_PATH_PNG = os.path.normpath(args.out)+".png"
        OUT_PATH_CSV = os.path.normpath(args.out)+".csv"
    plt.savefig(OUT_PATH_PNG, dpi=500, bbox_inches="tight")

    # write out csv of overall metrics
    if args.final_r:
        metric_str = "Final Return"
    elif args.AUC:
        metric_str = "AUC"
    else:
        metric_str = "AVG_Return"
    with open(OUT_PATH_CSV, "w+") as file:
        file.write("Model;"+ "LR="+str(tuple(args.lr_keys))+";" +metric_str+"\n")
        for k, v in model_2_best_performances.items():
            performance_str = ",".join([str(metric) for metric in v])
            file.write(str(k)+";"+str(model_2_best_LR[k])+";"+performance_str+"\n")
    if not args.no_show:
        plt.show()

    # make a bar plot of the overall metrics
    data = [] 
    y_label = metric_str
    use_AUC = False
    if metric_str == "AUC":
        use_AUC = True
        y_label = y_label + " (M)"
    min_val = np.inf
    num_runs = max([len(l) for l in list(model_2_best_performances.values())])
    for model in model_2_best_performances.keys():
        for run_num in range(num_runs):
            if len(model_2_best_performances[model]) <= run_num:
                print("Warning: missing runs!")
                continue
            value = model_2_best_performances[model][run_num]
            if use_AUC:
                value /= (10**6)
            event_dict = {"Model": model, 
                          y_label: value, 
                          "run_num": run_num,
                         }
            min_val = min(min_val, value)
            data.append(event_dict)
    fig = plt.figure()
    plt.xticks(rotation=90)
    if args.title:
        fig.suptitle("Overall Performance ({})".format(args.tag), fontsize=16)
    # Sort model order according to performance
    sorted_model_order = list(sorted(list(model_2_best_performances.keys()), key = lambda model: np.mean(model_2_best_performances[model])))
    plot = sns.barplot(x="Model", y=y_label, data=pd.DataFrame(data), 
        ci=ci, order=sorted_model_order, palette=COLOR_PALETTE) # 68% confidence interval
    fig.set_size_inches(13, 7)
    if args.log_scale:
        plot.set_yscale('log')
    if args.lower_ylim is not None:
        plot.set_ylim(bottom=args.lower_ylim)
    else:
        plot.set_ylim(bottom=0.9*min_val) # Clip bottom of plot
    if args.upper_ylim is not None:
        plot.set_ylim(top=args.upper_ylim) 
    out_path = os.path.normpath(args.out)+"_"+metric_str+"_Overall_Performance.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")


def main():
    parser = ArgumentParser()
    parser.add_argument("dir", type=str, help="the result dir to plot")
    parser.add_argument("--tag", type=str, help="the tensorflow tag to plot", default="Meta-Episode Return")
    parser.add_argument("--ignore", type=str, help="ignore directories containing this string", default="state_and_conf")
    parser.add_argument("--exclude", type=str, help="model names to exclude separated by commas with no spaces", default="")
    parser.add_argument("--only", type=str, help="only allow these model names. or allowed preset: HYPER_SET", default="")
    parser.add_argument("--ci", type=int, help="confidence interval for plot", default=68)
    parser.add_argument("--max_events", type=int, help="discard events so we plot maximum of this number of events", default=200)
    parser.add_argument("--step", type=int, help="max number of steps for plot in thousands (k), or None", default=None)
    parser.add_argument("--indiv", action='store_true', help="Whether to plot individual runs instead of confidence intervals", default=False)
    parser.add_argument("--best", action='store_true', help="Whether to display best over lr search (if multiple lr)", default=False)
    parser.add_argument("--lr_keys", type=str, nargs='+', default=["lr_vae"], help="The key for the tuned learning rate.")
    parser.add_argument("--nr", action='store_true', help="Include num runs in model legend", default=False)
    parser.add_argument("--lr", action='store_true', help="Show lr in legend if grid search done", default=False)
    parser.add_argument("--final_r", action='store_true', help="Use final reward instead of Average Return as metric for best lr", default=False)
    parser.add_argument("--AUC", action='store_true', help="Use AUC instead of Average Return as metric for best lr", default=False)
    parser.add_argument("--no_show", action='store_true', help="Do not show plot on webserver, just save", default=False)
    parser.add_argument("--no_legend", action='store_true', help="Do not show legend.", default=False)
    parser.add_argument("--no_ylabel", action='store_true', help="Do not show ylabel.", default=False)
    parser.add_argument("--log_scale", action='store_true', help="Plot in log scale", default=False)
    parser.add_argument("--full_names", action='store_true', help="Do not use simple_name() function", default=False)
    parser.add_argument("--out", type=str, help="file path for output. Default only works on my old VM.", default=None)
    parser.add_argument("--title", action='store_true', help="Show title", default=False)
    parser.add_argument("--lower_ylim", type=float, help="Lower ylim for plot. E.g. 10e-30.", default=None)
    parser.add_argument("--upper_ylim", type=float, help="Upper ylim for plot. E.g. 10e-30.", default=None)
    parser.add_argument("--smoothing_sz", type=float, help="Size of smoothing window as fraction of data. 1 is all data.", default=.1)
    parser.add_argument("--frames_per_update", type=int, help="number of frames in one update. Used for shifting plots with pre-compute", default=960)
    parser.add_argument("--shift_warmups", action='store_true', help="Shift plots with pre-compute specified in name of model.", default=False)
    parser.add_argument("--metric_tag", type=str, help="Pick best plot by this metric instead of --tag.", default=None)
    args = parser.parse_args()

    global SPECIAL_SETS

    if not args.full_names:
        global MODEL_ORDER
        global COLOR_PALETTE
        for i in range(len(MODEL_ORDER)):
            MODEL_ORDER[i] = simple_name(MODEL_ORDER[i])
        for k,v in list(COLOR_PALETTE.items()):
            del COLOR_PALETTE[k]
            COLOR_PALETTE[simple_name(k)] = v
        for k,v in SPECIAL_SETS.items():
            SPECIAL_SETS[k] = {simple_name(set_item) for set_item in SPECIAL_SETS[k]}
   
    # after processing, make special sets lower case
    for k,v in SPECIAL_SETS.items():
        SPECIAL_SETS[k] = {set_item.lower() for set_item in SPECIAL_SETS[k]}

    assert not (args.AUC and args.final_r)

    # Make dict of name to [(run_events, run_info), ...] (multiple runs). Note multiple runs share the same name, given multiple seeds.
    model_name_2_runs = get_model_name_2_runs(args)

    # Clean up data for plotting
    data, model_2_best_performances, model_2_best_LR, model_to_fullname, x_label, y_label, legend_name = get_clean_data(model_name_2_runs, args)

    # Plot
    plot_runs(data, model_2_best_performances, model_2_best_LR, model_to_fullname, x_label, y_label, legend_name, args)



if __name__ == "__main__":
    main()