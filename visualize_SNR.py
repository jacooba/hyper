'''
A script used to plot the Signal to Noise Ratio (hueristic) over time for untrained models.
Originally from AMRL paper
Github: https://github.com/jacooba/AMRL-ICLR2020/
Paper: https://iclr.cc/virtual_2020/poster_Bkl7bREtDr.html
'''

from collections import defaultdict, OrderedDict
from argparse import ArgumentParser, Namespace
from gym.spaces import Discrete, Box
from pathlib import Path
from glob import glob

import os
import copy
import torch
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from visualize_runs import MODEL_ORDER, COLOR_PALETTE
from metalearner import MetaLearner

from config.gridworld import \
    args_grid_oracle, args_grid_belief_oracle, args_grid_rl2, args_grid_varibad
from config.mujoco import \
    args_cheetah_dir_oracle, args_cheetah_dir_rl2, args_cheetah_dir_varibad, \
    args_cheetah_vel_oracle, args_cheetah_vel_rl2, args_cheetah_vel_varibad, args_cheetah_vel_avg, \
    args_ant_dir_oracle, args_ant_dir_rl2, args_ant_dir_varibad, \
    args_ant_goal_oracle, args_ant_goal_rl2, args_ant_goal_varibad, \
    args_walker_oracle, args_walker_avg, args_walker_rl2, args_walker_varibad

DEFAULT_CONFIG = args_grid_varibad # choose from above
SHARED_ARGS = {"encoder_gru_hidden_size": 256,
            # can add layers before encoder to mix up signal since there is no obs encoder... but transition encoder hasn't yet implemented this perfectly
               "encoder_layers_before_gru": []} 
MODELS = [
    {"exp_label": "Varibad",},
    {"exp_label": "Avg", 
        "encoder_agg_type": "avg"},
    {"exp_label": "Avg_ST", 
        "encoder_agg_type": "avg",
        "encoder_st_estimator": True,},
    {"exp_label": "Max", 
        "encoder_agg_type": "max",},
    {"exp_label": "Max_ST", 
        "encoder_agg_type": "max",
        "encoder_st_estimator": True,},
    {"exp_label": "Gauss", 
        "encoder_agg_type": "gauss",},
    {"exp_label": "Gauss_ST", 
        "encoder_agg_type": "gauss",
        "encoder_st_estimator": True,},
    {"exp_label": "Gauss_Trans", 
        "encoder_enc_type": "transition",
        "encoder_agg_type": "gauss",},
    {"exp_label": "No_Mem_Test", 
        "encoder_enc_type": None,},
    {"exp_label": "Varibad_Trans",
        "encoder_enc_type": "transition",},
    {"exp_label": "Avg_Trans", 
        "encoder_enc_type": "transition",
        "encoder_agg_type": "avg",},
    {"exp_label": "Avg_Weighted", 
        "encoder_agg_type": "weighted_avg",},
]
# add shared args
for key, value in SHARED_ARGS.items():
    for m_dict in MODELS:
        m_dict[key] = value
# add parsed args to each model dict
for m_dict in MODELS:
    arg_strs = []
    m_dict["args"] = DEFAULT_CONFIG.get_args(arg_strs) # args for this model
    for key, value in m_dict.items():
        setattr(m_dict["args"], key, value)


def calculate_SNR(agent_dict, args, agent_num, num_agents):
    SNRs = []
    for i in range(args.num_agents):
        agent_dict["config"].seed = i
        print("\n\nSeed:", i)
        print("Evaluating sample of agent {} of {}, named: {} ...\n\n".format(agent_num, num_agents, agent_dict["model_name"]))
        metalearner = MetaLearner(agent_dict["config"], make_logger=False)
        vae_enc = metalearner.vae.encoder
        for name, param in vae_enc.agg.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0)
        metalearner.envs.close()
        SNRs.append(calculate_SNR_for_policy(vae_enc, args))
    return SNRs

def calculate_SNR_for_policy(vae_enc, args):
    ones_input = np.ones(vae_enc.agg.input_size) * args.scale
    signals, noises = [], []
    for _ in range(args.num_runs_per_agent):
        if not args.noise: # if not just noises, need signals
            signal_observations = np.zeros((args.length_of_decay, vae_enc.agg.input_size))
            if args.strong: # signal randomly and repeatedly
                for i in range(len(signal_observations)-int(args.strong_block*len(signal_observations))):
                    if np.random.rand() < args.strong_p:
                        signal_observations[i] = ones_input
            else:
                signal_observations[0] = ones_input # signal at beginning
            signals.append(get_states_from_obs(vae_enc, signal_observations, args.grad))
        if not args.signal: # if not just signals, need noises
            noise_observations = [np.random.choice([1, -1], p=[0.5, 0.5]) * ones_input for _ in range(args.length_of_decay)]
            noises.append(get_states_from_obs(vae_enc, noise_observations, args.grad))
    if args.norm:
        outputs = signals if signals else noises
        runs, steps = zip(*outputs)
        assert len(runs) == 1, len(runs) # Don't want to compute an average over runs
        run, steps = runs[0], steps[0]
        run_norms = np.sqrt(np.sum(run**2, axis=1))
        mean_norm = np.mean(run_norms)
        trajectory = (run_norms - mean_norm)/mean_norm
        to_return = (trajectory, steps)
    elif args.signal:
        to_return = expected_powers(signals, args)
    elif args.noise:
        to_return = expected_powers(noises, args)
    else:
        (s, s_steps), (n, n_steps) = expected_powers(signals, args), expected_powers(noises, args)
        to_return = (s/n, s_steps)
        assert s_steps == n_steps, (s_steps, n_steps)
    return to_return

def get_states_from_obs(vae_enc, observations, grad, grad_step=1): # If grad, return gradients instead
    observations = torch.tensor(observations, dtype=torch.float, requires_grad=True).reshape((len(observations), 1, len(observations[0])))
    state = vae_enc.agg.init_state(1)
    if grad:
        assert grad_step==1, "grad_step !=1 is broken. Will need to debug later."
        recurrent_out, state = vae_enc.agg(observations, state)
        recurrent_out_mean, recurrent_out_logvar = torch.chunk(recurrent_out, 2, dim=-1)
        grads = []
        steps_to_return = range(0, len(observations), grad_step)
        for i in steps_to_return:
            g = torch.autograd.grad(torch.sum(recurrent_out_mean[i,:,:]), observations, allow_unused=True, retain_graph=True)[0][0]
            grads.append(g)
        vals_to_return = torch.stack(grads, dim=0).squeeze().detach().numpy()
        steps_to_return = range(len(observations))
    else:
        recurrent_out, state = vae_enc.agg(observations, state)
        assert recurrent_out.shape[1] == 1, recurrent_out.shape
        assert recurrent_out.shape[0] == len(observations), recurrent_out.shape
        recurrent_out_mean, recurrent_out_logvar = torch.chunk(recurrent_out, 2, dim=-1)
        # to sample, do:
            # sample = vae_enc.reparameterise(recurrent_out_mean, recurrent_out_logvar)
        vals_to_return = recurrent_out_mean.squeeze().detach().numpy()
        steps_to_return = range(len(observations))
    vals_to_return = vals_to_return.astype(np.float64) # Need the precision soon
    return vals_to_return, tuple(steps_to_return)

def expected_powers(runs, args):
    runs, steps = zip(*runs)
    for s in steps:
        assert s == steps[0], (s, steps[0]) ## All steps need to be same for calc
    length_of_runs = len(steps[0])
    runs = np.array(runs)
    assert len(runs.shape) == 3, runs.shape
    assert runs.shape[0] == args.num_runs_per_agent, (runs.shape[0], length_of_runs)
    assert runs.shape[1] == length_of_runs, (runs.shape[1], length_of_runs)
    power = np.sum(runs*runs, axis=-1)/(runs.shape[2])
    assert len(power.shape) == 2, power.shape
    assert power.shape[0] == args.num_runs_per_agent, (power.shape[0], args.num_runs_per_agent)
    assert power.shape[1] == length_of_runs, (power.shape[1], length_of_runs)
    expected_power = np.mean(power, axis=0)
    assert expected_power.shape == (length_of_runs,), (expected_power.shape, (length_of_runs,))
    return expected_power, steps[0]

def get_all_results(args):
    # Make list of {"model_name": str, "config": args_namespace, "snr": [snr1, snr2, ...], "steps":[1,2,3...]}, for each yaml
    results_dicts = []
    exclude_set = set([s for s in args.exclude.split(",") if s != ""])
    models = [m_dict for m_dict in MODELS if m_dict["exp_label"] not in exclude_set]
    for agent_num, m_dict in enumerate(models):
        # Parse Args 
        config = m_dict["args"]
        # Define model name
        model_name = m_dict["exp_label"]
        if (args.skip_ST and not args.grad) and ("ST" in model_name):
            continue # ex Avg_ST model has same SNR as AVG
        # Make dict for model
        agent_dict = {"model_name": model_name, "config": config, AGENT_KEY: []}
        # Calculate SNR
        if args.dummy:
            steps = [1,2]
            if args.signal:
                agent_dict[AGENT_KEY] = [([1.0,1.1], steps), ([1.1,1.2], steps)]
            elif args.noise:
                agent_dict[AGENT_KEY] = [([0.9,1.0], steps), ([1.1,1.2], steps)]
            else:
                agent_dict[AGENT_KEY] = [([0.8,1.0], steps), ([1.8,1.8], steps)]
        elif args.combo:
            grads = calculate_SNR(agent_dict, args, agent_num=agent_num+1, num_agents=len(models))
            snr_args = Namespace(**vars(args))
            snr_args.grad = False
            snr_args.signal = False
            snr_args.noise = False
            snrs = calculate_SNR(agent_dict, snr_args, agent_num=agent_num+1, num_agents=len(models))
            for grad_run, snr_run in zip(grads, snrs):
                grad_vals, grad_steps = grad_run
                snr_vals, snr_steps = snr_run
                assert snr_steps == tuple(range(args.length_of_decay)), (snr_steps, tuple(range(args.length_of_decay)))
                chosen_snr_vals = np.array([snr_vals[i] for i in grad_steps])
                combo_vals = chosen_snr_vals*grad_vals
                agent_dict[AGENT_KEY].append((combo_vals, grad_steps))
        else:
            agent_dict[AGENT_KEY] = calculate_SNR(agent_dict, args, agent_num=agent_num+1, num_agents=len(models))
        if args.final:
            agent_dict[AGENT_KEY] = [([vals[-1]], [steps[-1]]) for (vals, steps) in agent_dict[AGENT_KEY]]
            if AGENT_KEY == "snr":
                print("final SNRs for agent {}:".format(model_name), agent_dict["snr"], "\n")
        results_dicts.append(agent_dict)
    return results_dicts


def make_plot(results_dicts, args):
    global MODEL_ORDER
    global COLOR_PALETTE
    fig = plt.figure()
    data = []
    models_to_plot = set()
    # create y_label
    if args.combo:
        y_label = "grad*snr"
    elif args.norm:
        y_label = "(Norm - Mean)/Mean"
    elif args.grad:
        y_label = "grad"
    elif args.signal:
        y_label = "Signal Power"
    elif args.noise:
        y_label = "Noise Power"
    else:
        y_label = "snr"
    # get data in right format
    model_to_final_perfs = defaultdict(list) if args.final else None
    for agent_dict in results_dicts:
        if agent_dict["model_name"] not in MODEL_ORDER:
            print("Warning, model, {}, not in model order: {}".format(agent_dict["model_name"], MODEL_ORDER))
            MODEL_ORDER.append(agent_dict["model_name"])
            COLOR_PALETTE[agent_dict["model_name"]] = tuple(np.random.uniform(size=(3,))/2)
        models_to_plot.add(agent_dict["model_name"]) 
        for run_num, run in enumerate(agent_dict[AGENT_KEY]):
            v_s_pairs = list(zip(*run))
            if args.final:
                assert len(v_s_pairs) == 1, len(v_s_pairs)
                model_to_final_perfs[agent_dict["model_name"]].append(v_s_pairs[0][0])
            for value, step in v_s_pairs:
                event_dict = {"Model": agent_dict["model_name"], 
                              "Step": step,
                              y_label: value, 
                              "run_num": run_num,
                             }
                data.append(event_dict)
    # create plot
    if args.final:
        plt.xticks(rotation=90)
    # define model order for this plot based on args
    if args.final or args.sort:
        model_order = sorted(model_to_final_perfs.keys(), key=lambda m: np.mean(model_to_final_perfs[m]))
    else:
        model_order = [m for m in MODEL_ORDER if m in models_to_plot]
    model_order = list(OrderedDict.fromkeys(model_order)) # Make sure unique
    # Set and Sum have very similar SNRs, within the margin of error. To make the order standard, you can do:
        # if "SET" in model_order and "SUM" in model_order: # rearrange to put set after sum
        #     new_model_order = []
        #     for m in model_order:
        #         if m == "SET":
        #             continue
        #         new_model_order.append(m)
        #         if m == "SUM":
        #             new_model_order.append("SET")
        #     model_order = new_model_order
    palette = dict(COLOR_PALETTE) # Make a palette with random color for unknown models
    for m in models_to_plot:
        if m not in palette:
            MODEL_ORDER.append(m)
            COLOR_PALETTE[m] = tuple(np.random.uniform(size=(3,))/2)
    if args.final:
        plot = sns.barplot(x="Model", y=y_label, data=pd.DataFrame(data), 
            ci=args.ci, order=model_order, palette=palette)
    else:
        plot = sns.lineplot(x="Step", y=y_label, estimator="mean", hue="Model", 
            hue_order=model_order, data=pd.DataFrame(data), ci=args.ci, palette=palette)
    if not args.no_log:
        plot.set_yscale('log')
    if args.norm:
        title = "Output Noise Norm"
        plot.set_ylim([.25,-.25])
        for text in plot.legend_.texts:
            legend_str = text.get_text()
            legend_str = legend_str.replace("SET", "Average of LSTM Outputs")
            text.set_text(legend_str)
    else:
        if args.signal:
            title = "Signal"
        elif args.noise:
            title = "Noise"
        else:
            title = "SNR"
        if not args.noise: # noise doesnt change in strong vs weak setting
            if args.strong:
                title = title + " Strong"
        if args.combo:
            title = "Gradient and SNR" #"Combo (" + title + ")"
        elif args.grad:
            title = "Gradient (" + title + ")"
        if args.final:
            title = "Final " + title
    if args.lower_ylim is not None:
        plot.set_ylim(bottom=args.lower_ylim)
    if args.upper_ylim is not None:
        plot.set_ylim(top=args.upper_ylim)
    if args.title: 
        fig.suptitle(title, fontsize=16)
    # fig.set_size_inches(13*0.4, 7*0.4) # useful for smaller plots in final paper
    fig.set_size_inches(13, 7)
    OUT_PATH_PNG = os.path.join(args.out_dir, args.out_name+".png")
    OUT_PATH_CSV = os.path.join(args.out_dir, args.out_name+".csv")
    # plt.savefig(OUT_PATH_PNG, dpi=700*(1/0.8), bbox_inches="tight") # useful for smaller plots in final paper
    plt.savefig(OUT_PATH_PNG, dpi=700, bbox_inches="tight")
    if args.final:
        with open(OUT_PATH_CSV, "w+") as file:
            metric_str = title
            file.write("Model;"+metric_str+"\n")
            for k, v in model_to_final_perfs.items():
                performance_str = ",".join([str(metric) for metric in v])
                file.write(str(k)+";"+performance_str+"\n")


def main():
    os.environ["CUDA_VISIBLE_DEVICES"]="-1" # Should run faster this way, since not batching

    default_exclude = ""
    parser = ArgumentParser()
    parser.add_argument("--scale", type=float, help="the magnitude of each component of the signal or noise", default=1)
    parser.add_argument("--exclude", type=str, help="model names to exclude seperated by commas with no spaces", default=default_exclude)
    parser.add_argument("--extra_exclude", type=str, help="model names to exclude in addition to exclude (which has defaults)", default="")
    parser.add_argument("--ci", type=int, help="confidence interval for plot", default=68)
    parser.add_argument("--num_agents", type=int, help="number of samples (initialized agents) to use when calculating SNR", default=3)
    parser.add_argument("--num_runs_per_agent", type=int, help="number of runs of an agent to estimate a single SNR", default=None)
    parser.add_argument("--length_of_decay", type=int, help="number of timesteps to feed in no stimulus (or noise)", default=None)
    parser.add_argument("--dummy", action='store_true', help="Whether to return a dummy DNR for testing", default=False)
    parser.add_argument("--signal", action='store_true', help="Whether to display plot of signal over time instead", default=False)
    parser.add_argument("--noise", action='store_true', help="Whether to display plot of noise over time instead", default=False)
    parser.add_argument("--final", action='store_true', help="Whether to plot just final values as a bar plot", default=False)
    parser.add_argument("--strong", action='store_true', help="Whether the signal obs occurs repeatedly and radnomly", default=False)
    parser.add_argument("--strong_p", type=float, help="Probability of a signal per step if strong", default=0.01)
    parser.add_argument("--strong_block", type=float, help="Fraction of steps at end of strnog with no signal", default=.2)
    parser.add_argument("--grad", action='store_true', help="Plot gradients instead of SNR", default=False)
    parser.add_argument("--combo", action='store_true', help="Multiply grad plot by an SNR plot", default=False)
    parser.add_argument("--norm", action='store_true', help="Compute output norm for one run, instead of SNR. Overrides other args.", default=False)
    parser.add_argument("--no_log", action='store_true', help="Do not plot in log scale", default=False)
    parser.add_argument("--out_name", type=str, help="filename for output saved in plts.", default="snr")
    parser.add_argument("--lower_ylim", type=float, help="Lower ylim for plot. E.g. 10e-30.", default=None)
    parser.add_argument("--upper_ylim", type=float, help="Upper ylim for plot. E.g. 1.", default=None)
    parser.add_argument("--sort", action='store_true', help="Whether to sort by performance for the legend on lineplots", default=False)
    parser.add_argument("--skip_ST", action='store_true', help="Whether to skip models with \"ST\" in the name if calculating SNR, since straight through connections have the same SNR as without", default=False)
    parser.add_argument("--title", action='store_true', help="Show title", default=False)
    parser.add_argument('--out_dir', type=str, help='the base log dir on machine', default=os.path.dirname(os.path.realpath(__file__)))

    args = parser.parse_args()

    # Clean up args
    args.out_dir = os.path.expanduser(args.out_dir)
    args.exclude = args.exclude + "," + args.extra_exclude
    if args.norm:
        args.noise = True
        args.num_agents = 1
        args.num_runs_per_agent = 1
        args.no_log = True
        args.final = False
    if args.num_runs_per_agent == None: # Defaults
        args.num_runs_per_agent = 1 if args.grad else 5
    if args.length_of_decay == None: # Defaults
        args.length_of_decay = 2000 if args.strong else 100
    assert not (args.signal and args.noise), "--signal and --noise are mutually exclusive"
    assert not args.grad or (args.noise or args.signal), "--grad must be evaluated in signal or noise env"
    assert not (args.combo and not args.grad), "--combo is a modification of grad plot"

    global AGENT_KEY
    if args.norm:
        AGENT_KEY = "Norm"
    elif args.grad:
        AGENT_KEY = "Gradient"
    elif args.signal:
        AGENT_KEY = "Signal"
    elif args.noise:
        AGENT_KEY = "Noise"
    else:
        AGENT_KEY = "SNR"

    # Read in models and calculate SNR
    results_dicts = get_all_results(args)

    # Plot SNR (or Gradient)
    make_plot(results_dicts, args)



if __name__ == "__main__":
    main()