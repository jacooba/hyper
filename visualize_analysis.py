'''
A script used to plot the gradients and other models over time for untrained models at initialization.
Adapted from the AMRL paper (https://iclr.cc/virtual_2020/poster_Bkl7bREtDr.html).
'''

from collections import defaultdict, OrderedDict
from argparse import ArgumentParser, Namespace
from gym.spaces import Discrete, Box
from sklearn.manifold import TSNE
from pathlib import Path
from glob import glob

import os
import zlib
import copy
import torch
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from visualize_runs import MODEL_ORDER, COLOR_PALETTE, simple_name
from metalearner import MetaLearner

from config.gridworld import \
    args_grid_oracle, args_grid_belief_oracle, args_grid_rl2, args_grid_varibad, args_TLN_varibad
from config.mujoco import \
    args_cheetah_dir_oracle, args_cheetah_dir_rl2, args_cheetah_dir_varibad, \
    args_cheetah_vel_oracle, args_cheetah_vel_rl2, args_cheetah_vel_varibad, args_cheetah_vel_avg, \
    args_ant_dir_oracle, args_ant_dir_rl2, args_ant_dir_varibad, \
    args_ant_goal_oracle, args_ant_goal_rl2, args_ant_goal_varibad, \
    args_walker_oracle, args_walker_avg, args_walker_rl2, args_walker_varibad

from config import default_conf
from experiment_sets.models import * # models

DEFAULT_CONFIG = default_conf
ENV_CONFIG = args_TLN_varibad # choose from above
SHARED_ARGS = {        
    "encoder_gru_hidden_size": "256",
    #
    "hypernet_input": "latent",
    "init_hyper_for_policy": True,
    #
    "tbptt_stepsize": None, 
    #
    "full_transitions": True,
    "policy_task_embedding_dim": 25,
    "policy_latent_embedding_dim": 25,
    "latent_dim": 12, # Note: This value will be doubled in models without variational inference. 
    } 
MODELS = [   
    RNN_HN,
    AMRL,
    SplAgger,
    SplAgger_noRNN,
]

# add parsed args to each model dict
for m_dict in MODELS:
    arg_strs = []
    for k, v in m_dict.items(): # add args from model
        arg_strs += ["--"+k]
        arg_strs += str(v).split(" ")
    for k, v in SHARED_ARGS.items(): # add shared args if not in model
        if k not in m_dict:
            arg_strs += ["--"+k]
            arg_strs += str(v).split(" ")
    m_dict["args"], unkown_args = ENV_CONFIG.get_args(arg_strs) # parse args
    # get defaults for unknown args
    default_args = default_conf.get_args(unkown_args)
    for default_arg_key, default_arg_value in default_args.__dict__.items():
        if not default_arg_key in m_dict["args"]:
            setattr(m_dict["args"], default_arg_key, default_arg_value)
    m_dict["exp_label"] = simple_name(m_dict["exp_label"])
    m_dict['args'].exp_label = simple_name(m_dict['args'].exp_label)


def calculate_quantity(agent_dict, args, agent_num, num_agents):
    """
        Over multiple initializations, compute the quantity to measure.
        Note: We call this quantity SNR (signal to noise ration), but it may be a
              different quantity, based on the arguments passed to this script.
    """
    SNRs = []
    for i in range(args.num_agents):
        agent_dict["config"].seed = i
        print("\n\nSeed:", i)
        print("Evaluating sample of agent {} of {}, named: {} ...\n\n".format(agent_num, num_agents, agent_dict["model_name"]))
        metalearner = MetaLearner(agent_dict["config"], make_logger=False)
        vae_enc = metalearner.vae.encoder
        # Adding bias affects SNR, but not most others
        if args.snr:
            for name, param in vae_enc.agg.named_parameters():
                if 'bias' in name:
                    torch.nn.init.constant_(param, 0)
        # Close envs
        metalearner.envs.close()
        metalearner.envs.close_extras()
        for p in metalearner.envs.ps:
            p.close()
        SNRs.append(calculate_for_policy(vae_enc, args))
    return SNRs

def calculate_for_policy(vae_enc, args):
    ones_input = np.ones(vae_enc.agg.input_size) * args.scale
    signals, noises = [], []
    signal_noises, antisignal_noises = [], [] # A trajectory of combines signal and noise, or -signal and noise. Used if args.cor.
    perm_noises = [] # used if args.perm_diff
    for _ in range(args.num_runs_per_agent):
        if not args.noise: # if not just noises, need signals
            signal_observations = np.zeros((args.length_of_decay, vae_enc.agg.input_size))
            if args.strong: # signal randomly and repeatedly
                for i in range(len(signal_observations)-int(args.strong_block*len(signal_observations))):
                    if np.random.rand() < args.strong_p:
                        signal_observations[i] = ones_input
            else:
                signal_observations[0] = ones_input # signal at beginning
            signals.append(get_states_from_obs(vae_enc, signal_observations, args.grad, param_grad=args.param_grad, inputs_grad=args.inputs_grad))
        if not args.signal: # if not just signals, need noises
            # noise_observations = [np.random.choice([1, -1], p=[0.5, 0.5]) * ones_input for _ in range(args.length_of_decay)]
            noise_observations = [np.random.uniform(-1,1,vae_enc.agg.input_size) * ones_input for _ in range(args.length_of_decay)]
            noises.append(get_states_from_obs(vae_enc, noise_observations, args.grad, param_grad=args.param_grad, inputs_grad=args.inputs_grad))
        if args.cor or args.tdist or args.lcc:
            signal_noise_obs     = [np.random.choice([1, -1], p=[0.5, 0.5]) * ones_input for _ in range(args.length_of_decay)]
            antisignal_noise_obs = [np.random.choice([1, -1], p=[0.5, 0.5]) * ones_input for _ in range(args.length_of_decay)]
            signal_noise_obs[0]     = ones_input
            antisignal_noise_obs[0] = ones_input*-1 # negative signal at start
            signal_noises.append(get_states_from_obs(vae_enc, signal_noise_obs, False))
            antisignal_noises.append(get_states_from_obs(vae_enc, antisignal_noise_obs, False))
        if args.perm_diff:
            perm_noise = []
            for i in range(args.length_of_decay):
                perm_obs_so_far = np.random.permutation(np.array(noise_observations[:i+1]))
                noises_so_far, _ = get_states_from_obs(vae_enc, perm_obs_so_far, False)
                if i != 0:
                    noises_so_far = noises_so_far[-1]
                perm_noise.append(noises_so_far)
            perm_noises.append(np.array(perm_noise))
            # perm_observations = np.random.permutation(np.array(noise_observations))
            # perm_noises.append(get_states_from_obs(vae_enc, perm_observations, False))
    if args.normalized_norm:
        outputs = signals if args.signal else noises
        runs, steps = zip(*outputs)
        assert len(runs) == 1, len(runs) # Don't want to compute an average over runs
        run, steps = runs[0], steps[0]
        run_norms = np.sqrt(np.sum(run**2, axis=1))
        mean_norm = np.mean(run_norms)
        trajectory = (run_norms - mean_norm)/mean_norm
        to_return = (trajectory, steps)
    elif args.norm or args.grad or args.param_grad or args.inputs_grad:
        # Note: if args.grad, gradient computed in get_states_from_obs above, but the norm of that is computed here.
        # Also, prior experiments removed "or args.grad" and allowed expected_powers() below in signal or noise to process grad.
        outputs = signals if args.signal else noises
        trajectory, steps = zip(*outputs)
        assert not np.isnan(trajectory).any(), trajectory
        norms = np.sum(np.array(trajectory)**2, axis=-1)**.5
        mean_norms = np.mean(norms, axis=0)
        to_return = (mean_norms, steps[0])
    elif args.perm_diff:
        noises, steps = zip(*noises)
        # perm_noises, _ = zip(*perm_noises)
        to_return = (perm_diff(np.array(noises), np.array(perm_noises)), steps[0])
    elif args.signal:
        to_return = expected_powers(signals, args)
    elif args.noise:
        to_return = expected_powers(noises, args)
    elif args.snd:
        signals, steps = zip(*signals)
        noises, _ = zip(*noises)
        assert (signals[0] == signals[-1]).all(), "All signals should be the same."
        to_return = (snd(np.array(signals[0:1]), np.array(noises)), steps[0])
    elif args.invv:
        noises, steps = zip(*noises)
        to_return = (invv(np.array(noises)), steps[0])
    elif args.cor:
        signal_noises, steps = zip(*signal_noises)
        antisignal_noises, _ = zip(*antisignal_noises)
        to_return = (cor(np.array(signal_noises), np.array(antisignal_noises)), steps[0])
    elif args.tdist:
        signal_noises, steps = zip(*signal_noises)
        antisignal_noises, _ = zip(*antisignal_noises)
        to_return = (tdist(np.array(signal_noises), np.array(antisignal_noises)), steps[0])
    elif args.lcc:
        signal_noises, steps = zip(*signal_noises)
        antisignal_noises, _ = zip(*antisignal_noises)
        to_return = (linear_classifier_confidence(np.array(signal_noises), np.array(antisignal_noises)), steps[0])
    else:
        (s, s_steps), (n, n_steps) = expected_powers(signals, args), expected_powers(noises, args)
        to_return = (s/n, s_steps)
        assert s_steps == n_steps, (s_steps, n_steps)
    steps_to_return = to_return[1]
    assert np.array(steps_to_return).shape == (args.length_of_decay,), "Must have a single dimension of proper length"
    return to_return

def get_states_from_obs(vae_enc, observations, grad, param_grad=False, inputs_grad=False, grad_step=1):
    observations = torch.tensor(observations, dtype=torch.float, requires_grad=True).reshape((len(observations), 1, len(observations[0])))
    state = vae_enc.agg.init_state(1)
    if inputs_grad: # Compute gradient of final output w.r.t. each input over time
        assert grad_step==1, "grad_step !=1 is not tested."
        recurrent_out, state = vae_enc.agg(observations, state)
        grads = []
        steps_to_return = range(0, len(observations), grad_step)
        for i in steps_to_return:
            g = torch.autograd.grad(torch.sum(recurrent_out[-1,:,:]), observations, create_graph=True, allow_unused=True)[0][i]
            grads.append(g)
        vals_to_return = torch.stack(grads, dim=0).squeeze().detach().numpy()
        steps_to_return = range(len(observations))
    elif param_grad: # Compute gradient of output w.r.t. parameters of aggregator over time
        assert grad_step==1, "grad_step !=1 is not tested."
        recurrent_out, state = vae_enc.agg(observations, state)
        params = list(vae_enc.agg.parameters())
        grads = []
        steps_to_return = range(0, len(observations), grad_step)
        for i in steps_to_return:
            g = torch.autograd.grad(torch.sum(recurrent_out[i,:,:]), params, create_graph=True, allow_unused=True)
            g = torch.cat([(torch.zeros(param.shape) if param_grad is None else param_grad).reshape((-1,)) for param_grad, param in zip(g, params)])
            grads.append(g)
        vals_to_return = torch.stack(grads, dim=0).squeeze().detach().numpy()
        steps_to_return = range(len(observations))
    elif grad: # Compute gradient of each output w.r.t. initial observation
        assert grad_step==1, "grad_step !=1 is broken. Will need to debug later."
        recurrent_out, state = vae_enc.agg(observations, state)
        # recurrent_out_mean, recurrent_out_logvar = torch.chunk(recurrent_out, 2, dim=-1)
        grads = []
        steps_to_return = range(0, len(observations), grad_step)
        for i in steps_to_return:
            try:
                g = torch.autograd.grad(torch.sum(recurrent_out[i,:,:]), observations, allow_unused=True, retain_graph=True)[0][0]
            except Exception as e:
                print(e)
                import pdb; pdb.set_trace()
            grads.append(g)
        vals_to_return = torch.stack(grads, dim=0).squeeze().detach().numpy()
        steps_to_return = range(len(observations))
    else:
        recurrent_out, state = vae_enc.agg(observations, state)
        assert recurrent_out.shape[1] == 1, recurrent_out.shape
        assert recurrent_out.shape[0] == len(observations), recurrent_out.shape
        # recurrent_out_mean, recurrent_out_logvar = torch.chunk(recurrent_out, 2, dim=-1) # to share comparison with VI methods
        # to sample, do:
            # sample = vae_enc.reparameterise(recurrent_out_mean, recurrent_out_logvar)
        # vals_to_return = recurrent_out_mean.squeeze().detach().numpy()
        vals_to_return = recurrent_out.squeeze().detach().numpy()
        steps_to_return = range(len(observations))
    vals_to_return = vals_to_return.astype(np.float64) # Need the precision soon
    return vals_to_return, tuple(steps_to_return)

def cor(signal_noises, antisignal_noises):
    """ 
    Compute 1 - Cor(signal_noises, antisignal_noises).
    return a vector where each element represents the average correlation over features for each corresponding timestep. 
    signal and nosies shape: (samples, timesteps, features)
    """    
    mean_signal = np.mean(signal_noises, axis=0)
    std_signal = np.std(signal_noises, axis=0)
    # import pdb; pdb.set_trace()

    mean_antisignal = np.mean(antisignal_noises, axis=0)
    std_antisignal = np.std(antisignal_noises, axis=0)

    covariance = np.mean((signal_noises - mean_signal) * (antisignal_noises - mean_antisignal), axis=0)
    correlation = covariance / (std_signal * std_antisignal)

    return np.mean(1 - correlation, axis=1) # Compute 1 - correlation, and average over features

def perm_diff(noises, perm_noises, eps=1e-30):
    """ Normalize outputs and compute average difference between permutations """

    all_samples = np.concatenate((noises, perm_noises), axis=0)
    mean = np.mean(all_samples, axis=0, keepdims=True)
    std = np.std(all_samples, axis=0, keepdims=True)
    std = np.clip(std, eps, None)

    # normalized_noises = (noises-mean)/std
    # normalized_perm_noises = (perm_noises-mean)/std
    normalized_noises = noises / np.clip(np.linalg.norm(noises, axis=-1, keepdims=True), 1e-30, None)
    normalized_perm_noises = perm_noises / np.clip(np.linalg.norm(perm_noises, axis=-1, keepdims=True), 1e-30, None)

    diffs = np.linalg.norm(normalized_noises-normalized_perm_noises, axis=-1, keepdims=False)
    mean_diff = np.mean(diffs, axis=0)

    return mean_diff

def tdist(signal_noises, antisignal_noises, k=2):
    # Reshape data to 2D (samples*timesteps, features)
    signal_flat = signal_noises.reshape(-1, signal_noises.shape[2])
    antisignal_flat = antisignal_noises.reshape(-1, antisignal_noises.shape[2])

    # Concatenate the datasets
    combined_data = np.concatenate([signal_flat, antisignal_flat], axis=0)

    # Compute the mean and standard deviation for normalization
    mean = np.mean(combined_data, axis=0)
    std = np.std(combined_data, axis=0)

    # Normalize the data
    combined_data_normalized = (combined_data - mean) / std

    # Apply t-SNE
    tsne = TSNE(n_components=k)
    combined_tsne = tsne.fit_transform(combined_data_normalized)

    # Split the t-SNE output back into signal and antisignal
    signal_tsne = combined_tsne[:signal_flat.shape[0]]
    antisignal_tsne = combined_tsne[signal_flat.shape[0]:]

    # Reshape back to the original shape (samples, timesteps, k)
    signal_tsne = signal_tsne.reshape(signal_noises.shape[0], signal_noises.shape[1], k)
    antisignal_tsne = antisignal_tsne.reshape(antisignal_noises.shape[0], antisignal_noises.shape[1], k)

    # Compute distances at each point in time
    distances = np.linalg.norm(signal_tsne - antisignal_tsne, axis=-1)

    # Average the distances over samples
    avg_distances = np.mean(distances, axis=0)

    return avg_distances

def linear_classifier_confidence(signal_noises, antisignal_noises):
    # Flatten the data to 2D (samples*timesteps, features)
    signal_flat = signal_noises.reshape(-1, signal_noises.shape[-1])
    antisignal_flat = antisignal_noises.reshape(-1, antisignal_noises.shape[-1])

    # Concatenate the datasets for normalization
    combined_data = np.concatenate([signal_flat, antisignal_flat], axis=0)

    # Compute the mean and standard deviation for normalization
    mean = np.mean(combined_data, axis=0)
    std = np.std(combined_data, axis=0)

    # Normalize the data
    signal_flat_normalized = (signal_flat - mean) / std
    antisignal_flat_normalized = (antisignal_flat - mean) / std

    # Concatenate the normalized datasets and create labels
    x = np.concatenate([signal_flat_normalized, antisignal_flat_normalized], axis=0)
    y = np.concatenate([np.ones(signal_flat.shape[0]), np.zeros(antisignal_flat.shape[0])])

    # Add a bias term
    x = np.hstack([x, np.ones((x.shape[0], 1))])

    # Solve for the parameters using the pseudo-inverse
    params = np.linalg.pinv(x.T @ x) @ x.T @ y

    # Apply the classifier to the normalized signal data
    signal_with_bias = np.hstack([signal_flat_normalized, np.ones((signal_flat_normalized.shape[0], 1))])
    confidences = signal_with_bias @ params

    # Reshape confidences back to original shape (samples, timesteps)
    confidences_reshaped = confidences.reshape(signal_noises.shape[0], signal_noises.shape[1])

    # Plot the average confidence for each timestep
    avg_confidences = np.mean(confidences_reshaped, axis=0)

    return avg_confidences


def snd(signal, noises):
    # Compute En[(s-n)^2]  # optionally, modify to divide: /E(n**2)
    #   signal and nosies shape: (samples, timesteps, features)
    numerator = np.mean((signal-noises)**2, axis=-1) # (s-n)^2 across features
    denominator = np.mean(noises**2, axis=-1) # n across features
    E = np.mean(numerator, axis=0) # Expected numerator, across runs
    E_n = np.mean(denominator, axis=0) # Expectation of n**2, across runs
    return E # /E_n

def invv(noises):
    # Compute 1/V(n)
    #   signal and nosies shape: (samples, timesteps, features)
    denominator = np.mean(noises, axis=-1) # n across features
    V = np.var(denominator, axis=0) # Variance of n, across runs
    return V**-1

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
            grads = calculate_quantity(agent_dict, args, agent_num=agent_num+1, num_agents=len(models))
            snr_args = Namespace(**vars(args))
            snr_args.grad = False
            snr_args.signal = False
            snr_args.noise = False
            snrs = calculate_quantity(agent_dict, snr_args, agent_num=agent_num+1, num_agents=len(models))
            for grad_run, snr_run in zip(grads, snrs):
                grad_vals, grad_steps = grad_run
                snr_vals, snr_steps = snr_run
                assert snr_steps == tuple(range(args.length_of_decay)), (snr_steps, tuple(range(args.length_of_decay)))
                chosen_snr_vals = np.array([snr_vals[i] for i in grad_steps])
                combo_vals = chosen_snr_vals*grad_vals
                agent_dict[AGENT_KEY].append((combo_vals, grad_steps))
        else:
            agent_dict[AGENT_KEY] = calculate_quantity(agent_dict, args, agent_num=agent_num+1, num_agents=len(models))
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
    elif args.normalized_norm:
        y_label = "(Norm - Mean)/Mean"
    elif args.norm:
        y_label = "Norm"
    elif args.perm_diff:
        y_label = "Permutation Difference"
    elif args.grad:
        y_label = "Initial Input Gradient Norm"
    elif args.param_grad:
        y_label = "Parameter Gradient Norm"
    elif args.inputs_grad:
        y_label = "All Inputs Gradient Norm"
    elif args.signal:
        y_label = "Signal Power"
    elif args.noise:
        y_label = "Noise Power"
    elif args.snd:
        y_label = "SND"
    elif args.invv:
        y_label = "INVV"
    elif args.tdist:
        y_label = "t-SNE Dist"
    elif args.lcc:
        y_label = "Linear Classifier Confidence"
    elif args.cor:
        y_label = "Cor"
    else:
        assert args.snr
        y_label = "snr"
    # get data in right format
    model_to_final_perfs = defaultdict(list) if args.final else None
    for agent_dict in results_dicts:
        if agent_dict["model_name"] not in MODEL_ORDER:
            print("Warning, model, {}, not in model order.".format(agent_dict["model_name"]))
            MODEL_ORDER.append(agent_dict["model_name"])
        if agent_dict["model_name"] not in COLOR_PALETTE:
            print("Warning: model name,", agent_dict["model_name"], "not in COLOR_PALETTE. It will be added with a random color.")
            h = zlib.adler32(agent_dict["model_name"].encode()) # cannot use python hash(), since it is stochastic
            r = np.random.default_rng(seed=abs(h)) # use hash of model name as seed so colors are reproducible between runs
            COLOR_PALETTE[agent_dict["model_name"]] = tuple(.75 * r.uniform(size=(3,)))
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
        # Plot in reverse model order so the first model appears on top
        plot = sns.lineplot(x="Step", y=y_label, estimator="mean", hue="Model", 
            hue_order=reversed(model_order), data=pd.DataFrame(data), ci=args.ci, palette=palette)
        # Adjust the legend - reverse the order, so they appear in model_order
        handles, labels = plot.get_legend_handles_labels()
        plt.legend(handles[::-1], labels[::-1])
    if not args.no_log:
        plot.set_yscale('log')
    if args.normalized_norm:
        title = "Output Noise Norm"
        plot.set_ylim([.25,-.25])
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
        elif args.param_grad:
            title = "Parameter Gradient (" + title + ")"
        elif args.inputs_grad:
            title = "All Gradient (" + title + ")"
        if args.final:
            title = "Final " + title
    if args.lower_ylim is not None:
        plot.set_ylim(bottom=args.lower_ylim)
    if args.upper_ylim is not None:
        plot.set_ylim(top=args.upper_ylim)
    if args.title: 
        fig.suptitle(title, fontsize=16)
    fig.set_size_inches(4, 2.6) # useful for much smaller plots in final paper
    # fig.set_size_inches(13, 7)
    OUT_PATH_PNG = os.path.join(args.out_dir, args.out_name+".png")
    OUT_PATH_CSV = os.path.join(args.out_dir, args.out_name+".csv")
    plt.savefig(OUT_PATH_PNG, dpi=1000, bbox_inches="tight") # useful for much smaller plots in final paper
    # plt.savefig(OUT_PATH_PNG, dpi=700, bbox_inches="tight")
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
    parser.add_argument("--snr", action='store_true', help="Plot the SNR.", default=False)
    parser.add_argument("--grad", action='store_true', help="Plot gradients with respect to initial input.", default=False)
    parser.add_argument("--param_grad", action='store_true', help="Plot gradients with respect to parameters over time.", default=False)
    parser.add_argument("--inputs_grad", action='store_true', help="Plot gradients of final output with respect to each input over time.", default=False)
    parser.add_argument("--snd", action='store_true', help="Plot signal noise difference (novel formula).", default=False)
    parser.add_argument("--perm_diff", action='store_true', help="Plot average difference in permutation encodings.", default=False)
    parser.add_argument("--invv", action='store_true', help="Plot inverse variance of noise.", default=False)
    parser.add_argument("--cor", action='store_true', help="Plot correlation of the signals over time.", default=False)
    parser.add_argument("--tdist", action='store_true', help="Plot distance of t-SNEover times instead of SNR", default=False)
    parser.add_argument("--lcc", action='store_true', help="Plot confidence of linear classifier over time.", default=False)
    parser.add_argument("--combo", action='store_true', help="Multiply grad plot by an SNR plot", default=False)
    parser.add_argument("--normalized_norm", action='store_true', help="Compute norm of the observations for one run, and normalize over time.", default=False)
    parser.add_argument("--norm", action='store_true', help="Compute output norms.", default=False)
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
    if args.normalized_norm:
        args.noise = True
        args.num_agents = 1
        args.num_runs_per_agent = 1
        args.no_log = True
        args.final = False
    if args.num_runs_per_agent == None: # Defaults
        args.num_runs_per_agent = 1 if args.grad or args.inputs_grad or args.param_grad else 5
    if args.length_of_decay == None: # Defaults
        args.length_of_decay = 2000 if args.strong else 100
    assert not (args.signal and args.noise), "--signal and --noise are mutually exclusive"
    if args.norm or args.grad or args.param_grad or args.inputs_grad:
        assert (args.noise or args.signal), "--grad must be evaluated in signal or noise env"
    assert not (args.combo and not args.grad), "--combo is a modification of grad plot"
    assert (int(args.norm) + int(args.normalized_norm) + int(args.snr) + int(args.snd) + int(args.invv) + int(args.cor) + int(args.grad) + int(args.param_grad) + int(args.inputs_grad) + int(args.tdist) + int(args.perm_diff)) == 1, "Exactly one quantity to measure must be specified."

    global MODELS
    for m_dict in MODELS:
        m_dict["args"].max_trajectory_len = args.length_of_decay

    global AGENT_KEY
    if args.normalized_norm:
        AGENT_KEY = "Norm_Norm"
    if args.norm:
        AGENT_KEY = "Norm"
    if args.perm_diff:
        AGENT_KEY = "Perm_Diff"
    elif args.grad:
        AGENT_KEY = "Gradient"
    elif args.param_grad:
        AGENT_KEY = "Parameter Gradient"
    elif args.inputs_grad:
        AGENT_KEY = "All Gradient"
    elif args.signal:
        AGENT_KEY = "Signal"
    elif args.noise:
        AGENT_KEY = "Noise"
    elif args.snd:
        AGENT_KEY = "SND"
    elif args.invv:
        AGENT_KEY = "INVV"
    elif args.tdist:
        AGENT_KEY = "TDIST"
    elif args.lcc:
        AGENT_KEY = "LCC"
    elif args.cor:
        AGENT_KEY = "COR"
    else:
        AGENT_KEY = "SNR"

    # Read in models and calculate SNR
    results_dicts = get_all_results(args)

    # Plot SNR (or Gradient)
    make_plot(results_dicts, args)



if __name__ == "__main__":
    main()