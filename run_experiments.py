"""
Script to schedule all experiments that have not been completed. 
Will wait for more local resources if not enough. 
Can be run on multiple servers.

Note:
    Some structure of VariBAD logs is assumed.
    VariBAD script must be modified out a file called "DONE" in its terminated experiments.
        It is also recommended that a unique id (that ignores the set seed) is added to the log directory name to avoid time collision, e.g. hex(random.Random().randint(0,1e6)).replace("0x","")
    Plotting takes an key for the learning rate and will plot best over all learning rates. (only one tuned lr at a time is supported, default=lr_vae)
    It is assumed that data and repos are in the same location on all servers (other than server names)
    See argument defaults for docker scripts commands that are required

Example cmd:
    This script generates cmds such as:
    ~/jake_docker/run_cpu.sh mujoco150 0 python ~/hyper/main.py --env-type gridworld_varibad --results_log_dir /users/jaceck/data/dummy/ --exp_label test --seed 73

Experiments should be defined as example below.
"""

import re
import os
import glob
import json
import math
import signal
import atexit
import shutil
import socket
import getpass
import argparse
import itertools
import subprocess

from multiprocessing import Pool
from time import sleep, time
from random import shuffle, sample
from select import select


CPU_EXPERIMENT_SETS = [] # This will be a list of sets of experiments for cpu only
GPU_EXPERIMENT_SETS = [] # This will be a list of sets of experiments for gpu


THIS_DIR = os.path.dirname(os.path.realpath(__file__))


# Take in a cpu or gpu set and get the cmd args
def get_cmds(args, experiment_sets):
    cmds = []
    cmdStr_2_argDict, cmdStr_2_mujoco = {}, {}
    for experiment_set in experiment_sets:
        log_dir = os.path.join(args.docker_log_base_dir, experiment_set["dir_name"])
        for experiment in experiment_set["experiments"]:
            # create Cartesian product over search parameters
            search_params = itertools.product(*experiment_set["search_arguments"].values())
            for search_param_setting in search_params:
                arg_dict = {}
                # add shared params other parameters
                arg_dict.update(experiment_set["shared_arguments"])
                # add params for this experiment
                arg_dict.update(experiment) # this will override any shared params
                # add this combination to the args dict. Will overwrite prior two.
                named_search_params = {list(experiment_set["search_arguments"].keys())[i] : param_value for i, param_value in enumerate(search_param_setting)}
                arg_dict.update(named_search_params)
                arg_dict.update({"results_log_dir": log_dir})
                cmd = []
                for key, value in sorted(list(arg_dict.items())):
                    cmd.append("--"+key)
                    cmd.append(str(value))
                cmds.append(cmd)
                cmd_str = " ".join(cmd)
                cmdStr_2_argDict[cmd_str] = arg_dict
                cmdStr_2_mujoco[cmd_str] = str(experiment_set["mujoco_version"]) if "mujoco_version" in experiment_set else args.default_mujoco_version
    return cmds, cmdStr_2_argDict, cmdStr_2_mujoco

def is_match(exper_dict_1, exper_dict_2, override_logdir_1=None, override_logdir_2=None):
    # defines a match between exp configs
    # Note: override_logdir useful if directories have been moved, but we still need to check name of exp set
    logdir1 = exper_dict_1["results_log_dir"] if override_logdir_1 is None else override_logdir_1
    logdir2 = exper_dict_2["results_log_dir"] if override_logdir_2 is None else override_logdir_2
    return (os.path.basename(os.path.normpath(logdir1)) == \
            os.path.basename(os.path.normpath(logdir2))) \
        and (exper_dict_1["env_name"] == exper_dict_2["env_name"]) \
        and (exper_dict_1["exp_label"] == exper_dict_2["exp_label"]) \
        and (exper_dict_1["seed"] == exper_dict_2["seed"]) \
        and (exper_dict_1["lr_vae"] == exper_dict_2["lr_vae"]) \
        and (exper_dict_1["lr_policy"] == exper_dict_2["lr_policy"]) 

def pretty_str(arg_dict):
    return "{} {} {}; lr_v {}; lr_p {}; seed {}".format(
                arg_dict["env_name"],
                arg_dict["exp_label"],
                os.path.basename(os.path.normpath(arg_dict["results_log_dir"])),
                arg_dict["lr_vae"],
                arg_dict["lr_policy"],
                arg_dict["seed"],
                )

def filter_prior_success(cmds_lists, cmdStr_2_argDict, args, unmatched_warning=False, quiet=False):
    if not quiet:
        print("\nInspecting Prior Experiments...", flush=True)
    # Find local directories with results for each individual run of an experiment
    exp_dirs = set()
    for _, arg_dict in cmdStr_2_argDict.items():
        log_dir = args.log_base_dir_here # on machine
        dir_name = os.path.basename(os.path.normpath(arg_dict["results_log_dir"])) # within docker
        log_dir = os.path.join(log_dir, dir_name)
        exp_dirs_in_envs = glob.glob(os.path.normpath(log_dir) + "/*/*/") # log_base_dir/dir_name/env_name/*/
        exp_dirs.update(exp_dirs_in_envs) # collection of all directories for individual runs
    
    # Find prior success and failures
    successful_config_dir_tup = [] # List of tuples of successful experiment configs and their directory
    failed_dirs = []
    for d in exp_dirs:
        is_done, has_conf = os.path.exists(os.path.join(d, "DONE")), os.path.exists(os.path.join(d, "config.json"))
        if is_done and has_conf:
            json_path = os.path.join(d, "config.json")
            with open(json_path, "r") as json_config:
                success_dict_1 = json.load(json_config) # config from successful experiments
                successful_config_dir_tup.append((success_dict_1, d))
        else:
            failed_dirs.append(d)
            if is_done:
                print("Found directory DONE but no config. Likely SCP was unfinished. Please remove:", d)
    
    # Remove prior success from commands (and check for duplicates or unmatched prior experiments)
    new_cmds_lists = []
    duplicate_dirs = []
    success_dirs = set()
    unmatched_successful_dirs = set(suc_d for _, suc_d in successful_config_dir_tup)
    for cmd_list in cmds_lists:
        new_cmds = []
        for cmd in cmd_list:
            arg_dict = cmdStr_2_argDict[" ".join(cmd)]
            matches = []
            for success_dict, suc_d in successful_config_dir_tup:
                if is_match(arg_dict, success_dict, # override the logdir to reflect where we found it...
                        override_logdir_2=os.path.dirname(os.path.dirname(os.path.normpath(suc_d)))): 
                    matches.append(suc_d)
                    if suc_d in unmatched_successful_dirs:
                        unmatched_successful_dirs.remove(suc_d) # This successful prior exp has a matching cmd
            if matches:
                # too much to print:
                # if not quiet:
                #     print("Found match; skipping exp:", pretty_str(arg_dict))
                success_dirs.update(matches)
                if len(matches) > 1:
                    duplicate_dirs.append(tuple(matches))
            else:
                new_cmds.append(cmd)
        new_cmds_lists.append(new_cmds)
    
    # Warn about any successful runs in experiment set that don't match any commands
    if unmatched_warning and unmatched_successful_dirs:
        print(("\n\nWarning: Prior successful experiments exist in an experiment set that are not specified by that set.\n"
                "If this is a mistake, it may mess up plots. You may want to move or delete these:\n"
                ), unmatched_successful_dirs, "\n\n")
    
    # Done
    if not quiet:
        print("Done Inspecting", flush=True)
    return new_cmds_lists, list(success_dirs), failed_dirs, duplicate_dirs

def confirm_clean(bad_dirs, dir_type):
    assert dir_type in ["failed", "duplicate"]
    if dir_type == "failed":
        to_remove_dirs = bad_dirs
        msg = ("\nFound failed experiments. Please add done files or remove them. Would you like to automatically"
         " remove the following dirs?\n{}").format("\n".join(bad_dirs))
    else:
        to_remove_dirs = [d for duplicates in bad_dirs for d in duplicates[1:]]
        to_remove_dirs = list(set(to_remove_dirs))
        msg = ("\nFound duplicate experiments:\n{}\n Please remove all but one for each set.\n"
            "Would you like to pick arbitrarily, removing the following dirs:\n{}").format(bad_dirs, to_remove_dirs)
    if bad_dirs:
        print(msg, flush=True)
        ans = input()
        if ans.lower() == "y" or ans.lower() == "yes":
            #remove failed experiments
            for d in to_remove_dirs:
                shutil.rmtree(d) 
        exit()


def plot(args, free_cpus, exps_names=None, async_procs=None):
    # Plot all learning curves
    print("\r\n\nPlotting...", flush=True)
    plot_dir_here = os.path.join(args.log_base_dir_here, "plts")
    os.makedirs(plot_dir_here, exist_ok=True) # ensure plot dir exists here
    plot_dir = os.path.join(args.docker_log_base_dir, "plts") # convert to docker path
    if args.plot_dir is None:
        assert exps_names is not None, exps_names
        env_dirs = [p for e in exps_names for p in glob.glob(os.path.normpath(args.log_base_dir_here) + "/"+e+"/logs_"+args.plot_env+"/")]
    else:
        env_dirs = glob.glob(os.path.normpath(args.log_base_dir_here) + "/"+args.plot_dir+"/logs_"+args.plot_env+"/")
    for dir_to_plot in env_dirs:
        for tag in args.plot_tags:
            dir_to_plot = os.path.normpath(dir_to_plot)
            set_dir, env_name = os.path.basename(os.path.dirname(dir_to_plot)), os.path.basename(dir_to_plot)
            dir_to_plot = os.path.join(args.docker_log_base_dir, set_dir, env_name) # convert to docker path
            if set_dir == "plts":
                continue
            tag_str = "_default" if tag is None else "_"+tag
            tag_str = tag_str.replace("/","_") # / in out path will mess up saving in the plotting script
            plot_out_path = os.path.join(plot_dir, set_dir+"_"+env_name+tag_str).replace("_logs", "")
            cmd_str = args.docker_util_cmd + " " + args.viz_python_cmd + " " + dir_to_plot + \
                      " --no_show " + "--out " + plot_out_path + " --title " + "--lr_keys " + args.lr_keys
            # e.g. cmd_str:
            # ~/jake_docker/run_cpu.sh mujoco150 20 python ~/hyper/visualize_runs.py /users/jaceck/data/hyper/aggregate_single_LR_short_pns60/logs_T-LN-v0 --no_show --out /users/jaceck/data/hyper/plts/aggregate_single_LR_short_pns60_T-LN-v0 --best --title --lr_key lr_vae
            if env_name == "logs_T-LN-P1-v0" and set_dir == "HML1": # override specific to plotting this experiment
                cmd_str += " --step " + "750"
            if args.plot_exclude:
                cmd_str += " --exclude " + args.plot_exclude
            if args.plot_include:
                cmd_str += " --only " + args.plot_include
            if args.low_y_lim:
                cmd_str += " --lower_ylim " + args.low_y_lim
            if args.up_y_lim:
                cmd_str += " --upper_ylim " + args.up_y_lim
            if args.smooth:
                cmd_str += " --smoothing_sz " + args.smooth
            if args.step:
                cmd_str += " --step " + args.step
            if args.metric_tag:
                cmd_str += " --metric_tag " + args.metric_tag
            if args.indiv:
                cmd_str += " --indiv"
            if args.final_r:
                cmd_str += " --final_r"
            if args.no_legend:
                cmd_str += " --no_legend"
            if args.log_scale:
                cmd_str += " --log_scale"
            if args.shift_warmups:
                cmd_str += " --shift_warmups"
            if not args.all_lr:
                cmd_str += " --best"
            if tag:
                cmd_str += " --tag " + tag
            try:
                print("\r\nRunning plot cmd:\n", cmd_str, flush=True)    
                if async_procs is None:
                    plot_output = subprocess.check_output(cmd_str, stderr=subprocess.STDOUT, shell=True)
                else:
                    async_procs.append(subprocess.Popen(cmd_str, stderr=subprocess.STDOUT, shell=True))
                    plot_output = ""
            except Exception as e:
                print("Error in plot", e)
                if hasattr(e, "output"):
                    print(e.output.decode())
                print("Will skip these dirs:", env_dirs)
                print("Used cmd:", cmd_str)
                continue
            if async_procs is None:
                if "No data found" in plot_output.decode():
                    print("No data for plot.")
                    print("Dir:", dir_to_plot)
                    print("Cmd:", cmd_str)
                else:
                    print("\r"+plot_output.decode(), flush=True)
    if async_procs is not None:
        while async_procs:
            print("\rWaiting on a plot...", flush=True)
            async_procs.pop().wait()


def filter_and_clean(cpu_cmds, gpu_cmds, cmdStr_2_argDict, args):
    # Remove previously successful cmds and failed experiments
    num_total_cmds = len(cpu_cmds)+len(gpu_cmds)
    # filter and clean
    (cpu_cmds,gpu_cmds), _, failed_dirs, duplicate_dirs = filter_prior_success([cpu_cmds,gpu_cmds], cmdStr_2_argDict, args, unmatched_warning=True, quiet=False)
    confirm_clean(failed_dirs, "failed")
    confirm_clean(duplicate_dirs, "duplicate")
    assert not failed_dirs, failed_dirs
    assert not duplicate_dirs, duplicate_dirs
    num_new_cmds = len(cpu_cmds)+len(gpu_cmds)
    print("\nFound {} new cmds to run.".format(num_new_cmds))
    print("Found {} cmds total.".format(num_total_cmds))
    print("{} new on CPU.".format(len(cpu_cmds)))
    print("{} new on GPU.".format(len(gpu_cmds)))
    print("new cmds:")
    for cmd in cpu_cmds+gpu_cmds:
        print(pretty_str(cmdStr_2_argDict[" ".join(cmd)]))
    return cpu_cmds, gpu_cmds

def get_free_devices(args, force_detect=False):
    low_cpu_free, hi_cpu_free = args.cpu_free.split("-")
    free_cpus = set(range(int(low_cpu_free), int(hi_cpu_free)+1))
    print("Free CPUs Given:", free_cpus)
    low_gpu_free, hi_gpu_free = args.gpu_free.split("-")
    free_gpus = set(range(int(low_gpu_free), int(hi_gpu_free)+1))
    print("Free GPUs Given:", free_gpus)
    return free_cpus, free_gpus

def get_cpu_and_gpu_cmds(args):
    cmdStr_2_argDict, cmdStr_2_mujoco = {}, {}
    if args.force_gpu:
        cpu_cmds, cpuCmdStr_2_argDict, cpuCmdStr_2_mujoco = get_cmds(args, [])
        gpu_cmds, gpuCmdStr_2_argDict, gpuCmdStr_2_mujoco = get_cmds(args, CPU_EXPERIMENT_SETS+GPU_EXPERIMENT_SETS)
    else:
        cpu_cmds, cpuCmdStr_2_argDict, cpuCmdStr_2_mujoco = get_cmds(args, CPU_EXPERIMENT_SETS)
        gpu_cmds, gpuCmdStr_2_argDict, gpuCmdStr_2_mujoco = get_cmds(args, GPU_EXPERIMENT_SETS)
    cmdStr_2_argDict.update(cpuCmdStr_2_argDict)
    cmdStr_2_argDict.update(gpuCmdStr_2_argDict)
    cmdStr_2_mujoco.update(cpuCmdStr_2_mujoco)
    cmdStr_2_mujoco.update(gpuCmdStr_2_mujoco)
    return cpu_cmds, gpu_cmds, cmdStr_2_argDict, cmdStr_2_mujoco

def remove_term_procs(devID_2_procs, pid_2_cmd, cmdStrs_2_stopTime, cmdStr_2_argDict, active_cmds_here_file, async_procs, args):
    # Remove procs that have terminated from device assignment
    cmds_terminated = []
    cmds_failed = []
    for dev_id, procs in list(devID_2_procs.items()):
        procs_running = []
        for p in procs:
            if p.poll() is None:
                procs_running.append(p)
            else:
                cmd_terminated = pid_2_cmd[p.pid]
                print("\rExp terminated:", pretty_str(cmdStr_2_argDict[" ".join(cmd_terminated)]))
                print("\ri.e. cmd:", " ".join(cmd_terminated))
                _, matched_dirs, _, _ = filter_prior_success([[cmd_terminated]], cmdStr_2_argDict, args, quiet=True)
                success = len(matched_dirs) >= 1
                if success:
                    cmds_terminated.append(cmd_terminated)
                else:
                    cmds_failed.append(cmd_terminated)
                cmdStrs_2_stopTime[" ".join(cmd_terminated)] = time()
        devID_2_procs[dev_id] = procs_running
    return cmds_terminated, cmds_failed

def expand_dev_2_procs(more_free_IDs, devID_2_procs):
    for dev_id in more_free_IDs:
        if dev_id not in devID_2_procs:
            devID_2_procs[dev_id] = []

def update_active_cmds(add_cmds, active_cmds_here_file, cmdStr_2_argDict, args, async_procs=None):
    for cmd in add_cmds:
        cmd_str = " ".join(cmd)
        with open(active_cmds_here_file, "a+") as f:
            f.write(cmd_str+"\n")
    with open(active_cmds_here_file, "r") as f:
        cmd_set = set(f.read().split("\n"))
        if "" in cmd_set:
            cmd_set.remove("")
    print("\r\nNumber of running cmds:", len(cmd_set))

def fill_devices(cmds, procs_per_dev, devID_2_procs, docker_cmd, pid_2_cmd, pid_2_devID, 
        cmdStr_2_argDict, cmdStr_2_startTime, cmdStr_2_mujoco, active_cmds_here_file, async_procs, args, sleep_time=.2):
    cmds_left = cmds[:]
    slots_left = True
    started_a_cmd = False
    while slots_left and cmds_left:
        slots_left = False
        for dev_id, procs in list(devID_2_procs.items()):
            if (len(procs) < procs_per_dev) and cmds_left:
                slots_left = True
                next_cmd = cmds_left.pop()
                mujoco_version_str = cmdStr_2_mujoco[" ".join(next_cmd)]
                cmdStr_2_startTime[" ".join(next_cmd)] = time()
                full_next_cmd = [docker_cmd+mujoco_version_str, str(dev_id), args.python_cmd] + next_cmd
                full_cmd_str = " ".join(full_next_cmd)
                try:
                    stdout = subprocess.DEVNULL if args.devnull else subprocess.PIPE
                    new_proc = subprocess.Popen(full_cmd_str, stdout=stdout, stderr=subprocess.PIPE, shell=True)
                    os.set_blocking(new_proc.stdout.fileno(), False) # so we can read from these later
                    os.set_blocking(new_proc.stderr.fileno(), False)
                except Exception as e:
                    print("\rError starting next experiment in main loop:", e)
                    if hasattr(e, "output"):
                        print(e.output)
                    print("\rWill skip this cmd:", full_cmd_str)
                    print("\ri.e. exp:", pretty_str(cmdStr_2_argDict[" ".join(next_cmd)]))
                    continue 
                async_procs.append(new_proc)
                procs.append(new_proc)
                pid_2_cmd[new_proc.pid] = next_cmd
                pid_2_devID[new_proc.pid] = dev_id
                print("\r\nRan exp:", pretty_str(cmdStr_2_argDict[" ".join(next_cmd)]))
                print("\ri.e. cmd: {}".format(full_cmd_str), flush=True)
                started_a_cmd = True
                update_active_cmds([next_cmd], active_cmds_here_file, cmdStr_2_argDict, args, async_procs=async_procs)
                sleep(sleep_time)
    return cmds_left, started_a_cmd

def pretty_print_cmdstrs(cmd_strs, cmdStr_2_argDict, subsample=False, subsample_num=6):
    if subsample and len(cmd_strs) > subsample_num:
        print("\r"+"... some examples out of {} ...".format(len(cmd_strs)))
        cmd_strs = sample(cmd_strs, subsample_num)
    for c_str in cmd_strs:
        if c_str in cmdStr_2_argDict:
            print("\r"+pretty_str(cmdStr_2_argDict[c_str]))
        else:
            print("\r"+c_str)

def print_loop_updates(cmds_terminated, cmds_failed, running_cmds, cmds_remaining, last_print_time, pid_2_devID,
                       num_done_at_last_print, cmdStr_2_argDict, cmdStr_2_startTime, cmdStrs_2_stopTime, async_procs, pid_2_cmd, pid_2_printInfo):
    cmdStr_2_pid = {" ".join(cmd): pid for pid, cmd in pid_2_cmd.items()}
    # print output from async_procs
    output_streams =  [(proc.stdout, proc.pid) for proc in async_procs if proc.pid in pid_2_cmd]
    output_streams += [(proc.stderr, proc.pid) for proc in async_procs if proc.pid in pid_2_cmd]
    procs_printed = False
    for stream, pid in output_streams:
        printed_lines = 0
        while printed_lines < 100:
            out_line = stream.readline().decode()
            if out_line.strip() != "": # There is something to print for this line
                if (pid in pid_2_cmd) and (" ".join(pid_2_cmd[pid]) in cmdStr_2_argDict) and (printed_lines == 0):
                     # find out what cmd is printing and add header
                    cmd_prefix = pretty_str(cmdStr_2_argDict[" ".join(pid_2_cmd[pid])])
                    cmd_prefix += ": "
                    print("\r"+cmd_prefix, flush=True)
                print("\r  "+out_line, flush=True)
                # find updates in prints
                if "Updates" in out_line:
                    update_strs =  re.findall("Updates.*FPS.*\n", out_line)
                    if update_strs:
                        info = update_strs[0].rstrip()
                        pid_2_printInfo[pid] = info
                # find potential errors in prints
                if "Error" in out_line and pid in pid_2_printInfo:
                    pid_2_printInfo[pid] += " Likely Error: " + out_line
                procs_printed = True
            printed_lines += 1
    
    # print updates if cmds have ended or async_procs printed or its been a while
    if ((len(cmds_terminated)+len(cmds_failed)) > num_done_at_last_print) or procs_printed or (time()-last_print_time > 60):
        
        def print_cmd_updates(cmd):
            cmd_str = " ".join(cmd)
            info = ""
            if cmd_str in cmdStr_2_pid:
                pid = cmdStr_2_pid[cmd_str]
                if pid in pid_2_devID:
                    devID_str = str(pid_2_devID[pid])
                    info += "DevID: " + devID_str
                    info += "       "[:-len(devID_str)] # ensures constant length
                if pid in pid_2_printInfo:
                    info += pid_2_printInfo[pid]
            up_time = None
            if cmd_str in cmdStr_2_startTime:
                if cmd_str in cmdStrs_2_stopTime:
                    up_time = "{:.2f}".format((cmdStrs_2_stopTime[cmd_str]-cmdStr_2_startTime[cmd_str])/3600)
                else:
                    up_time = "{:.2f}".format((time()-cmdStr_2_startTime[cmd_str])/3600)
            print("\r"+pretty_str(cmdStr_2_argDict[cmd_str]), flush=True)
            if up_time is not None:
                print("\r      Uptime {} hours, ".format(up_time) + info, flush=True)
        
        frac_done = len(cmds_terminated+cmds_failed) / len(cmds_terminated+running_cmds+cmds_remaining+cmds_failed)
        print("\r\nRunning... percent done: {}".format(100. * frac_done), flush=True)
        print("\rRemaining cmds:")
        for cmd_rem in cmds_remaining:
            print_cmd_updates(cmd_rem)
        print("\rTerminated cmds:")
        for cmd_term in cmds_terminated:
            print_cmd_updates(cmd_term)
        print("\rFailed cmds:")
        for cmd_f in cmds_failed:
            print_cmd_updates(cmd_f)
        print("\rRunning cmds:")
        for cmd_run in running_cmds:
            print_cmd_updates(cmd_run)
        num_done_at_last_print = len(cmds_terminated)+len(cmds_failed)
    return num_done_at_last_print

def safe_exec_cmd_output_syncronous(cmd, name_of_caller):
    success = True
    output = None
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=False)
        output = output.decode()
    except Exception as e:
        print("Error in", name_of_caller+":", e)
        print("for cmd:", cmd)
        if hasattr(e, "output"):
            print(e.output)
            output = e.output
        success = False
    return success, output

def safe_exec_cmdstr_syncronous(cmd_str, name_of_caller, print_info=True):
    success = True
    try:
        if print_info:
            exit_code = subprocess.call(cmd_str, stderr=subprocess.STDOUT, shell=True)
        else:
            exit_code = subprocess.call(cmd_str, stderr=subprocess.STDOUT, stdout=subprocess.DEVNULL, shell=True)
        assert exit_code == 0, "exit_code was not 0: {}".format(exit_code)
    except Exception as e:
        if print_info:
            print("Error in", name_of_caller+":", e)
            print("for cmd:", cmd_str)
            if hasattr(e, "output"):
                print(e.output)
        success = False
    return success

def is_prioritized(cmd, args):
    for s in cmd:
        if args.prioritize in str(s):
            return True
    return False

def is_excluded(cmd, args):
    for s in cmd:
        if args.exclude in str(s):
            return True
    return False

def setup_files():
    file_dir = os.path.dirname(os.path.realpath(__file__))
    temp_dir = os.path.join(file_dir, "temp")
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
    active_cmds_dir = os.path.join(temp_dir, "active_cmds")
    if not os.path.exists(active_cmds_dir):
        os.mkdir(active_cmds_dir)
    active_cmds_here_file = os.path.join(active_cmds_dir, "active_cmds")
    # make sure active_cmds_here_file exists and is empty
    with open(active_cmds_here_file, "w") as f:
        pass
    return file_dir, temp_dir, active_cmds_here_file

def import_experiments(experiment_files):
    global CPU_EXPERIMENT_SETS
    global GPU_EXPERIMENT_SETS
    for filename in experiment_files:
        imported_file = __import__("experiment_sets."+filename, fromlist=[''])
        CPU_EXPERIMENT_SETS.extend(imported_file.CPU_EXPERIMENT_SETS)
        GPU_EXPERIMENT_SETS.extend(imported_file.GPU_EXPERIMENT_SETS)

def bool_arg(bool_str):
    assert bool_str.lower() in ["true", "false"], bool_str
    return bool_str.lower() == "true"

def build_dockerfile(docker_file, docker_dir):
    dockerfile_name = os.path.basename(docker_file)
    print("Updating Docker:", dockerfile_name, flush=True)
    success = safe_exec_cmdstr_syncronous("cd "+docker_dir+" && bash build.sh " + dockerfile_name, "build_dockers()", print_info=False)
    if not success:
        print("Warning: Docker file could not be built:", docker_file, flush=True)

def build_dockers(args):
    if args.docker_dir is None or args.docker_dir == ["None"]:
        return
    docker_files = glob.glob(os.path.join(os.path.expanduser(args.docker_dir), "Dockerfile_*"))
    print("\nUpdating Dockers...", flush=True)
    with Pool(processes=len(docker_files)) as p_pool:
        p_pool.starmap(build_dockerfile, [(f, args.docker_dir) for f in docker_files])
    print("", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('EXPERIMENT_FILES', help='the filenames of experiment_sets to run (in ./experiment_sets/) with no commas and no extension', nargs='+')
    parser.add_argument('--docker_dir', help='specify a directories with docker imgs to build. Names must start with Dockerfile_', nargs='+', default=THIS_DIR)
    parser.add_argument('--cpu_cmd', help='a cmd for running with cpu argument, NOT detached', type=str, default=os.path.join(THIS_DIR,'run_cpu.sh')+' mujoco')
    parser.add_argument('--gpu_cmd', help='a cmd for running with gpu argument, NOT detached', type=str, default=os.path.join(THIS_DIR,'run_gpu.sh')+' mujoco')
    parser.add_argument('--default_mujoco_version', help='assumed mujoco_version if not specified', type=str, default='150')
    parser.add_argument('--docker_util_cmd', type=str, help='a cmd for running docker that does not detach and has psutil package', default=os.path.join(THIS_DIR,'run_any_dev.sh'+' mujoco150'))
    parser.add_argument('--python_cmd', type=str, default='python ' + os.path.join(THIS_DIR,'main.py'))
    parser.add_argument('--viz_python_cmd', type=str, default='python ' + os.path.join(THIS_DIR,'visualize_runs.py'))
    parser.add_argument('--cpu_free', type=str, default="0-0", help='None or the range of cpus free, e.g. 0-10, inclusive') # required if cpu experiments
    parser.add_argument('--gpu_free', type=str, default="0-0", help='None or the range of gpus free, e.g. 0-3, inclusive')  # required if gpu experiments
    parser.add_argument('--experiments_per_cpu', type=int, default=1, help='the number of cpu only experiments per cpu, e.g. 3')
    parser.add_argument('--experiments_per_gpu', type=int, default=5, help='the number of gpu experiments per gpu, e.g. 3')
    parser.add_argument('--force_gpu', help='move all experiments to gpu', action='store_true')
    parser.add_argument('--cpu_measure_sec', type=int, default=10, help='the number of seconds over which to estimate cpu usage')
    parser.add_argument('--log_base_dir_here', type=str, help='the base log dir on machine', default=os.path.join(THIS_DIR,'data'))
    parser.add_argument('--docker_log_base_dir', type=str, help='the same base log dir, but from within docker', default=os.path.join(THIS_DIR,'data'))
    parser.add_argument('--plot_at_end', help='plot learning curves at end', type=bool_arg, default=False)
    parser.add_argument('--plot', help='only plot learning curves and nothing else', action='store_true')
    parser.add_argument('--prioritize', help='any experiment with this str in any part of the command will be run first (useful for slow commands)', type=str, default="_NoRec")
    parser.add_argument('--exclude', help='any experiment with this str will not be run', type=str, default=None)
    parser.add_argument('--shuffle', help='shuffle order of the experiments', action='store_true')
    parser.add_argument('--async_plot', help='plot asynchronously... faster but may make errors hard to read', type=bool_arg, default=True)
    parser.add_argument('--plot_dir', type=str, default=None, help='relative dir in log_base_dir_here to plot. e.g. HMT1 If None, all will be plotted. Supports glob regex. * will print all. None will use dirs in experiments.')
    parser.add_argument('--plot_env', type=str, default="*", help='env name to plot. e.g. Hop-v0. If None, all envs will be plotted. Supports glob regex')
    parser.add_argument('--plot_exclude', help='model names to exclude from all plots separated by commas with no spaces', type=str, default='')
    parser.add_argument('--plot_include', help='model names to include in plots separated by commas with no spaces', type=str, default='')
    parser.add_argument('--low_y_lim', help='starting value for y axis as a string, e.g. \'-5\'', type=str, default=None)
    parser.add_argument('--up_y_lim', help='ending value for y axis as a string, e.g. \'-5\'', type=str, default=None)
    parser.add_argument('--ignore_not_done', help='run all experiments, ignoring those for which it is not clear if they have completed', action='store_true')
    parser.add_argument('--expand', help='use more devices as they become free', type=bool_arg, default=False)
    parser.add_argument('--devnull', help='pipe experiment output to devnull. useful for debugging runner output.', action='store_true')
    parser.add_argument('--indiv', help='plot individual runs instead of confidence interval', action='store_true')
    parser.add_argument('--all_lr', help='plot all lrs, not best over lrs', action='store_true')
    parser.add_argument('--lr_keys', type=str, help='The key for the tuned learning rate. Plotting only supports search over one lr.', default='lr_policy lr_vae')
    parser.add_argument('--plot_tags', nargs='+', default=[None], help='tag to plot. If a tag is None, default from plt script')
    parser.add_argument('--final_r', help='plot order by final return instead of average return over training', action='store_true')
    parser.add_argument('--no_legend', help='no legend for plotting', action='store_true')
    parser.add_argument('--log_scale', help='plot in log scale', type=bool_arg, default=False)
    parser.add_argument('--smooth', help='smooth sz for plots', type=str, default=None)
    parser.add_argument('--step', help='max step for plotting (k)', type=str, default=None)
    parser.add_argument('--shift_warmups', help='Shift plots with pre-compute specified in the name.', action='store_true')
    parser.add_argument('--metric_tag', help='metric tag for plotting', type=str, default=None)
    args = parser.parse_args()

    # Import Experiments
    import_experiments(args.EXPERIMENT_FILES)

    # only plot experiments
    if args.plot:
        free_cpus, _ = get_free_devices(args)
        exps_names={exp["dir_name"] for exp in CPU_EXPERIMENT_SETS+GPU_EXPERIMENT_SETS}
        plot(args, free_cpus, exps_names=exps_names, async_procs=[] if args.async_plot else None)
        exit()

    print("\n")
    # Set up files
    file_dir, temp_dir, active_cmds_here_file = setup_files()

    #### Signal Handlers ####
    # Set up signal handler and exit handler to close asnyc procs in case something goes wrong
    async_procs, corruptable_async_procs = [], []
    cpuID_2_procs, gpuID_2_procs = {}, {} # Storage for later, will be good to print at exit
    def close_procs_handler(sig, frame):
        for proc in async_procs:
            proc.terminate()
        while corruptable_async_procs:
            corruptable_async_procs.pop().wait()
        print("\nClosed procs given signal.")
        exit(0) # Should call handler below, but close procs above just in case
    signal.signal(signal.SIGINT, close_procs_handler)
    signal.signal(signal.SIGTERM, close_procs_handler)
    def at_exit_handler():
        for proc in async_procs:
            proc.terminate()
        while corruptable_async_procs:
            corruptable_async_procs.pop().wait()
        print("\nClosed procs at exit.")
        print("pids killed on each free cpu:", [[p.pid for p in procs] for procs in list(cpuID_2_procs.values())])
        print("pids killed on each free gpu:", [[p.pid for p in procs] for procs in list(gpuID_2_procs.values())])
        with open(active_cmds_here_file, "w") as f:
            pass
    atexit.register(at_exit_handler)

    

    #### Setup ####
    build_dockers(args)     # Build Dockers (make sure up to date)
    # get commands to run on each device
    cpu_cmds, gpu_cmds, cmdStr_2_argDict, cmdStr_2_mujoco = get_cpu_and_gpu_cmds(args)
    # Remove previously successful cmds and failed experiments data
    cpu_cmds, gpu_cmds = filter_and_clean(cpu_cmds, gpu_cmds, cmdStr_2_argDict, args)
    # Get free devices
    free_cpus, free_gpus = get_free_devices(args)
    # Make sure there are some free devices
    if cpu_cmds:
        assert free_cpus or args.expand, "Must have free CPU or expand=True if running on CPU"
    if gpu_cmds:
        assert free_gpus or args.expand, "Must have free GPU or expand=True if running on GPU"
    # reverse order so when we pop we go in order defined at top of file
    cpu_cmds.reverse()
    gpu_cmds.reverse()
    # if you want to shuffle order...
    if args.shuffle:
        shuffle(cpu_cmds)
        shuffle(gpu_cmds)
    # prioritize certain runs
    if args.prioritize: # keep in mind that cmds are popped from the end!
        cpu_to_prioritize = [cmd for cmd in cpu_cmds if is_prioritized(cmd, args)]
        gpu_to_prioritize = [cmd for cmd in gpu_cmds if is_prioritized(cmd, args)]
        cpu_cmds = [cmd for cmd in cpu_cmds if not is_prioritized(cmd, args)] + cpu_to_prioritize
        gpu_cmds = [cmd for cmd in gpu_cmds if not is_prioritized(cmd, args)] + gpu_to_prioritize
        print("Number of commands prioritized:", len(cpu_to_prioritize)+len(gpu_to_prioritize))
    # exclude certain runs
    if args.exclude: # exclude certain runs
        cpu_cmds = [cmd for cmd in cpu_cmds if not is_excluded(cmd, args)]
        gpu_cmds = [cmd for cmd in gpu_cmds if not is_excluded(cmd, args)]

    




    #### Main Scheduling Loop ####
    start_time = time()
    # ask if everything looks right..
    print("\nCmds to run here:")
    all_cmds = cpu_cmds+gpu_cmds
    pretty_print_cmdstrs([" ".join(cmd) for cmd in all_cmds], cmdStr_2_argDict)
    print("\nnumber to run here (if they fit): {}\n".format(len(all_cmds)))
    if len(all_cmds) == 0:
        print("No new cmds")
        exit()
    print("\ncontinue?", flush=True)
    ans = input()
    if not (ans.lower() == "y" or ans.lower() == "yes"):
        exit()
    print("")
    # While there are commands left to run, assign them to a device if possible or wait
    cpuID_2_procs.update({dev_id: [] for dev_id in free_cpus})
    gpuID_2_procs.update({dev_id: [] for dev_id in free_gpus})
    pid_2_cmd, pid_2_devID, cmdStr_2_startTime, cmdStrs_2_stopTime, pid_2_printInfo, cmds_terminated, cmds_failed, running_cmds = {}, {}, {}, {}, {}, [], [], []
    num_done_at_last_print = 0
    last_print_time = time()
    while cpu_cmds or gpu_cmds or running_cmds:
        # print updates if something has changed or there is output from async_procs
        num_done_at_last_print = print_loop_updates(cmds_terminated, cmds_failed, running_cmds, cpu_cmds+gpu_cmds, last_print_time, pid_2_devID,
                                                    num_done_at_last_print, cmdStr_2_argDict, cmdStr_2_startTime, cmdStrs_2_stopTime, async_procs, pid_2_cmd, pid_2_printInfo)
        # Remove procs that have terminated from device assignment. CPU then GPU.
        more_terminated, more_failed = remove_term_procs(cpuID_2_procs, pid_2_cmd, cmdStrs_2_stopTime, cmdStr_2_argDict, active_cmds_here_file, corruptable_async_procs, args)
        cmds_terminated += more_terminated
        cmds_failed += more_failed
        more_terminated, more_failed = remove_term_procs(gpuID_2_procs, pid_2_cmd, cmdStrs_2_stopTime, cmdStr_2_argDict, active_cmds_here_file, corruptable_async_procs, args)
        cmds_terminated += more_terminated
        cmds_failed += more_failed

        # expand free devices if --expand and cmds left
        if args.expand and (cpu_cmds or gpu_cmds):
            free_cpus, free_gpus = get_free_devices(args, force_detect=True)
            expand_dev_2_procs(free_cpus, cpuID_2_procs)
            expand_dev_2_procs(free_gpus, gpuID_2_procs)
        # fill up free slots on devices with cmds that still need to be run. CPU then GPU.
        cpu_cmds, started_a_cpu_cmd = fill_devices(cpu_cmds, args.experiments_per_cpu, cpuID_2_procs, args.cpu_cmd,
            pid_2_cmd, pid_2_devID, cmdStr_2_argDict, cmdStr_2_startTime, cmdStr_2_mujoco, active_cmds_here_file, async_procs, args)
        gpu_cmds, started_a_gpu_cmd = fill_devices(gpu_cmds, args.experiments_per_gpu, gpuID_2_procs, args.gpu_cmd,
            pid_2_cmd, pid_2_devID, cmdStr_2_argDict, cmdStr_2_startTime, cmdStr_2_mujoco, active_cmds_here_file, async_procs, args)
        # update running cmds list for printing
        running_cmds = []
        for procs in list(cpuID_2_procs.values())+list(gpuID_2_procs.values()):
            running_cmds.extend([pid_2_cmd[p.pid] for p in procs])
        # sleep
        sleep(15)
    ####
    print_loop_updates(cmds_terminated, cmds_failed, running_cmds, cpu_cmds+gpu_cmds, last_print_time, pid_2_devID,
                            num_done_at_last_print, cmdStr_2_argDict, cmdStr_2_startTime, cmdStrs_2_stopTime, async_procs, pid_2_cmd, pid_2_printInfo)
    print("\nDone with exps.", flush=True)




    #### PLOT at end ####
    # Plot all learning curves.
    if args.plot_at_end:
        free_cpus.add(0)
        exps_names={exp["dir_name"] for exp in CPU_EXPERIMENT_SETS+GPU_EXPERIMENT_SETS}
        plot(args, free_cpus, exps_names=exps_names, async_procs=async_procs if args.async_plot else None)

    #### WAIT #####
    while async_procs:
        async_procs.pop().wait(0)

    print("\r\n\nAll Done!")
    print("\rTook", (time()-start_time)/3600, "hours", flush=True)




if __name__ == "__main__":
    main()








