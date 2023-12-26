# Hypernetworks in Meta-RL

This repository contains code for the papers *Hypernetworks in Meta-Reinforcement Learning* (Beck et al., 2022), published at CoRL, and *Recurrent Hypernetworks are Surprisingly Strong in Meta-RL* (Beck et al., 2023), published at NeurIPS.

```
@inproceedings{beck2022hyper,
  author    = {Jacob Beck and
               Matthew Jackson and
               Risto Vuorio and
               Shimon Whiteson},
  title     = {Hypernetworks in Meta-Reinforcement Learning},
  booktitle = {Conference on Robot Learning},
  year      = {2022},
}
@inproceedings{beck2023recurrent,
  author     =  {Jacob Beck and Risto Vuorio and Zheng Xiong and Shimon Whiteson},
  title      =  {Recurrent Hypernetworks are Surprisingly Strong in Meta-RL},
  booktitle  =  {Thirty-seventh Conference on Neural Information Processing Systems},
  year       =  {2023},
  url        =  {https://openreview.net/forum?id=pefAAzu8an}
}
```

This code is based on *VariBAD: A very good method for Bayes-Adaptive Deep RL via Meta-Learning* (Zintgraf et al., 2020). If you use this code, please additionally cite this paper:

```
@inproceedings{zintgraf2020varibad,
  author    =  {Zintgraf, Luisa and Shiarlis, Kyriacos and Igl, Maximilian and Schulze, Sebastian and Gal, Yarin and Hofmann, Katja and Whiteson, Shimon},
  title     =  {VariBAD: A Very Good Method for Bayes-Adaptive Deep RL via Meta-Learning},
  booktitle =  {International Conference on Learning Representation (ICLR)},
  year      =  {2020}}
```

Finally, the T-Maze environments, Minecraft environments, aggregators in `aggregator.py`, and SNR visualization are reproduced from *AMRL: Aggregated Memory For Reinforcement Learning* (Beck et al., 2020). If you use any of those modules, please cite this paper:

```
@inproceedings{beck2020AMRL,
  author     =  {Jacob Beck and Kamil Ciosek and Sam Devlin and Sebastian Tschiatschek and Cheng Zhang and Katja Hofmann},
  title      =  {AMRL: Aggregated Memory For Reinforcement Learning},
  booktitle  =  {International Conference on Learning Representations},
  year       =  {2020},
  url        =  {https://openreview.net/forum?id=Bkl7bREtDr}
}
```

### Usage

The experiments can be found in `experiment_sets/`. The models themselves are defined in `models.py`. Main results on initialization methods (Beck et al., 2022) can be found in `init_main_results.py`. Main results on supervision (Beck et al., 2023) can be found in `main_results.py`. Analysis and the remaining environments can be found in `analysis.py` and `all_envs.py`, respectively.

`run_experiments.py` can be used to build dockers, launch experiments, and start new experiments when there is sufficient space.

*Example usage:*
```
python3 run_experiments.py main_results --shuffle --gpu_free 0-7 --experiments_per_gpu 3 |& tee log.txt
```

The script, `run_experiments.py`, automatically runs commands using the docker files, e.g., executing `run_cpu.sh mujoco150 0 python ~/MetaMem/main.py --env-type gridworld_varibad`, to run gridworld on CPU 0. Within a docker, this command could be run with `python main.py --env-type gridworld_varibad`. 

The main training loop itself can be found in `metalearner.py`, the hypernetwork is in `policy.py`, and added supervision for task inference is in `ppo.py`.

After training, `visualize_runs.py` can be used for plotting. To automatically plot all results for a set of experiments, you can also use the `run_experiments.py` script.

*Example usage:*
```
python3 run_experiments.py main_results --plot
```

### Comments

- The *env-type* argument refers to a config in `config/`, and is a list of default arguments common to an environment, but these can be overridden in the experiment set.
- Different environments require one of three different dockers, specifying different MuJoCo versions, as documented in the respective experiments sets.
The dockerfiles can be built automatically with `run_experiments.py`, or manually with, e.g., `bash build.sh Dockerfile_mj150`.
- `requirements.txt` is legacy from VariBAD, and likely out of date.
