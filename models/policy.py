"""
Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
"""
import gym
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Policy(nn.Module):
    def __init__(self,
                 args,
                 # input
                 pass_state_to_policy,
                 pass_latent_to_policy,
                 pass_belief_to_policy,
                 pass_task_to_policy,
                 dim_state,
                 dim_latent,
                 dim_belief,
                 dim_task,
                 # hidden
                 hidden_layers,
                 activation_function,  # tanh, relu, leaky-relu
                 policy_initialisation,  # orthogonal / normc
                 # output
                 action_space,
                 init_std,
                 norm_actions_of_policy,
                 action_low,
                 action_high,
                 ):
        """
        The policy can get any of these as input:
        - state (given by environment)
        - task (in the (belief) oracle setting)
        - latent variable (from VAE)
        """
        super(Policy, self).__init__()

        self.args = args

        # set up activation
        if activation_function == 'tanh':
            self.activation_function = nn.Tanh()
        elif activation_function == 'relu':
            self.activation_function = nn.ReLU()
        elif activation_function == 'leaky-relu':
            self.activation_function = nn.LeakyReLU()
        else:
            raise ValueError

        # set up init_
        gain = nn.init.calculate_gain(activation_function)
        self.HFI = False
        def get_init(init_str):
            if init_str == 'normc':
                init_ = lambda m: init(m, init_normc_, lambda x: nn.init.constant_(x, 0), gain)
            elif init_str == 'orthogonal':
                init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain)
            else:
                assert init_str == 'kaiming', init_str
                init_ = lambda m: init(m, lambda tensor, gain: nn.init.kaiming_uniform_(tensor, mode='fan_in', nonlinearity=activation_function), 
                                      lambda x: nn.init.constant_(x, 0), 
                                      gain=1) # This is additional gain now, since kaiming_uniform_ calculates gain itself
            return init_
        if policy_initialisation.lower() != "hfi":
            init_ = get_init(policy_initialisation)
        else: # Hyperfan-In from (Chang et al., 2020): https://gist.github.com/crazyoscarchang/c9a11b67c420202da1f26e0d20786750
            # asserts and checks
            assert policy_initialisation.lower() == "hfi"
            assert self.args.use_hypernet
            assert not self.args.use_film
            if args.hypernet_input == "task_embed":
                if not self.args.norm_task_for_policy:
                    print("Warning, HFI really should normalize task input.")
            else:
                if args.rlloss_through_encoder:
                    print("Warning, HFI expects normalized latent, but you cannot do that if passing rlloss_through_encoder")
                elif not self.args.norm_latent_for_policy:
                    print("Warning, HFI init will normalize latent input despite the argument setting")
                    self.args.norm_latent_for_policy = True
            self.HFI = True
            # define init for all but last layer
            init_ = lambda m: init(m, lambda tensor, gain: nn.init.kaiming_uniform_(tensor, mode='fan_in', nonlinearity=activation_function), 
                                      lambda x: nn.init.constant_(x, 0), 
                                      gain=1)
        

        self.pass_state_to_policy = pass_state_to_policy
        self.pass_latent_to_policy = pass_latent_to_policy
        self.pass_task_to_policy = pass_task_to_policy
        self.pass_belief_to_policy = pass_belief_to_policy

        self.use_hyper = self.args.use_hypernet
        self.use_film = self.args.use_film
        self.multinet = self.args.multinet

        if self.multinet or (args.pass_task_to_policy and args.pass_task_as_onehot_id) or (args.hyper_onehot_chance is not None):
            # need this env and info within to convert to one_hot
            self.dummy_env, _ = utl.make_env(args)
            assert hasattr(self.dummy_env, "task_to_id"), "args.env_name must have task_to_id() function"
            assert hasattr(self.dummy_env, "num_tasks"), "args.env_name must have num_tasks attribute"
            dim_task = self.dummy_env.num_tasks

        if self.multinet: # multinet (separate net for each task) must have one-hot representation of input task
            assert args.pass_task_to_policy
            assert args.policy_task_embedding_dim is None
            self.args.norm_task_for_policy = False
            self.use_hyper = self.args.use_hypernet = args.use_hypernet = False
            self.use_film = self.args.use_film = args.use_film = False
            self.pass_task_as_onehot_id = self.args.pass_task_as_onehot_id = args.pass_task_as_onehot_id = False
            num_nets = dim_task
            self.net_2_actor_layers = [nn.ModuleList() for _ in range(num_nets)]
            self.net_2_critic_layers = [nn.ModuleList() for _ in range(num_nets)]
            self.net_2_st_layer = [None for _ in range(num_nets)]
            self.net_2_dist_layer = [None for _ in range(num_nets)]
            self.net_2_critic_head_layer = [None for _ in range(num_nets)]
            self.num_nets = num_nets

        # set normalisation parameters for the inputs
        # (will be updated from outside using the RL batches)
        self.norm_state = self.args.norm_state_for_policy and (dim_state is not None)
        if self.pass_state_to_policy and self.norm_state:
            self.state_rms = utl.RunningMeanStd(shape=(dim_state))
        self.norm_latent = self.args.norm_latent_for_policy and (dim_latent is not None)
        if self.pass_latent_to_policy and self.norm_latent:
            self.latent_rms = utl.RunningMeanStd(shape=(dim_latent))
        self.norm_belief = self.args.norm_belief_for_policy and (dim_task is not None)
        if self.pass_belief_to_policy and self.norm_belief:
            self.belief_rms = utl.RunningMeanStd(shape=(dim_belief))
        self.norm_task = self.args.norm_task_for_policy and (dim_belief is not None)
        if self.pass_task_to_policy and self.norm_task:
            self.task_rms = utl.RunningMeanStd(shape=(dim_task))

        curr_input_dim = dim_state * int(self.pass_state_to_policy) + \
                         dim_latent * int(self.pass_latent_to_policy) + \
                         dim_belief * int(self.pass_belief_to_policy) + \
                         dim_task * int(self.pass_task_to_policy)
        # initialise encoders for separate inputs
        self.use_state_encoder = self.args.policy_state_embedding_dim is not None
        if self.pass_state_to_policy and self.use_state_encoder:
            if self.multinet:
                for j in range(num_nets): # make a state_encoder for each task network
                    self.net_2_st_layer[j] = utl.FeatureExtractor(dim_state, self.args.policy_state_embedding_dim, self.activation_function)
            else:
                self.state_encoder = utl.FeatureExtractor(dim_state, self.args.policy_state_embedding_dim, self.activation_function)
            curr_input_dim = curr_input_dim - dim_state + self.args.policy_state_embedding_dim
        self.use_latent_encoder = self.args.policy_latent_embedding_dim is not None
        if self.pass_latent_to_policy and self.use_latent_encoder:
            self.latent_encoder = utl.FeatureExtractor(dim_latent, self.args.policy_latent_embedding_dim, self.activation_function)
            curr_input_dim = curr_input_dim - dim_latent + self.args.policy_latent_embedding_dim
        self.use_belief_encoder = self.args.policy_belief_embedding_dim is not None
        if self.pass_belief_to_policy and self.use_belief_encoder:
            self.belief_encoder = utl.FeatureExtractor(dim_belief, self.args.policy_belief_embedding_dim, self.activation_function)
            curr_input_dim = curr_input_dim - dim_belief + self.args.policy_belief_embedding_dim
        self.use_task_encoder = self.args.policy_task_embedding_dim is not None
        if self.pass_task_to_policy and self.use_task_encoder:
            self.task_encoder = utl.FeatureExtractor(dim_task, self.args.policy_task_embedding_dim, self.activation_function)
            curr_input_dim = curr_input_dim - dim_task + self.args.policy_task_embedding_dim
        if self.multinet:
            curr_input_dim -= dim_task # actually not conditioned on through activation
        if self.use_hyper: # if using hypernet, we will instead pass task embedding to hypernet
            if args.hypernet_input == "task_embed":
                assert self.pass_task_to_policy, "The Hypernet needs the task passed to the policy in order to condition on it"
                assert dim_task != 0, "Dimension of task cannot be 0. You may want to check that env has env.task_dim set properly."
                hyper_in_dim = dim_task if self.args.policy_task_embedding_dim is None else self.args.policy_task_embedding_dim
            else:
                assert args.hypernet_input == "latent", args.hypernet_input
                assert self.pass_latent_to_policy, "The Hypernet needs the latent passed to the policy in order to condition on it"
                hyper_in_dim = dim_latent if self.args.policy_latent_embedding_dim is None else self.args.policy_latent_embedding_dim
            if args.hyper_onehot_chance is not None:
                assert args.hypernet_input == "latent", args.hypernet_input
                assert self.pass_task_to_policy
                assert self.pass_latent_to_policy
                assert self.args.policy_task_embedding_dim is None
                curr_input_dim -= dim_task
            curr_input_dim -= hyper_in_dim

        self.dim_state, self.dim_latent, self.dim_belief, self.dim_task = dim_state, dim_latent, dim_belief, dim_task
        self.state_in_dim = self.args.policy_state_embedding_dim if self.use_state_encoder else dim_state
        self.latent_in_dim = self.args.policy_latent_embedding_dim if self.use_latent_encoder else dim_latent
        self.belief_in_dim = self.args.policy_belief_embedding_dim if self.use_belief_encoder else dim_belief
        self.task_in_dim = self.args.policy_task_embedding_dim if self.use_task_encoder else dim_task

        if args.task_chance is not None: # sometimes passing task, sometimes latent, but not both
            assert self.pass_task_to_policy
            assert self.pass_latent_to_policy
            assert self.args.policy_task_embedding_dim is not None, self.args.policy_task_embedding_dim 
            assert self.args.policy_task_embedding_dim == self.args.policy_latent_embedding_dim, (self.args.policy_task_embedding_dim, self.args.policy_latent_embedding_dim)
            curr_input_dim -= self.args.policy_task_embedding_dim # doesn't matter which one here, because they must be the same size

        # initialize hypernetwork (except the heads of the hypernetwork, which are defined by the policy network)
        if self.use_hyper:
            if self.args.init_hyper_for_policy and self.args.HN_init is not None:
                # HN init may be different from base net init
                HN_init_ = get_init(self.args.HN_init)
            else:
                HN_init_ = init_
            self.hyper_layers = nn.ModuleList()
            cur_dim = hyper_in_dim
            for next_sz in [int(sz) for sz in self.args.hypernet_layers]: 
                fc = HN_init_(nn.Linear(cur_dim, next_sz))
                self.hyper_layers.append(fc)
                cur_dim = next_sz
            final_hyper_hidden_sz = cur_dim
        if self.use_film:
            assert self.use_hyper, "If using FiLM, you must have use_hypernet=True, since we implement FiLM as a simplified Hypernet"

        # initialise actor and critic
        hidden_layers = [int(h) for h in hidden_layers]
        self.hidden_layers = hidden_layers
        self.actor_layers = nn.ModuleList() # actor layers, or None if this layer is produced by a hypernetwork
        self.critic_layers = nn.ModuleList()
        self.hyper_final_to_actor_weight  = nn.ModuleList() # defines transformations from final_hyper_hidden_sz to each required weight, or None
        self.hyper_final_to_actor_bias    = nn.ModuleList() # defines transformations from final_hyper_hidden_sz to each required bias, or None
        self.hyper_final_to_critic_weight = nn.ModuleList()
        self.hyper_final_to_critic_bias   = nn.ModuleList()
        if self.use_hyper:
            for is_hyper in args.policy_layers_hyper:
                assert is_hyper in ['0','1'], is_hyper
            policy_layer_is_hyper = [True if is_hyper == '1' else False for is_hyper in args.policy_layers_hyper]
            assert len(policy_layer_is_hyper) == (len(hidden_layers) + 1), "--policy_layers_hyper must be specified for every layer and the policy head"
            assert sum(policy_layer_is_hyper) >= 1, "Must have some layers hyper if using hypernet. Adjust --policy_layers_hyper."
            if args.init_hyper_for_policy:
                    assert args.init_adjust_weights or args.init_adjust_bias, "Must adjust weights or bias in order to init hypernet to match original policy init"
        for i in range(len(hidden_layers)):
            layer_is_hyper = self.use_hyper and policy_layer_is_hyper[i]
            if layer_is_hyper:
                # Film changes
                if self.use_film: # hypernet only point-wise scales activations, but still produces full bias
                    assert args.hypernet_head_bias, "If using FiLM, you need a bias in the hypernet"
                    self.actor_layers.append(init_(nn.Linear(curr_input_dim, hidden_layers[i], bias=False))) # generate base weights
                    self.critic_layers.append(init_(nn.Linear(curr_input_dim, hidden_layers[i], bias=False))) # generate base weights
                    # define hypernet head for this layer
                    weight_sz = hidden_layers[i] # number of scaling factors to produce for this layer
                    bias_sz = hidden_layers[i] # size of bias required for this layer
                else:
                    self.actor_layers.append(None)
                    self.critic_layers.append(None)
                    # define hypernet head for this layer
                    weight_sz = curr_input_dim * hidden_layers[i] # size of weights required for this layer
                    bias_sz = hidden_layers[i] # size of bias required for this layer
                # select init method for HyperNet
                if self.HFI:
                    bias_layer_init   = HFI_bias_layer_init(final_hyper_hidden_sz, gain=gain)
                    weight_layer_init = HFI_weight_layer_init(final_hyper_hidden_sz, curr_input_dim, gain=gain)
                elif args.init_hyper_for_policy:
                    bias_layer_init   = hyper_bias_layer_init()
                    weight_layer_init = hyper_weight_layer_init(activation_function, policy_initialisation, final_hyper_hidden_sz, curr_input_dim, 
                                                                hidden_layers[i], args.adjust_init_b_gain, adjust_weights=args.init_adjust_weights, adjust_bias=args.init_adjust_bias,
                                                                use_film=self.use_film)
                else:
                    bias_layer_init, weight_layer_init = init_, init_
                # Create Hyper Head layers
                self.hyper_final_to_actor_weight.append(weight_layer_init(nn.Linear(final_hyper_hidden_sz, weight_sz, bias=args.hypernet_head_bias)))
                self.hyper_final_to_actor_bias.append(bias_layer_init(nn.Linear(final_hyper_hidden_sz, bias_sz, bias=args.hypernet_head_bias)))
                self.hyper_final_to_critic_weight.append(weight_layer_init(nn.Linear(final_hyper_hidden_sz, weight_sz, bias=args.hypernet_head_bias)))
                self.hyper_final_to_critic_bias.append(bias_layer_init(nn.Linear(final_hyper_hidden_sz, bias_sz, bias=args.hypernet_head_bias)))
            else:
                if self.multinet: # make one layer for each task network
                    for j in range(num_nets):
                        fc = init_(nn.Linear(curr_input_dim, hidden_layers[i]))
                        self.net_2_actor_layers[j].append(fc)
                        fc = init_(nn.Linear(curr_input_dim, hidden_layers[i]))
                        self.net_2_critic_layers[j].append(fc)
                else:
                    fc = init_(nn.Linear(curr_input_dim, hidden_layers[i]))
                    self.actor_layers.append(fc)
                    fc = init_(nn.Linear(curr_input_dim, hidden_layers[i]))
                    self.critic_layers.append(fc)
                # Not hypernet head needed for this layer
                self.hyper_final_to_actor_weight.append(None)
                self.hyper_final_to_actor_bias.append(None)
                self.hyper_final_to_critic_weight.append(None)
                self.hyper_final_to_critic_bias.append(None)
            curr_input_dim = hidden_layers[i]
        # some asserts
        if not self.multinet:
            assert len(self.actor_layers) == len(self.hyper_final_to_actor_weight), (len(self.actor_layers),len(self.hyper_final_to_actor_weight))
        if not self.use_hyper:
            assert None not in self.actor_layers, self.actor_layers
            assert None not in self.critic_layers, self.critic_layers
        else:
            assert sum([1 if f is not None else 0 for f in self.hyper_final_to_actor_weight]) == sum(policy_layer_is_hyper[:-1])

        # initialize head of critic
        self.head_is_hyper = self.use_hyper and policy_layer_is_hyper[-1]
        if self.head_is_hyper:
            # Film changes
            if self.use_film:
                self.critic_linear = nn.Linear(hidden_layers[-1], 1, bias=False)
                weight_sz = bias_sz = 1
            else:
                self.critic_linear = None
                weight_sz = hidden_layers[-1]
                bias_sz = 1
             # select init method for HyperNet
            if self.HFI:
                bias_layer_init   = HFI_bias_layer_init(final_hyper_hidden_sz, gain=1) # linear so gain = 1
                weight_layer_init = HFI_weight_layer_init(final_hyper_hidden_sz, hidden_layers[-1], gain=1)
            elif args.init_hyper_for_policy:
                bias_layer_init   = hyper_bias_layer_init(weight_for_uniform=torch.zeros((hidden_layers[-1], 1))) # the bias was not initialized to 0 here in base critic. This tensor gives us the shape to recreate the base init.
                weight_layer_init = hyper_weight_layer_init('linear' if args.adjust_init_b_gain else activation_function, 'kaiming', final_hyper_hidden_sz, hidden_layers[-1], 1, # the base critic used default nn.Linear, which uses kaiming
                                                            args.adjust_init_b_gain, adjust_weights=args.init_adjust_weights, adjust_bias=args.init_adjust_bias,
                                                            use_film=self.use_film)
            else:
                bias_layer_init, weight_layer_init = init_, init_
            # Create Hyper Head layers
            self.hyper_final_to_critic_linear_weight = weight_layer_init(nn.Linear(final_hyper_hidden_sz, weight_sz, bias=args.hypernet_head_bias))
            self.hyper_final_to_critic_linear_bias   = bias_layer_init(nn.Linear(final_hyper_hidden_sz, bias_sz, bias=args.hypernet_head_bias))
        else:
            if self.multinet:
                for j in range(num_nets): # make a state_encoder for each task network
                    self.net_2_critic_head_layer[j] = nn.Linear(hidden_layers[-1], 1)
            else:
                self.critic_linear = nn.Linear(hidden_layers[-1], 1) # Uses Kaiming init by default
            self.hyper_final_to_critic_linear_weight = None
            self.hyper_final_to_critic_linear_bias   = None

        # initialize output distributions of the policy
        if self.multinet:
            for j in range(num_nets):
                dist, dist_gain, num_outputs, _ = self.get_dist(policy_initialisation, action_space, hidden_layers, init_std, action_low, action_high, norm_actions_of_policy)
                self.net_2_dist_layer[j] = dist
        else:
            self.dist, dist_gain, num_outputs, actor_head_init = self.get_dist(policy_initialisation, action_space, hidden_layers, init_std, action_low, action_high, norm_actions_of_policy)

        self.num_outputs = num_outputs
        if self.head_is_hyper:
            # Film changes
            if self.use_film:
                weight_sz = bias_sz = num_outputs
            else:                                      
                weight_sz = hidden_layers[-1]*num_outputs
                bias_sz = num_outputs
            # select init method for HyperNet
            if self.HFI:
                bias_layer_init   = HFI_bias_layer_init(final_hyper_hidden_sz, gain=dist_gain) # use same gain for policy head as in other methods
                weight_layer_init = HFI_weight_layer_init(final_hyper_hidden_sz, hidden_layers[-1], gain=dist_gain)
            elif args.init_hyper_for_policy:
                bias_layer_init   = hyper_bias_layer_init() 
                weight_layer_init = hyper_weight_layer_init('linear' if args.adjust_init_b_gain else activation_function, actor_head_init, final_hyper_hidden_sz, hidden_layers[-1], num_outputs, 
                                                            args.adjust_init_b_gain, override_gain=dist_gain, adjust_weights=args.init_adjust_weights, adjust_bias=args.init_adjust_bias,
                                                            use_film=self.use_film)
            else:
                    bias_layer_init, weight_layer_init = init_, init_  
            # Create Hyper Head layers
            self.hyper_final_to_actor_linear_weight = weight_layer_init(nn.Linear(final_hyper_hidden_sz, weight_sz, bias=args.hypernet_head_bias))
            self.hyper_final_to_actor_linear_bias   = bias_layer_init(nn.Linear(final_hyper_hidden_sz, bias_sz, bias=args.hypernet_head_bias))
        else:
            # actor linear weights were created in self.dist
            self.hyper_final_to_actor_linear_weight = None
            self.hyper_final_to_actor_linear_bias = None


        if self.multinet: # convert list of nets to module list
            self.net_2_actor_layers = nn.ModuleList(self.net_2_actor_layers)
            self.net_2_critic_layers = nn.ModuleList(self.net_2_critic_layers)
            self.net_2_st_layer = nn.ModuleList(self.net_2_st_layer)
            self.net_2_dist_layer = nn.ModuleList(self.net_2_dist_layer)
            self.net_2_critic_head_layer = nn.ModuleList(self.net_2_critic_head_layer)


    def get_bn_param_dim(self):
        return self.task_bn_params(torch.zeros((1, self.dim_task,)).to(device)).shape[-1] if self.args.use_hypernet else None

    def task_bn_params(self, task):
        state_shape = task.shape[:-1] + (self.dim_state,)
        belief_shape = task.shape[:-1] + (self.dim_belief,)
        latent_shape = task.shape[:-1] + (self.dim_latent,)
        action_shape = task.shape[:-1] + (self.num_outputs,) # This shape may not be correct always, but should be ignore anyway since return_only_bn_params=True in call below
        base_params_task = self.evaluate_actions(state=torch.zeros(state_shape).to(device), latent=torch.zeros(latent_shape).to(device),
                                       belief=torch.zeros(belief_shape).to(device), task=task.to(device),
                                       action=torch.zeros(action_shape).to(device), return_action_mean=True,
                                       return_base_params=True, return_only_bn_params=True,
                                       return_info=False, training=True, update_num=0, force_task_input=True, # force task
                                       )
        return torch.cat(base_params_task,-1)


    ## Seems like this is never used? And self.actor does not exist? And now this would break with hypernetwork. ##
    # def get_actor_params(self):
    #     return [*self.actor.parameters(), *self.dist.parameters()]
    # def get_critic_params(self):
    #     return [*self.critic.parameters(), *self.critic_linear.parameters()]

    def should_overwrite_task(self, training, update_num):
        # override task encoding overwrites latent encoding if self.args.hyper_onehot_chance or self.args.task_chance, with some probability
        
        chance_argument = self.args.task_chance if self.args.task_chance is not None else self.args.hyper_onehot_chance
        num_warmup_argument = self.args.task_num_warmup if self.args.task_num_warmup is not None else self.args.hyper_onehot_num_warmup
        assert not (self.args.task_chance and self.args.hyper_onehot_chance),         "Cannot have both args.task_chance and args.hyper_onehot_chance"
        assert not (self.args.task_num_warmup and self.args.hyper_onehot_num_warmup), "Cannot have both args.task_num_warmup and args.hyper_onehot_num_warmup"

        assert not (chance_argument is None and num_warmup_argument is not None), "if args.hyper_onehot/task_num_warmup is specified, args.hyper_onehot/task_chance should be specified as well"
        
        if (chance_argument is not None) and training: # just use one_hot task with some chance
            chance = 1 if (num_warmup_argument is not None) and (update_num < num_warmup_argument) else chance_argument # set chance
            if torch.rand(1).item() <= chance:
                return True
        return False

    def reset_nontask_parameters(self, reset_parameters):
        # reset all parameters except for task encoding
        for name, params in self.named_parameters():
            if "task_encoder" not in name:
                with torch.no_grad():
                    params.data.copy_(reset_parameters[name])

    def reset_hyper_parameters(self, reset_parameters):
        # reset all parameters except for head of hypernet
        for name, params in self.named_parameters():
            if "hyper_final_to" not in name:
                with torch.no_grad():
                    params.data.copy_(reset_parameters[name])
        # Note, names of hypernet heads:
            # self.hyper_final_to_critic_weight, self.hyper_final_to_critic_bias,
            # self.hyper_final_to_critic_linear_weight, self.hyper_final_to_critic_linear_bias


    def forward_hyper(self, inputs, hyper_input, out_dim,
            policy_layers, hyper_final_to_policy_weight, hyper_final_to_policy_bias,
            hyper_final_to_policy_linear_weight, hyper_final_to_policy_linear_bias, 
            store_base_params=False, replace_base_params_with_activations=False, 
            one_hot_tasks=None, training=False, update_num=None):
        batch_sz = None if len(inputs.shape)==1 else inputs.shape[0]

        if replace_base_params_with_activations:
            assert not store_base_params
            activations = []
    
       # Forward hypernet (except head)
        assert self.use_hyper
        h_hyper = hyper_input
        for i in range(len(self.hyper_layers)):
            h_hyper = self.hyper_layers[i](h_hyper)
            use_softmax = i == (len(self.hyper_layers)-1) and self.args.hyper_softmax_temp is not None
            af = (lambda x: F.softmax(x/self.args.hyper_softmax_temp, dim=-1)) if use_softmax else self.activation_function
            h_hyper = af(h_hyper)

        # override forward if self.args.hyper_onehot_chance, with some probability
        task_overwrite = self.should_overwrite_task(training, update_num) and (self.args.hyper_onehot_chance is not None)
        if task_overwrite:
            assert one_hot_tasks is not None
            assert one_hot_tasks.shape == h_hyper.shape, "Please make sure layer before hypernet head has dim "+str(one_hot_tasks.shape[-1])
            h_hyper = one_hot_tasks.expand_as(h_hyper) #overwrite

        # Forward policy
        h = inputs
        base_params = []
        for fc, hyper_w_head, hyper_b_head, next_sz in zip(policy_layers, hyper_final_to_policy_weight, hyper_final_to_policy_bias, self.hidden_layers):
            assert fc is not None or hyper_w_head is not None, (fc, hyper_w_head, hyper_b_head)
            if fc is None: # use hypernetwork
                assert hyper_w_head is not None and hyper_b_head is not None, (fc, hyper_w_head, hyper_b_head)
                assert not self.use_film
                weight = hyper_w_head(h_hyper)
                if store_base_params:
                    base_params.append(weight)
                weight = weight.reshape((next_sz, h.shape[-1])) if batch_sz is None else weight.reshape((batch_sz, next_sz, h.shape[-1]))
                bias = hyper_b_head(h_hyper)
                if self.args.hyper_onehot_no_bias and task_overwrite:
                    bias = torch.zeros_like(bias)
                if store_base_params:
                    base_params.append(bias)
                h = linear_batched_weights(h, weight, bias)
            elif hyper_w_head is None: # do not use hypernetwork
                assert hyper_w_head is None and hyper_b_head is None, (fc, hyper_w_head, hyper_b_head)
                h = fc(h) 
            else: # use FiLM
                assert self.use_film, "Must be using FiLM if hyper and base parameters"   
                assert fc.bias is None, fc.bias
                h = fc(h) # base policy, w/o bias
                scaling = hyper_w_head(h_hyper)
                bias = hyper_b_head(h_hyper)
                if self.args.hyper_onehot_no_bias and task_overwrite:
                    bias = torch.zeros_like(bias)
                # assert .99 <= scaling.mean() <= 1.001, scaling   # True until after first train step if init_b method used
                # assert -.01 <= bias.mean() <= .01, bias
                h = h*scaling + bias # feature-wise scaling
                if store_base_params:
                    base_params.append(bias)
                    base_params.append(scaling)
            h = self.activation_function(h)
            if replace_base_params_with_activations:
                activations.append(h)

        # Get final linear parameters (or None if this is not produced by the hypernetwork)
        if hyper_final_to_policy_linear_weight is None: # head of policy is not produced by hypernetwork
            assert not self.head_is_hyper
            final_weight, final_bias, final_scaling = None, None, None
        elif self.use_film:
            # here we will scale the rows of the weight matrix instead of scaling the activations, which is equivalent but better for implementation
            assert self.head_is_hyper
            final_scaling = hyper_final_to_policy_linear_weight(h_hyper)
            final_bias = hyper_final_to_policy_linear_bias(h_hyper)
            final_weight = None
            if store_base_params:
                base_params.append(final_scaling)
                base_params.append(final_bias)
        else:
            assert self.head_is_hyper
            final_weight = hyper_final_to_policy_linear_weight(h_hyper)
            if store_base_params:
                base_params.append(final_weight)
            final_weight = final_weight.reshape((out_dim, h.shape[-1])) if batch_sz is None else final_weight.reshape((batch_sz, out_dim, h.shape[-1]))
            final_bias = hyper_final_to_policy_linear_bias(h_hyper)
            if store_base_params:
                base_params.append(final_bias)
            final_scaling = None
        if replace_base_params_with_activations:
            base_params = activations
        if self.args.hyper_onehot_no_bias and task_overwrite and self.head_is_hyper:
            final_bias = torch.zeros_like(final_bias)
        return h, final_weight, final_bias, base_params, final_scaling

    def forward_actor(self, inputs, hyper_input=None, return_base_params=False, return_actor_activations=False, one_hot_tasks=None, training=False, update_num=None):
        if return_actor_activations:
            activations = []

        # No Hypernet - Forward actor
        # Note: This code should be redundant, but makes the degenerate case clear
        if hyper_input is None:
            h = inputs
            for i in range(len(self.actor_layers)):
                h = self.actor_layers[i](h)
                h = self.activation_function(h)
                if return_actor_activations:
                    activations.append(h)
            if return_actor_activations:
                return activations
            return self.dist(h)

        if self.multinet:
            h = inputs
            for i in range(len(self.net_2_actor_layers[0])):
                h = self.apply_multinet_layer([self.net_2_actor_layers[net][i] for net in range(self.num_nets)], h, hyper_input)
                h = self.activation_function(h)
                if return_actor_activations:
                    activations.append(h)
            if return_actor_activations:
                return activations
            return self.apply_multinet_layer(self.net_2_dist_layer, h, hyper_input, dist_layer=True)

        # Apply hyper and normal layers
        h, final_weight, final_bias, base_params, final_scaling = self.forward_hyper(inputs, hyper_input, self.num_outputs,
                                                         self.actor_layers, self.hyper_final_to_actor_weight, self.hyper_final_to_actor_bias,
                                                         self.hyper_final_to_actor_linear_weight, self.hyper_final_to_actor_linear_bias,
                                                         store_base_params=return_base_params, replace_base_params_with_activations=return_actor_activations, 
                                                         one_hot_tasks=one_hot_tasks, training=training, update_num=update_num)
        d = self.dist(h, weight=final_weight, bias=final_bias, scaling=final_scaling)
        if return_actor_activations:
            base_params.append(h) # base_params replaced with activations
            return base_params
        elif return_base_params:
            return d, base_params
        return d

    def forward_critic(self, inputs, hyper_input=None, return_base_params=False, one_hot_tasks=None, training=False, update_num=None):
        # No Hypernet
        # Note: This code should be redundant, but makes the degenerate case clear
        if hyper_input is None:
            h = inputs
            for i in range(len(self.critic_layers)):
                h = self.critic_layers[i](h)
                h = self.activation_function(h)
            return self.critic_linear(h)

        if self.multinet:
            h = inputs
            for i in range(len(self.net_2_critic_layers[0])):
                h = self.apply_multinet_layer([self.net_2_critic_layers[net][i] for net in range(self.num_nets)], h, hyper_input)
                h = self.activation_function(h)
            return self.apply_multinet_layer(self.net_2_critic_head_layer, h, hyper_input)

        # Apply hyper and normal layers
        h, final_weight, final_bias, base_params, final_scaling = self.forward_hyper(inputs, hyper_input, 1,
                                                         self.critic_layers, self.hyper_final_to_critic_weight, self.hyper_final_to_critic_bias,
                                                         self.hyper_final_to_critic_linear_weight, self.hyper_final_to_critic_linear_bias,
                                                         store_base_params=return_base_params, one_hot_tasks=one_hot_tasks, training=training, update_num=update_num)
        
        if final_weight is None and final_bias is None and final_scaling is None: # Normal
            v = self.critic_linear(h) 
        elif final_scaling is None: # HyperNet
            assert final_weight is not None and final_bias is not None
            v = linear_batched_weights(h, final_weight, final_bias)
        else: # FiLM
            assert final_weight is None and final_bias is not None 
            assert self.use_film
            v = self.critic_linear(h)*final_scaling + final_bias
        
        if return_base_params:
            return v, base_params
        return v

    def apply_multinet_layer(self, multinet_layers, batch, tasks, dist_layer=False):
        # Apply a given layer from the multi-net to the batch using tasks to identify which net to use
        if len(batch.shape) == 1: # add batch dim if needed
            batch = batch.unsqueeze(0)
            tasks = tasks.unsqueeze(0)
            resqueeze = True
        else:
            resqueeze = False

        # Mask by task (i.e. network):
        outputs = None # will be made into a tensor
        normal_dist_outputs2 = None # needed just to keep track of extra params for normal dist
        for task_id in range(self.num_nets):
            net = multinet_layers[task_id]
            batch_task_ids = self.task_2_id(tasks, one_hot=False)
            if type(net) is Categorical:
                # If this layer outputs a distribution, we must stack results into batch then make single distribution
                forward = net(batch, return_dist_params=True)
                if outputs is None:
                    outputs = torch.zeros_like(forward)
                mask = (batch_task_ids == task_id).expand(*forward.shape)
                outputs = torch.where(mask, forward, outputs)
            elif type(net) is DiagGaussian:
                for1, for2 = net(batch, return_dist_params=True)
                for2 = for2.unsqueeze(0).expand(len(batch),self.num_outputs) # this is std for each action. it should be shared across batch
                if outputs is None:
                    outputs = torch.zeros_like(for1)
                if normal_dist_outputs2 is None:
                    normal_dist_outputs2 = torch.zeros_like(for2)
                mask1 = (batch_task_ids == task_id).expand(*for1.shape)
                outputs = torch.where(mask1, for1, outputs)
                mask2 = (batch_task_ids == task_id).expand(*for2.shape)
                normal_dist_outputs2 = torch.where(mask2, for2, normal_dist_outputs2)
            else:
                forward = net(batch)
                if outputs is None:
                    outputs = torch.zeros_like(forward)
                mask = (batch_task_ids == task_id).expand(*forward.shape)
                outputs = torch.where(mask, forward, outputs)

        # Deal with distributions
        if type(net) is Categorical:
            batch = outputs
            if resqueeze:
                batch = batch.squeeze(0)
            batch = FixedCategorical(logits=batch)
        elif type(net) is DiagGaussian:
            batch1 = outputs
            batch2 = normal_dist_outputs2
            if resqueeze:
                batch1 = batch1.squeeze(0)
                batch2 = batch2.squeeze(0)
            batch = FixedNormal(batch1, batch2)
        else:
            batch = outputs
            if resqueeze:
                batch = batch.squeeze(0)

        return batch

    def get_encoded_inputs(self, state, latent, belief, task):
        # handle inputs (normalise + embed)
        
        info = {}

        if self.args.hyper_onehot_chance is not None:
            assert task is not None
            assert self.args.hypernet_input == "latent", self.args.hypernet_input
            assert self.pass_task_to_policy, self.pass_task_to_policy
            assert self.args.pass_task_as_onehot_id, self.argsargs.pass_task_as_onehot_id
            one_hot_task = torch.autograd.Variable(self.task_2_id(task), requires_grad=True)
        else:
            one_hot_task = None

        if self.args.task_chance is not None:
            assert task is not None
            assert self.pass_task_to_policy, self.pass_task_to_policy
            assert self.pass_latent_to_policy, self.pass_latent_to_policy
            assert self.args.policy_task_embedding_dim == self.args.policy_latent_embedding_dim

        if self.pass_state_to_policy:
            # TODO: somehow don't normalise the "done" flag (if existing)
            if self.norm_state:
                state = (state - self.state_rms.mean) / torch.sqrt(self.state_rms.var + 1e-8)
            if self.use_state_encoder:
                if self.multinet: # use appropriate network for each item in the batch
                    state = self.apply_multinet_layer(self.net_2_st_layer, state, task)
                else:
                    state = self.state_encoder(state)
        else:
            state = torch.zeros(0, ).to(device)
        if self.pass_latent_to_policy and latent is not None:
            if self.norm_latent:
                latent = (latent - self.latent_rms.mean) / torch.sqrt(self.latent_rms.var + 1e-8)
            if self.use_latent_encoder:
                latent = self.latent_encoder(latent)
            if len(latent.shape) == 1 and len(state.shape) == 2:
                latent = latent.unsqueeze(0)
        else:
            latent = torch.zeros(0, ).to(device)
        if self.pass_belief_to_policy:
            if self.norm_belief:
                belief = (belief - self.belief_rms.mean) / torch.sqrt(self.belief_rms.var + 1e-8)
            if self.use_belief_encoder:
                belief = self.belief_encoder(belief)
            belief = belief.float()
            if len(belief.shape) == 1 and len(state.shape) == 2:
                belief = belief.unsqueeze(0)
        else:
            belief = torch.zeros(0, ).to(device)
        if self.pass_task_to_policy:
            if self.args.pass_task_as_onehot_id:
                one_hot_task = torch.autograd.Variable(self.task_2_id(task), requires_grad=True)
                info["one_hot_task"] = one_hot_task
                task = one_hot_task # convert description to one_hot id
            if self.norm_task:
                task = (task - self.task_rms.mean) / torch.sqrt(self.task_rms.var + 1e-8)
            if self.use_task_encoder:
                task = self.task_encoder(task.float())
            if len(task.shape) == 1 and len(state.shape) == 2:
                task = task.unsqueeze(0)
            task = task.float()
        else:
            task = torch.zeros(0, ).to(device)

        return state, latent, belief, task, one_hot_task, info


    def forward(self, state, latent, belief, task, return_base_params=False, return_info=False, 
        return_actor_activations=False, training=False, update_num=None, force_task_input=False, force_latent_input=False):

        state, latent, belief, task, one_hot_task, info = self.get_encoded_inputs(state, latent, belief, task)

        assert not (force_task_input and force_latent_input)

        # concatenate inputs
        if self.use_hyper:
            if self.args.task_chance is not None:
                if (self.should_overwrite_task(training, update_num) or force_task_input) and not force_latent_input:
                    inputs = state
                    hyper_input = task
                else: 
                    inputs = state
                    hyper_input = latent
            elif self.args.hyper_onehot_chance is not None:
                task = torch.zeros(0, ).to(device) # reset to 0, we already have one_hot task saved in one_hot_task
                inputs = torch.cat((state, belief, task), dim=-1)
                hyper_input = latent
                one_hot_task = one_hot_task.float()
            elif (self.args.hypernet_input == "task_embed"):
                inputs = torch.cat((state, latent, belief), dim=-1)
                hyper_input = task
            else:
                assert self.args.hypernet_input == "latent", args.hypernet_input
                inputs = torch.cat((state, belief, task), dim=-1)
                hyper_input = latent
        elif self.multinet: # train multiple networks
            inputs = torch.cat((state, latent, belief), dim=-1)
            hyper_input = task # Note: I call this hyper_input but really no hypernet used
        elif self.args.task_chance is not None: # Single network, No HN, but using task sometimes and latent sometimes
            if (self.should_overwrite_task(training, update_num) or force_task_input) and not force_latent_input:
                inputs = torch.cat((state, task), dim=-1)
                # if training and not force_task_input:
                #     print("task")
            else: 
                inputs = torch.cat((state, latent), dim=-1)
                # if training and not force_latent_input:
                #     print("latent")
            hyper_input = None
        else: # No HN, normal case
            inputs = torch.cat((state, latent, belief, task), dim=-1)
            hyper_input = None

        # forward through critic/actor part
        if return_actor_activations:
            return self.forward_actor(inputs, hyper_input=hyper_input, return_actor_activations=True, one_hot_tasks=one_hot_task, training=training, update_num=update_num) 
        elif return_base_params:
            assert self.use_hyper
            value, c_base_params = self.forward_critic(inputs, hyper_input=hyper_input, return_base_params=True, one_hot_tasks=one_hot_task, training=training, update_num=update_num)
            dist, a_base_params = self.forward_actor(inputs, hyper_input=hyper_input, return_base_params=True, one_hot_tasks=one_hot_task, training=training, update_num=update_num) 
            if return_info: 
                return value, dist, c_base_params+a_base_params, info
            else:
                return value, dist, c_base_params+a_base_params
        else:
            value = self.forward_critic(inputs, hyper_input=hyper_input, return_base_params=False, one_hot_tasks=one_hot_task, training=training, update_num=update_num)
            dist = self.forward_actor(inputs, hyper_input=hyper_input, return_base_params=False, one_hot_tasks=one_hot_task, training=training, update_num=update_num)
            if return_info:
                return value, dist, info
            else:
                return value, dist

    def get_dist(self, policy_initialisation, action_space, hidden_layers, init_std, action_low, action_high, norm_actions_of_policy):
        if action_space.__class__.__name__ == "Discrete":
            actor_head_init = policy_initialisation if policy_initialisation == 'kaiming' else 'orthogonal' # leave kaiming alone, but otherwise use orthogonal
            dist_gain = 0.01
            num_outputs = action_space.n
            dist = Categorical(hidden_layers[-1], num_outputs, dist_gain, create_params = self.use_film or not self.head_is_hyper)
        elif action_space.__class__.__name__ == "Box":
            actor_head_init = policy_initialisation if policy_initialisation == 'kaiming' else 'normc' # leave kaiming alone, but otherwise use normc
            dist_gain = 1.0
            num_outputs = action_space.shape[0]
            dist = DiagGaussian(hidden_layers[-1], num_outputs, init_std, min_std=1e-6,
                                     action_low=action_low, action_high=action_high,
                                     norm_actions_of_policy=norm_actions_of_policy, 
                                     gain=dist_gain, create_params = self.use_film or not self.head_is_hyper)
        else:
            raise NotImplementedError
        return dist, dist_gain, num_outputs, actor_head_init

    def act(self, state, latent, belief, task, deterministic=False, training=False, update_num=None):
        value, dist = self.forward(state=state, latent=latent, belief=belief, task=task, training=training, update_num=update_num)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs

    def get_value(self, state, latent, belief, task, training=False, update_num=None):
        value, _ = self.forward(state, latent, belief, task, training=training, update_num=update_num)
        return value

    def update_rms(self, args, policy_storage):
        """ Update normalisation parameters for inputs with current data """
        if self.pass_state_to_policy and self.norm_state:
            state = policy_storage.prev_state[:-1]
            self.state_rms.update(state)
        if self.pass_latent_to_policy and self.norm_latent:
            latent = utl.get_latent_for_policy(args,
                                               torch.cat(policy_storage.latent_samples[:-1]),
                                               torch.cat(policy_storage.latent_mean[:-1]),
                                               torch.cat(policy_storage.latent_logvar[:-1])
                                               )
            self.latent_rms.update(latent)
        if self.pass_belief_to_policy and self.norm_belief:
            self.belief_rms.update(policy_storage.beliefs[:-1])
        if self.pass_task_to_policy and self.norm_task:
            task = policy_storage.tasks[:-1]
            task = self.task_2_id(task) if self.args.pass_task_as_onehot_id else task # convert to one-hot if needed
            self.task_rms.update(task)

    def task_2_id(self, task, one_hot=True):
        tasks = self.dummy_env.task_to_id(task).to(device)
        if not one_hot:
            return tasks.int()
        one_hot_tasks = F.one_hot(tasks, num_classes=self.dummy_env.num_tasks)
        # or equivalently:
        # classes = dummy_env.task_to_id(task).to(device)
        # one_hot_classes = torch.zeros((classes.shape[0],dummy_env.num_tasks), dtype=tasks.dtype)
        # one_hot_classes[:,classes] = 1
        # assert torch.equal(one_hot_classes, one_hot_tasks)
        if len(one_hot_tasks.shape) == 2 and len(task.shape) == 1:
            one_hot_tasks = torch.squeeze(one_hot_tasks)
            assert len(one_hot_tasks.shape) == len(task.shape), (len(one_hot_tasks.shape), len(task.shape))
        return torch.squeeze(one_hot_tasks.float())

    def evaluate_actions(self, state, latent, belief, task, action, return_action_mean=False, return_base_params=False, return_info=False, training=False, update_num=None, force_task_input=False, force_latent_input=False, return_only_bn_params=False):

        if return_base_params:
            value, dist, base_params, info = self.forward(state, latent, belief, task, return_info=True, return_base_params=True, training=training, update_num=update_num, force_task_input=force_task_input, force_latent_input=force_latent_input)
        else:
            value, dist, info = self.forward(state, latent, belief, task, return_info=True, training=training, update_num=update_num, force_task_input=force_task_input, force_latent_input=force_latent_input)

        if return_only_bn_params:
            assert return_base_params
            return base_params

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        to_return = (value, action_log_probs, dist_entropy)
        if return_action_mean:
            to_return = to_return + (dist.mode(), dist.stddev)
        if return_base_params:
            to_return = to_return + (base_params,)
        if return_info:
            to_return = to_return + (info,)
        return to_return


FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(self, actions.squeeze(-1)).unsqueeze(-1)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=-1, keepdim=True)

FixedNormal = torch.distributions.Normal
log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)

entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean


def init(module, weight_init, bias_init, gain=1.0, gain_for_bias_also=False):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        if gain_for_bias_also:
            bias_init(module.bias.data, gain=gain)
        else:
            bias_init(module.bias.data)
    return module


# https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))


def total_num_params(layer_sizes):
    num_params = 0
    for dim_in, dim_out in layer_sizes:
        num_params += dim_in*dim_out + dim_out # num params in linear layer
    return num_params


def linear_batched_weights(x, weight, bias):
    # a generalization of F.linear that allows for a different weight and bias for each item in the batch
    # Otherwise, F.linear fails on transposing the 3D weight tensor
    # E.g. This should pass the assert:
    if len(weight.shape)==2:
        return F.linear(x, weight, bias)
    assert len(weight.shape)==3, weight.shape
    assert len(x) == len(weight), (x.shape, weight.shape)

    w_T = weight.transpose(-1,-2)
    v = (x.unsqueeze(-1).expand_as(w_T)*w_T).sum(-2) + bias
    
    # You can check with this slow assert (comment out for speed):
    # batch_sz = len(x)
    # slow_results = []
    # for b in range(batch_sz):
    #     slow_results.append(F.linear(x[b], weight[b], bias[b]))
    # slow_results = torch.stack(slow_results)
    # assert v.shape == slow_results.shape, (v.shape, slow_results.shape)
    # assert torch.allclose(v, slow_results, atol=1e-05, rtol=0), (v, slow_results)

    return v

# kaiming uniform from https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_uniform_
# but with gain argument instead of 'nonlinearity'
def kaiming_uniform_with_gain(tensor, mode='fan_in', gain=1):
    fan = nn.init._calculate_correct_fan(tensor, mode)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)

def kaiming_uniform_ignore_gain(tensor, gain=None):
    return nn.init.kaiming_uniform_(tensor)

# Hyperfan-In uniform for bias from (Chang et al., 2020), assuming ReLU and producing weights and bias
def HFI_bias_layer_init(final_hyper_hidden_sz, gain=1): # This gain should be passed as sqrt(2) for RelU
    variance = 1/(2*final_hyper_hidden_sz) # = 1/(2 * dl), since the rest of the terms cancel. (numerator will be 2 here if gain is passed for relu)
    bound = gain*np.sqrt(3*variance) # gain is sqrt(2) for RelU
    return lambda m: init(m, lambda w, gain: nn.init.uniform_(w, -bound, bound), 
                             lambda b: nn.init.constant_(b, 0), 
                             gain=None, gain_for_bias_also=False)

# Hyperfan-In uniform for weight from (Chang et al., 2020), assuming ReLU and producing weights and bias
def HFI_weight_layer_init(final_hyper_hidden_sz, base_curr_input_dim, gain=1): # This gain should be passed as sqrt(2) for RelU
    variance = 1/(2*final_hyper_hidden_sz*base_curr_input_dim) # = 1/(2 * dj * dk), since the rest of the terms cancel
    bound = gain*np.sqrt(3*variance) # gain is sqrt(2) for RelU
    return lambda m: init(m, lambda w, gain: nn.init.uniform_(w, -bound, bound),
                             lambda b: nn.init.constant_(b, 0), 
                             gain=None, gain_for_bias_also=False)

# Defines the initialization of the weight or biases of the hyper-network so that each set of produced parameters is reasonable. 
# (Used in heads of the hypernetwork that output a weight matrix)
def init_hyper_match(param, is_weight, policy_initialisation_str, hyper_layer_dim, input_dim, output_dim, adjust_init_b_gain, gain=None, scale=None):
    if policy_initialisation_str == 'normc':
        policy_initialisation = init_normc_ 
    elif policy_initialisation_str == 'orthogonal':
        policy_initialisation = nn.init.orthogonal_
    else:
        assert policy_initialisation_str == 'kaiming', policy_initialisation_str
        policy_initialisation = kaiming_uniform_with_gain if adjust_init_b_gain else kaiming_uniform_ignore_gain
    # Assuming that the final layer (of size hyper_layer_dim) is one-hot, we want each weight matrix produced
    # of shape (input_dim, output_dim) to produce the original policy_initialisation
    if is_weight:
        assert param.shape == (input_dim*output_dim, hyper_layer_dim)
    else:
        assert param.shape == (input_dim*output_dim,), (param.shape, input_dim*output_dim)
    # each column is a weight matrix for the policy, so init each column
    # original_data = param.data.clone()
    if is_weight:
        for col_indx in range(hyper_layer_dim):
            col = param[:, col_indx]
            policy_weight = col.reshape((output_dim, input_dim)) # This should be the shape expected for the init tensor inputs
            policy_initialisation(policy_weight, gain=gain)
    else:
        policy_weight = param.reshape((output_dim, input_dim))
        policy_initialisation(policy_weight, gain=gain)
    # original_data = param.data.clone()
    if scale:
        with torch.no_grad():
            param.data.copy_(scale * param.data)

# How to initialize the hypernetwork head (weight and bias) that produces biases for the base network
def hyper_bias_layer_init(weight_for_uniform=None): # weight_for_uniform needed to use default initialization from Linear class
    # TODO: this assumes that the bias in the base net is created by bias in the hypernet. Normally this is 0, so it doesn't matter,
    # but the final layer of the critic by default has bias initialized with uniform init. (When weight_for_uniform is not None.)
    # In this case, we may want to allow for the option to generate the bias init using weights. (But I doubt this will affect anything.)
    if weight_for_uniform is not None:
        # default behavior from Linear class
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight_for_uniform)
        bound = 1 / math.sqrt(fan_in)
        init_for_hyper_bias = lambda b: nn.init.uniform_(b, -bound, bound)
    else:
        init_for_hyper_bias = lambda b: nn.init.constant_(b, 0)
    # get an init function that sets weight and bias to 0 so that bias produced is 0
    return lambda m: init(m, lambda w, gain: nn.init.constant_(w, 0), 
                             init_for_hyper_bias, 
                             None)
                
# How to initialize the hypernetwork head (weight and bias) that produces weights for the base network
def hyper_weight_layer_init(activation_function, policy_initialisation_str, hyper_layer_dim, input_dim, output_dim, adjust_init_b_gain, override_gain=None,
                            adjust_weights=True, adjust_bias=False, use_film=False):
    if override_gain is None:
        override_gain = nn.init.calculate_gain(activation_function)
    # get an init function that sets each weight so it is similar to original policy weight init
    scale = .5 if adjust_weights and adjust_bias else None
    # define weight init
    if adjust_weights:
        if use_film:
            weight_init = lambda w, gain: nn.init.constant_(w, 1 if scale is None else scale) # In FiLM, init all weights that produce scaling to 1 so that one-hot input produces scaling = 1. (.5 is shared with bias)
        else:
            weight_init = lambda w, gain: init_hyper_match(w, True, policy_initialisation_str, hyper_layer_dim, input_dim, output_dim, adjust_init_b_gain, gain=gain, scale=scale)
    else:
        weight_init = lambda w, gain: nn.init.constant_(w, 0)
    # define bias init
    if adjust_bias:
        if use_film:
            bias_init = lambda b: nn.init.constant_(b, 1 if scale is None else scale) # In FiLM, init all bias that produce scaling to 1 so that it produces scaling = 1. (.5 is shared with weights)
        else:
            gain_for_bias_also = True
            bias_init = lambda b, gain: init_hyper_match(b, False, policy_initialisation_str, hyper_layer_dim, input_dim, output_dim, adjust_init_b_gain, gain=gain, scale=scale)
    else:
        gain_for_bias_also = False
        bias_init = lambda b: nn.init.constant_(b, 0)
    # define layer initializer
    if use_film:
        override_gain = 1 # all gain on weights should be taken care of through weight init in base net
        gain_for_bias_also = False
    return lambda m: init(m, weight_init, 
                             bias_init, 
                             override_gain, gain_for_bias_also=gain_for_bias_also)

class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, gain, create_params=True):
        super(Categorical, self).__init__()

        self.create_params = create_params

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               gain=gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs)) if self.create_params else None

    def forward(self, x, weight=None, bias=None, scaling=None, return_dist_params=False):
        if (weight is None) and (bias is None): # Normal
            assert self.create_params
            x = self.linear(x)
        elif scaling is None: # HyperNet
            assert (weight is not None) and (bias is not None), "Must have weight and bias for HyperNet" 
            x = linear_batched_weights(x, weight, bias)
        else: # FiLM
            assert (weight is None) and (bias is not None)
            self.linear.bias = None
            x = self.linear(x)
            x = x*scaling + bias
        if return_dist_params:
            return x
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, init_std, min_std,
                 action_low, action_high, norm_actions_of_policy, gain, create_params=True):
        # create_params=False allows forward to take in a linear layer parameters. Necessary when using hypernetwork.
        # Note: create_params=False WILL still create logstd parameter, shared across all networks
        super(DiagGaussian, self).__init__()

        self.create_params = create_params

        init_ = lambda m: init(m,
                               init_normc_,
                               lambda x: nn.init.constant_(x, 0),
                               gain=gain)

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs)) if self.create_params else None
        self.logstd = nn.Parameter(np.log(torch.zeros(num_outputs) + init_std)) # Always created (easier to implement)
        self.min_std = torch.tensor([min_std]).to(device)

        # whether or not to conform to the action space given by the env
        # (scale / squash actions that the network outpus)
        self.norm_actions_of_policy = norm_actions_of_policy
        if len(np.unique(action_low)) == 1 and len(np.unique(action_high)) == 1:
            self.unique_action_limits = True
        else:
            self.unique_action_limits = False

        self.action_low = torch.from_numpy(action_low).to(device)
        self.action_high = torch.from_numpy(action_high).to(device)

    def forward(self, x, weight=None, bias=None, scaling=None, return_dist_params=False):
        if (weight is None) and (bias is None): # Normal
            assert self.create_params
            action_mean = self.fc_mean(x)
        elif scaling is None: # HyperNet
            assert (weight is not None) and (bias is not None), "Must have weight and bias for HyperNet" 
            action_mean = linear_batched_weights(x, weight, bias)
        else: # FiLM
            assert (weight is None) and (bias is not None)
            self.fc_mean.bias = None
            action_mean = self.fc_mean(x)
            action_mean = action_mean*scaling + bias

        if self.norm_actions_of_policy:
            if self.unique_action_limits and \
                    torch.unique(self.action_low) == -1 and \
                    torch.unique(self.action_high) == 1:
                action_mean = torch.tanh(action_mean)
            else:
                # Note: this isn't tested
                action_mean = torch.sigmoid(action_mean) * (self.action_high - self.action_low) + self.action_low
        std = torch.max(self.min_std, self.logstd.exp())
        if return_dist_params:
            return action_mean, std
        return FixedNormal(action_mean, std)


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().reshape(1, -1)
        else:
            bias = self._bias.t().reshape(1, -1, 1, 1)

        return x + bias
