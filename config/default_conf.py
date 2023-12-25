import argparse
from utils.helpers import boolean_argument, float_or_none, int_or_none


def get_args(rest_args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--compute_grad_info', type=boolean_argument, default=False, help='whether to log cosine similarity of gradients between tasks and norms of gradients (adds computation)')

    parser.add_argument('--multinet', type=boolean_argument, default=False, help='Instead of using a hypernet, implement multi-net slow way for debugging.')

    parser.add_argument('--use_hypernet', type=boolean_argument, default=False, help='whether to use a hypernetwork to condition policy on task embedding/latent')
    parser.add_argument('--hypernet_input', type=str, default='task_embed', help='\'task_embed\' or \'latent\' will condition hypernet on task embedding or latent respectively')
    parser.add_argument('--init_hyper_for_policy', type=boolean_argument, default=True, help='whether to init the hypernetwork head so that it produces similar initial policy weights')
    parser.add_argument('--init_adjust_weights', type=boolean_argument, default=True, help='whether to adjust hypernet weights so that init_hyper_for_policy works')
    parser.add_argument('--init_adjust_bias', type=boolean_argument, default=False, help='whether to adjust hypernet bias so that init_hyper_for_policy works')
    parser.add_argument('--hypernet_head_bias', type=boolean_argument, default=True, help='whether the hypernetwork has bias on its output parameters')
    parser.add_argument('--policy_layers_hyper', nargs='+', default=['1', '1', '1'], 
        help='Whether each layer in the policy net should be produced by the hypernet. 1 is True, 0 False. Length must equal length of --policy_layers + 1 (for the head).')
    parser.add_argument('--hypernet_layers', nargs='+', default=[], 
        help=('hidden layers defining the a hypernetwork.'
              '\nNote, the following is equivalent to a multi-head network with one head (set of parameters) per task'
              '\nhypernet_layers=[] along with:'
              '\n--hypernet_head_bias=False, --policy_layers_hyper=[0,0,1], --pass_task_as_onehot_id=True,  --policy_task_embedding_dim=None, --norm_task_for_policy=False'))
    parser.add_argument('--hyper_softmax_temp', type=float_or_none, default=None, help='If specified, the activations in the hypernet before the last layer will use a softmax with this temperature')
    
    parser.add_argument('--hyper_onehot_chance', type=float_or_none, default=None, help='If specified, the hypernetwork will use simultaneous one-hot multi-task training to help learn weights with this probability.')
    parser.add_argument('--hyper_onehot_num_warmup', type=int_or_none, default=None, help='If specified, the probability for hyper_onehot_chance will be set to 1 for this number of updates at the start of meta-training.')
    parser.add_argument('--hyper_onehot_hard_reset', type=boolean_argument, default=False, help='Whether to reset all policy parameters trained in the warmup other than the columns in the last hypernet layer.')
    parser.add_argument('--hyper_onehot_no_bias', type=boolean_argument, default=False, help='Whether to set bias in last hypernet head to 0 during warmup.')
    
    parser.add_argument('--task_chance', type=float_or_none, default=None, help='If specified, will use simultaneous multi-task training (with normal task embedding) to help learn weights with this probability.')
    parser.add_argument('--task_num_warmup', type=int_or_none, default=None, help='If specified, the probability for task_chance will be set to 1 for this number of updates at the start of meta-training.')
    parser.add_argument('--freeze_vae_during_warmup', type=boolean_argument, default=False, help='If specified, the recurrent encoder will not be trained during multi-task pre-training.')

    parser.add_argument('--ti_target', type=str, default='task', help='\'task\' or \'base_net\'. Whether to use the task encoding or base_network as a task inference target.')
    parser.add_argument('--ti_target_stop_grad', type=boolean_argument, default=False, help='Whether to add a stop grad to ti target. Recommended if not continually training task encoding.')
    parser.add_argument('--ti_coeff', type=float_or_none, default=None, help='If specified, will add a task inference loss with the given weight. Target determined by ti_target. Note: uses learned task embedding, unlike --decode_task for vae')
    parser.add_argument('--ti_hard_reset', type=boolean_argument, default=False, help='Whether to reset all parameters other than task encoding (if target == task), for task inference.')

    parser.add_argument('--use_film', type=boolean_argument, default=False, help='whether to use a FiLM layer by making a simplified hypernetwork. Requires use_hypernet=True')

    parser.add_argument('--pass_task_as_onehot_id', type=boolean_argument, default=False, help='whether to convert task description to one_hot ID')

    parser.add_argument('--policy_layers', nargs='+', default=[32, 32])

    parser.add_argument('--vae_resample_decodes', type=boolean_argument, default=False, help='Instead of vae_subsample_decodes, you can re-sample so they have the same (max) length')

    parser.add_argument('--adjust_init_b_gain', type=boolean_argument, default=False, help='False seems so work fine and True has not been tested. With this set to False, init_b and init_w will use the standard gain (root(2)),'
        '\nwhich is correct only for ReLU units. Setting this to True is needed for other activation types.'
        '\nAdditionally, setting this to True is needed if you want to adjust the gain for the head (distribution layer) to another value.'
        '\nIn VariBAD, these are modified by default to 0.01 for categorical distributions (i.e. in gridworld) and 1 for continuous distributions (as is standard for linear layers)')

    parser.add_argument('--HN_init', type=str, default=None, help='if using init_hyper_for_policy to adjust the head of the hypernet (i.e. init_b or init_w),'
        '\nthen the init can also be different for initial hypernetwork layers, so you can specify a different init with HN_init.'
        '\nOtherwise defaults to policy_initialisation, which may change other layers too.')

    # new for aggregator:
    parser.add_argument('--encoder_layers_after_agg', nargs='+', type=int, default=[])
    parser.add_argument('--encoder_enc_type', type=str, default="gru", help='type of encoder in Aggregator, e.g. gru, transition, amu, None')
    parser.add_argument('--encoder_skip_type', type=str, default=None, help='type of skip connection in Aggregator, e.g. enc, None')
    parser.add_argument('--encoder_agg_type', type=str, default=None, help='type of aggregation in Aggregator, e.g. max, avg, gauss')
    parser.add_argument('--encoder_st_estimator', type=boolean_argument, default=False, help='whether to pass grad estimate in Aggregator straight through')
    parser.add_argument('--encoder_max_init_low', type=boolean_argument, default=False, help='whether to start max aggregator state at an arbitrary low constant')
    parser.add_argument('--learn_init_state', type=boolean_argument, default=False, help='whether to learn the initial state for the encoder')



    return parser.parse_args(rest_args)