'''
This file defines models used in experiments.
The functions modify models (represented by dictionaries.)
The dictionaries are specified at the end of this file.
'''


def onehot_task_chance(exp_dict, chance):
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = "TaskChance" + str(int(100*chance)) +"_" + exp_dict["exp_label"]
    new_dict["hyper_onehot_chance"] = chance 
    new_dict["pass_task_to_policy"] = True
    new_dict["pass_latent_to_policy"] = True
    new_dict["pass_task_as_onehot_id"] = True
    new_dict["policy_task_embedding_dim"] = None
    return new_dict

def full_task_chance(exp_dict, chance):
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = "FullTaskChance" + str(int(100*chance)) +"_" + exp_dict["exp_label"]
    new_dict["task_chance"] = chance 
    new_dict["pass_task_to_policy"] = True
    new_dict["pass_latent_to_policy"] = True
    new_dict["pass_task_as_onehot_id"] = False
    return new_dict

def num_warmups(exp_dict, updates, hard=False):
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = "Warmups" + str(updates) +"_" + new_dict["exp_label"]
    new_dict["hyper_onehot_num_warmup"] = updates 
    if hard:
        new_dict["hyper_onehot_hard_reset"] = True
        new_dict["hyper_onehot_no_bias"] = True
        new_dict["exp_label"] = "hard" + "_" + new_dict["exp_label"]
    return new_dict

def freeze_vae_warmup(exp_dict):
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = "freezeVae_" + new_dict["exp_label"]
    new_dict["freeze_vae_during_warmup"] = True 
    return new_dict

def vis_10(exp_dict):
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = new_dict["exp_label"] + "_vis10"
    new_dict["vis_interval"] = 10 
    return new_dict

def policy_num_steps(exp_dict, pns):
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = new_dict["exp_label"] + "_pns" + str(pns)
    new_dict["policy_num_steps"] = pns 
    return new_dict

def ti(exp_dict, weight, base_net=False, stopgrad=False, hard=False):
    new_dict = exp_dict.copy()
    if base_net:
        new_dict["exp_label"] = "bn_" + new_dict["exp_label"]
        new_dict["ti_target"] = "base_net"
    if stopgrad:
        new_dict["exp_label"] = "sg_" + new_dict["exp_label"]
        new_dict["ti_target_stop_grad"] = True
    if hard:
        new_dict["exp_label"] = "hard_" + new_dict["exp_label"]
        new_dict["ti_hard_reset"] = True
    new_dict["exp_label"] = "TI" + str(int(100*weight)) + "_" + new_dict["exp_label"]
    new_dict["ti_coeff"] = weight
    new_dict["decode_task"] = True
    new_dict['task_pred_type'] = 'task_description'
    return new_dict

def ti_naive(exp_dict, weight, task_sz):
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = "TINaive" + str(int(100*weight)) + "_" + new_dict["exp_label"]
    new_dict["task_loss_coeff"] = weight
    new_dict["decode_task"] = True
    new_dict['task_pred_type'] = 'task_description'
    new_dict["ti_coeff"] = None
    if task_sz is None:
        new_dict["exp_label"] = new_dict["exp_label"] + "_TNone"
    else: # only have to change this for RL2 methods, not varibad methods
        new_dict["policy_latent_embedding_dim"] = task_sz
    return new_dict

def num_full_warmups(exp_dict, updates):
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = "WarmupsFull" + str(updates) +"_" + exp_dict["exp_label"]
    new_dict["task_num_warmup"] = updates 
    return new_dict

def div_lr(exp_dict, factor, adjust_vae_lr=True): # useful to adjust LR of experiment to create new one
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = exp_dict["exp_label"] + "_lrD" + str(int(factor))
    new_dict["lr_policy"] = exp_dict["lr_policy"]/factor
    if adjust_vae_lr:
        new_dict["lr_vae"] = exp_dict["lr_vae"]/factor
    return new_dict

def vae_lr(exp_dict, lr):
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = exp_dict["exp_label"] + "_vaeLR" + format(lr, '.0e')
    new_dict["lr_vae"] = lr
    return new_dict

def make_RL2_fixedLR(exp_dict, lr=0.0007): # useful to turn VariBAD into RL2 with fixed LR. Note: still variational
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = "RL2_" + exp_dict["exp_label"]
    new_dict["lr_policy"] = lr
    new_dict["lr_vae"] = lr
    new_dict["disable_decoder"] = True 
    new_dict["rlloss_through_encoder"] = True
    new_dict["norm_latent_for_policy"] = False
    return new_dict

def make_varibad(exp_dict):
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = "Varibad_" + exp_dict["exp_label"]
    new_dict["disable_decoder"] = False 
    new_dict["rlloss_through_encoder"] = False
    new_dict["norm_latent_for_policy"] = True
    return new_dict

def add_RLenc(exp_dict): # add RL enc loss to VariBAD
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = exp_dict["exp_label"] + "_RLenc"
    new_dict["rlloss_through_encoder"] = True
    new_dict["norm_latent_for_policy"] = False
    return new_dict

def hidden_512(exp_dict): # add RL enc loss to VariBAD
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = exp_dict["exp_label"] + "_512H"
    new_dict["encoder_gru_hidden_size"] = "512"
    return new_dict

def make_smallPol(exp_dict): # useful to make HN base policy smaller
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = exp_dict["exp_label"] + "_smallPol"
    new_dict["policy_layers"] = "32 32"
    return new_dict

def make_smallEm(exp_dict): # useful to make HN state and latent embedding small
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = exp_dict["exp_label"] + "_smallEm"
    new_dict["policy_state_embedding_dim"] = 64
    new_dict["policy_latent_embedding_dim"] = 10
    new_dict["policy_task_embedding_dim"] = 10
    return new_dict

def make_skip(exp_dict):
    new_dict = exp_dict.copy()
    new_dict["encoder_skip_type"] = "enc"
    new_dict["exp_label"] = exp_dict["exp_label"] + "_skip"
    return new_dict

def make_nonVar(exp_dict): # makes method non-variational
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = exp_dict["exp_label"] + "_nonVar"
    new_dict["disable_stochasticity_in_latent"] = True
    new_dict["sample_embeddings"] = False # This is the default, but just be clear
    return new_dict

def make_nonVarNP(exp_dict): # makes method non-variational
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = exp_dict["exp_label"] + "_nonVar"
    new_dict["disable_stochasticity_in_latent"] = True
    new_dict["sample_embeddings"] = False # This is the default, but just be clear
    new_dict["precollect_len"] = 0
    return new_dict

def make_nonVarNPP(exp_dict): # makes method non-variational
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = exp_dict["exp_label"] + "_nonVarNPP"
    new_dict["disable_stochasticity_in_latent"] = True
    new_dict["sample_embeddings"] = False # This is the default, but just be clear
    new_dict["precollect_len"] = 0
    new_dict["pretrain_len"] = 0
    return new_dict
    
def make_var(exp_dict): # makes method variational
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = exp_dict["exp_label"] + "_var"
    new_dict["disable_stochasticity_in_latent"] = False
    new_dict["sample_embeddings"] = True
    return new_dict

def truncate(exp_dict, trunc_len=50):
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = exp_dict["exp_label"] + "_trunc" + str(int(trunc_len))
    new_dict["tbptt_stepsize"] = trunc_len
    return new_dict

def make_sum(exp_dict):
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = exp_dict["exp_label"] + "_sum"
    new_dict["encoder_agg_type"] = "sum"
    return new_dict

def make_avg(exp_dict):
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = exp_dict["exp_label"] + "_avg"
    new_dict["encoder_agg_type"] = "avg"
    return new_dict

def make_wavg(exp_dict):
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = exp_dict["exp_label"] + "_wavg"
    new_dict["encoder_agg_type"] = "weighted_avg"
    return new_dict

def make_max(exp_dict):
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = exp_dict["exp_label"] + "_max"
    new_dict["encoder_agg_type"] = "max"
    return new_dict

def make_am(exp_dict):
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = exp_dict["exp_label"] + "_am"
    new_dict["encoder_agg_type"] = "am"
    return new_dict

def make_softmax(exp_dict):
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = exp_dict["exp_label"] + "_softmax"
    new_dict["encoder_agg_type"] = "softmax"
    return new_dict

def make_max_norm(exp_dict):
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = exp_dict["exp_label"] + "_max_norm"
    new_dict["encoder_agg_type"] = "max_norm"
    return new_dict

def make_mlowinit(exp_dict):
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = exp_dict["exp_label"] + "_mlowinit"
    new_dict["encoder_max_init_low"] = True
    return new_dict

def make_gauss(exp_dict):
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = exp_dict["exp_label"] + "_gauss"
    new_dict["encoder_agg_type"] = "gauss"
    return new_dict

def make_trans(exp_dict):
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = exp_dict["exp_label"] + "_trans"
    new_dict["encoder_enc_type"] = "transition"
    return new_dict

def noRNN(exp_dict):
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = exp_dict["exp_label"] + "_noRNN"
    new_dict["encoder_enc_type"] = None
    return new_dict

def mru(exp_dict, norm):
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = new_dict["exp_label"] + "_mru"
    new_dict["encoder_enc_type"] = "mru"
    new_dict["mru_norm"] = norm
    new_dict["exp_label"] = new_dict["exp_label"] + "_norm" + str(norm)
    return new_dict

def lstm(exp_dict):
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = new_dict["exp_label"] + "_lstm"
    new_dict["encoder_enc_type"] = "lstm"
    return new_dict

def rnn_sz(exp_dict, sz):
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = exp_dict["exp_label"] + "_" + str(sz)
    new_dict["encoder_gru_hidden_size"] = sz
    return new_dict

def make_st(exp_dict):
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = exp_dict["exp_label"] + "_ST"
    new_dict["encoder_st_estimator"] = True
    return new_dict

def make_transformer(exp_dict, token_sz="default", num_layers="default", num_heads="default", mem_sz="default"):
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = exp_dict["exp_label"] + "_transformer"
    new_dict["encoder_enc_type"] = "transformer"
    if token_sz != "default":
        new_dict["transformer_token_sz"] = token_sz
        token_sz = None if token_sz is None else int(token_sz)
        new_dict["exp_label"] += f"_TokenSZ={token_sz}"
    if num_layers != "default":
        new_dict["transformer_num_layers"] = num_layers
        num_layers = None if num_layers is None else int(num_layers)
        new_dict["exp_label"] += f"_NTLayers={num_layers}"
    if num_heads != "default":
        new_dict["transformer_num_heads"] = num_heads
        num_heads = None if num_heads is None else int(num_heads)
        new_dict["exp_label"] += f"_NHeads={num_heads}"
    if mem_sz != "default":
        new_dict["transformer_mem_sz"] = mem_sz
        mem_sz = None if mem_sz is None else int(mem_sz)
        new_dict["exp_label"] += f"_TMemSz={mem_sz}"
    return new_dict

def make_learn_init_state(exp_dict):
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = exp_dict["exp_label"] + "_learnHS"
    new_dict["learn_init_state"] = True
    return new_dict

def make_multi(exp_dict):
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = "multi_" + new_dict["exp_label"]
    new_dict["hypernet_input"] = "task_embed"
    new_dict["pass_task_to_policy"] = True
    new_dict["pass_latent_to_policy"] = False
    return new_dict

def add_precollect(exp_dict, pre_len=5000):
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = new_dict["exp_label"] + "_pre" + str(int(pre_len))
    new_dict["precollect_len"] = pre_len
    return new_dict

def add_name(exp_dict, add_str):
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = new_dict["exp_label"] + "_" + add_str
    return new_dict

def set_latents(exp_dict, sz):
    new_dict = exp_dict.copy()
    new_dict["policy_latent_embedding_dim"] = sz
    new_dict["latent_dim"] = sz//2
    new_dict["exp_label"] = new_dict["exp_label"] + "_Lsz" + str(sz)
    return new_dict

def set_ldim(exp_dict, sz):
    new_dict = exp_dict.copy()
    new_dict["latent_dim"] = sz//2
    new_dict["exp_label"] = new_dict["exp_label"] + "_L" + str(sz)
    return new_dict

def set_ent(exp_dict, policy_entropy_coef):
    new_dict = exp_dict.copy()
    new_dict["policy_entropy_coef"] = policy_entropy_coef
    new_dict["exp_label"] = new_dict["exp_label"] + "_e" + str(policy_entropy_coef)
    return new_dict

def set_wavg_temp(exp_dict, wavg_temp):
    new_dict = exp_dict.copy()
    new_dict["wavg_temp"] = wavg_temp
    new_dict["exp_label"] = new_dict["exp_label"] + "_t" + str(wavg_temp)
    return new_dict

def set_softmax_temp(exp_dict, softmax_temp):
    new_dict = exp_dict.copy()
    new_dict["softmax_temp"] = softmax_temp
    new_dict["exp_label"] = new_dict["exp_label"] + "_smaxt" + str(softmax_temp)
    return new_dict

def set_kl_weight(exp_dict, kl_weight):
    new_dict = exp_dict.copy()
    new_dict["kl_weight"] = kl_weight
    new_dict["exp_label"] = new_dict["exp_label"] + "_kl" + str(kl_weight)
    return new_dict

def set_layers_after_rnn(exp_dict, size):
    layer_sizes = {
        "S": "24 25", # This is smaller than default (no layers after RNN) since RNN size is generally 256
    }
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = new_dict["exp_label"] + "_" + size
    new_dict["encoder_layers_after_gru"] = layer_sizes[size]
    return new_dict

def invariant_init(exp_dict, adjust_weights):
    new_dict = exp_dict.copy()
    new_dict["invariant_init"] = True
    new_dict["invariant_init_adjusts_weights"] = adjust_weights
    new_dict["exp_label"] = new_dict["exp_label"] + "_InvInit"
    if adjust_weights:
        new_dict["exp_label"] = new_dict["exp_label"] + "_adjustWeights"
    return new_dict

def make_pol_size(size, exp_dict):
    hn_pol_sizes = {
        "XS": (64, "64 32"),
        "S": (128, "64 64"),
        "M": (128, "128 64"),
        "L": (256, "128 128"),
        "XL": (256, "256 128"), # Default
        "XXL": (1024, "512 512")
    }
    assert size in hn_pol_sizes, "Given policy size {} not in defined set {}".format(size, list(hn_pol_sizes.keys()))
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = exp_dict["exp_label"] + "_pol_" + size
    new_dict["policy_state_embedding_dim"] = hn_pol_sizes[size][0]
    new_dict["policy_layers"] = hn_pol_sizes[size][1]
    return new_dict

def make_hn_size(size, exp_dict):
    hn_sizes = {
        "S": "16",
        "M": "32"
    }
    assert size in hn_sizes, "Given hypernetwork size {} not in defined set {}".format(size, list(hn_sizes.keys()))
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = exp_dict["exp_label"] + "_hyper_" + size
    new_dict["hypernet_layers"] = hn_sizes[size]
    return new_dict

# a function to perform quick check to make sure experiment labels above are unique within a given set (otherwise future plotting may be messed up)
# and that experiment sets have unique names
def check_exps(exp_sets):
    set_names_so_far = set()
    for experiment_set in exp_sets:
        # check for duplicate set names
        set_name = experiment_set["set_name"]
        assert not set_name in set_names_so_far, "duplicate set name: {}".format(set_name)
        # check for duplicates within set
        set_names_so_far.add(set_name)
        experiment_labels = [exp_dict["exp_label"] for exp_dict in experiment_set["experiments"]]
        labels_so_far = set()
        for l in experiment_labels:
            assert not l in labels_so_far, "exp_label duplicate found for {} in set {}".format(l, experiment_set["set_name"])
            labels_so_far.add(l)

def fixed(exp_dict):
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = new_dict["exp_label"] + "_Fixed"
    return new_dict

def set_frames(exp_dict, num_frames):
    new_dict = exp_dict.copy()
    new_dict["num_frames"] = int(num_frames)
    new_dict["exp_label"] = new_dict["exp_label"] + "_" + format(num_frames, '.0e') + "frames"
    return new_dict

def state_sz(exp_dict, sz):
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = new_dict["exp_label"] + "_STsz"+str(sz)
    if sz is None:
        new_dict["pass_state_to_policy"] = False
        new_dict["policy_state_embedding_dim"] = None
    else:
        new_dict["pass_state_to_policy"] = True
        new_dict["policy_state_embedding_dim"] = sz
    return new_dict

def mem_sz(exp_dict, sz):
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = new_dict["exp_label"] + "_memSZ"+str(sz)
    new_dict["encoder_gru_hidden_size"] = sz
    return new_dict

def layers_after_enc(exp_dict, layers_str):
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = new_dict["exp_label"] + "_afterEncL"+str(layers_str)
    new_dict["encoder_layers_after_gru"] = layers_str
    return new_dict

def make_RL2(exp_dict): # useful to turn VariBAD into RL2 with fixed LR. Note: still variational
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = "RL2_" + exp_dict["exp_label"]
    new_dict["disable_decoder"] = True 
    new_dict["rlloss_through_encoder"] = True
    new_dict["norm_latent_for_policy"] = False
    return new_dict

def make_EtoE_VI(exp_dict): # makes method variational, but end-to-end
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = exp_dict["exp_label"] + "_VIE2E"
    new_dict["disable_stochasticity_in_latent"] = False
    new_dict["sample_embeddings"] = True
    new_dict["precollect_len"] = 0
    new_dict["pretrain_len"] = 0
    new_dict["disable_decoder"] = True 
    new_dict["rlloss_through_encoder"] = True
    new_dict["norm_latent_for_policy"] = False
    new_dict["decode_reward"] = False
    new_dict["decode_state"] = False
    new_dict["decode_task"] = False
    return new_dict

def make_lr_decay(exp_dict): # useful to turn VariBAD into RL2 with fixed LR. Note: still variational
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = exp_dict["exp_label"] + "_lr_decay"
    new_dict["policy_anneal_lr"] = True 
    return new_dict

def make_metashare_coeff(exp_dict, coeff):
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = exp_dict["exp_label"] + "_MS" + format(coeff, '.0e')
    new_dict["metashare_loss_coeff"] = coeff 
    return new_dict

def make_metaWD_coeff(exp_dict, coeff):
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = exp_dict["exp_label"] + "_MWD" + format(coeff, '.0e')
    new_dict["meta_wd_coeff"] = coeff 
    return new_dict

def make_metaWConst_coeff(exp_dict, coeff, c):
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = exp_dict["exp_label"] + "_MC_Coeff" + format(coeff, '.0e') + "_Const" + format(c, '.0e')
    new_dict["meta_wconst_coeff"] = coeff 
    new_dict["meta_wconst"] = c 
    return new_dict

def set_name(exp_dict, set_str):
    new_dict = exp_dict.copy()
    new_dict["exp_label"] = set_str
    return new_dict

# Define dictionaries for models:

VARIBAD = {"exp_label": "Varibad",
            "disable_decoder": False,
            "rlloss_through_encoder": False,
            "norm_latent_for_policy": True}
RL2 = {"exp_label": "RL2",
        "disable_decoder": True,
        "norm_latent_for_policy": False,
        "rlloss_through_encoder": True,
        "lr_policy": 0.0007,
        "lr_vae": 0.0007,
        }

FiLM  = {"exp_label": "FiLM",   "use_film": True, "use_hypernet": True, "hypernet_head_bias": True,  "policy_layers_hyper": "1 1 1", "init_hyper_for_policy": False}
FiLM_init_b  = {"exp_label": "FiLM_init_b",   "use_film": True, "use_hypernet": True, "hypernet_head_bias": True,  "policy_layers_hyper": "1 1 1", "init_hyper_for_policy": True, "init_adjust_weights": False, "init_adjust_bias":True}

HyperNet_init_wb      = {"exp_label": "HyperNet_init_wb", "use_hypernet": True, "hypernet_head_bias": True,  "policy_layers_hyper": "1 1 1", "init_adjust_weights": True,  "init_adjust_bias":True}
HyperNet_init_w      = {"exp_label": "HyperNet_init_w", "use_hypernet": True, "hypernet_head_bias": True,  "policy_layers_hyper": "1 1 1", "init_adjust_weights": True,  "init_adjust_bias":False}
HyperNet_init_b       = {"exp_label": "HyperNet_init_b",  "use_hypernet": True, "hypernet_head_bias": True,  "policy_layers_hyper": "1 1 1", "init_adjust_weights": False, "init_adjust_bias":True}
HyperNet_no_init      = {"exp_label": "HyperNet_no_init", "use_hypernet": True, "hypernet_head_bias": True,  "policy_layers_hyper": "1 1 1", "init_hyper_for_policy": False}
HyperNet_init_kaiming      = {"exp_label": "HyperNet_kaiming", "use_hypernet": True, "hypernet_head_bias": True,  "policy_layers_hyper": "1 1 1", "init_hyper_for_policy": False, "policy_initialisation": "kaiming"}
HyperNet_init_normc      = {"exp_label": "HyperNet_normc", "use_hypernet": True, "hypernet_head_bias": True,  "policy_layers_hyper": "1 1 1", "init_hyper_for_policy": False, "policy_initialisation": "normc"}
HyperNet_init_ortho      = {"exp_label": "HyperNet_ortho", "use_hypernet": True, "hypernet_head_bias": True,  "policy_layers_hyper": "1 1 1", "init_hyper_for_policy": False, "policy_initialisation": "orthogonal"}
HyperNet_init_HFI       = {"exp_label": "HyperNet_init_HFI",  "use_hypernet": True, "hypernet_head_bias": True,  "policy_layers_hyper": "1 1 1", "policy_initialisation": "hfi"}

RL2_HyperNet_init_b = {"exp_label": "RL2_HyperNet_init_b",
        "disable_decoder": True,
        "norm_latent_for_policy": False,
        "rlloss_through_encoder": True,
        "use_hypernet": True, 
        "hypernet_head_bias": True,  
        "policy_layers_hyper": "1 1 1", 
        "init_adjust_weights": False, 
        "init_adjust_bias":True,
        }



# Adjust models (using functions above) to test initializations, set appropriate sizes, and test task inference and end-to-end supervision:

VI = set_name(make_pol_size("XL", VARIBAD), "VI") # Varibad baseline
VI_HN = set_name(make_pol_size("XL", make_varibad(HyperNet_init_b)), "VI+HN")
VI_HN_noInit = set_name(make_pol_size("XL", make_varibad(HyperNet_no_init)), "VI_HN_noInit")

TI_Naive = set_name(ti_naive(make_pol_size("XL", VARIBAD), 1., None), "TI_Naive")

# Note: On Gridworld, using state_sz() to set state to 25 worked best for RNN+S and RNN+HN. Comparisons between these models are reported with size 25 on Grid.
RNN = set_name(state_sz(make_pol_size("XL", make_nonVar(RL2)), None), "RNN") # RNN baseline
RNN_S = set_name(make_pol_size("XL", make_nonVar(RL2)), "RNN+S")
RNN_HN = set_name(make_pol_size("XL", make_nonVar(RL2_HyperNet_init_b)), "RNN+HN")
RNN_HN_Kaiming = set_name(make_pol_size("XL", make_nonVar(make_RL2(HyperNet_init_kaiming))), "RNN+HN_Kaiming")

FiLM_Normc = set_name(make_pol_size("XL", FiLM), "FiLM_Normc")
FiLM_Bias_HyperInit = set_name(make_pol_size("XL", FiLM_init_b), "FiLM_Bias-HyperInit")

# Note: As noted in the appendix, on Gridworld, the TI model only uses 100 updates to pretrain task embeddings
#       Models with "_Grid" have this number adjusted.
TI = set_name(ti(num_full_warmups(full_task_chance(make_pol_size("XL", VARIBAD), 0), 563), 1., base_net=False, stopgrad=True, hard=True), "TI")
TI_Grid = set_name(ti(num_full_warmups(full_task_chance(make_pol_size("XL", VARIBAD), 0), 100), 1., base_net=False, stopgrad=True, hard=True), "TI_g")

TI_pp = set_name(ti(num_full_warmups(full_task_chance(make_pol_size("XL", VARIBAD), 0), 563), 1., base_net=False, stopgrad=True, hard=False), "TI++")
TI_pp_Grid = set_name(ti(num_full_warmups(full_task_chance(make_pol_size("XL", VARIBAD), 0), 100), 1., base_net=False, stopgrad=True, hard=False), "TI++_g")

TI_pp_HN = set_name(ti(num_full_warmups(full_task_chance(make_pol_size("XL", make_varibad(HyperNet_init_b)), 0), 563), 1., base_net=False, stopgrad=True, hard=False), "TI++HN")
TI_pp_HN_Grid = set_name(ti(num_full_warmups(full_task_chance(make_pol_size("XL", make_varibad(HyperNet_init_b)), 0), 100), 1., base_net=False, stopgrad=True, hard=False), "TI++HN_g")

TI_p_HN = set_name(ti(num_full_warmups(full_task_chance(make_pol_size("XL", make_varibad(HyperNet_init_b)), 0), 563), 1., base_net=False, stopgrad=True, hard=True), "TI+HN")
TI_p_HN_Grid = set_name(ti(num_full_warmups(full_task_chance(make_pol_size("XL", make_varibad(HyperNet_init_b)), 0), 100), 1., base_net=False, stopgrad=True, hard=True), "TI+HN_g")

BI_pp_HN = set_name(ti(num_full_warmups(full_task_chance(make_pol_size("XL", make_varibad(HyperNet_init_b)), 0), 563), 1., base_net=True, stopgrad=True, hard=False), "BI++HN")

Multi_HN = set_name(make_multi(make_pol_size("XL", make_varibad(HyperNet_init_b))), "Multi+HN") # multi-task upper bound
# Note: You can use vae_lr() to adjust the LR of VAEs in models with VAEs to save them with different names.




# Adjust models using functions to test different sequence models, for the SplAgger paper (Beck et al., 2024):

# Make sure the default RNN model has no pre-training and no pre-collect phase.
# Note that in the context of the SplAgger paper, this default RNN_HN model is referred to as just RNN.
RNN_HN = set_name(make_nonVarNPP(RNN_HN), "RNN+HN")

SplAgger = set_name(make_skip(make_max(RNN_HN)), "SplAgger")

Max = set_name(noRNN(make_max(RNN_HN)), "Max") # Note: This also has no skip connection; skip makes little sense without RNN
Max_ST = set_name(noRNN(make_st(make_max(RNN_HN))), "Max-ST")

# AMRL aggregation as in Beck et al., 2020
AMRL = set_name(make_max(make_st(make_skip(RNN_HN))), "AMRL")
AMRL_noRNN = set_name(noRNN(make_max(make_st(RNN_HN))), "AMRL-noRNN")

# Avg aggregation as in a Conditional Neural Process (Garnelo et al., 2018)
CNP = set_name(noRNN(make_avg(RNN_HN)), "CNP+HN") # Note: Variational Inference Required for non-conditional NP

# LSTM with invariant initialization
LSTM_InvInit = set_name(invariant_init(lstm(RNN_HN), adjust_weights=True), "InvInit")
# Transformer. Note: when using this model, set "policy_num_steps": 175 in the shared arguments for the experiment set
Transformer = set_name(make_transformer(RNN_HN, token_sz=16, mem_sz=None), "Transformer")

# Gaussian Aggregation, as in PEARL (Rakelly et al., 2019). Variational Inference Required.
PEARL = set_name(set_kl_weight(make_EtoE_VI(make_gauss(VI_HN)), .000001), "PEARL")
PEARL_vari = set_name(make_gauss(VI_HN), "PEARL-vari")
# Tuning of PEARL KL:
PEARL_kl1 = set_name(set_kl_weight(make_EtoE_VI(PEARL), 1), "PEARL-kl1")
PEARL_klp01 = set_name(set_kl_weight(make_EtoE_VI(PEARL), .01), "PEARL-klp01")
PEARL_klp0001 = set_name(set_kl_weight(make_EtoE_VI(PEARL), .0001), "PEARL-klp0001")
PEARL_klp000001 = set_name(set_kl_weight(make_EtoE_VI(PEARL), .000001), "PEARL-klp000001")
PEARL_kl0 = set_name(set_kl_weight(make_EtoE_VI(PEARL), 0), "PEARL-kl0")
# Note vis_10(PEARL) was used to save the visualization more frequently

# SplAgger with different binary operators:
SplAgger_avg = set_name(make_skip(make_avg(RNN_HN)), "SplAgger-avg")
SplAgger_avgmax = set_name(make_skip(make_am(RNN_HN)), "SplAgger-avgmax")
SplAgger_softmax = set_name(make_skip(set_softmax_temp(make_softmax(RNN_HN), .1)), "SplAgger-softmax")
SplAgger_wsoftmax = set_name(make_skip(set_wavg_temp(make_wavg(RNN_HN), .1)), "SplAgger-wsoftmax")
SplAgger_wavg = set_name(make_skip(make_wavg(RNN_HN)), "SplAgger-wavg")
# Tuning of temperature:
wsoftmax_t1 = set_name(set_wavg_temp(noRNN(make_wavg(RNN_HN)), 1), "wsoftmax-t1")
wsoftmax_tp1 = set_name(set_wavg_temp(noRNN(make_wavg(RNN_HN)), .1), "wsoftmax-tp1")
SOFTMAX_t1 = set_name(noRNN(set_softmax_temp(make_softmax(RNN_HN), 1)), "softmax-t1")
SOFTMAX_tp1 = set_name(noRNN(set_softmax_temp(make_softmax(RNN_HN), .1)), "softmax-tp1")
# More SplAgger variants
SplAgger_noRNN = set_name(noRNN(make_max(RNN_HN)), "SplAgger-noRNN")
SplAgger_noRNN_avg = set_name(noRNN(make_avg(RNN_HN)), "SplAgger-noRNN-avg")
SplAgger_noSplit = set_name(make_max(RNN_HN), "SplAgger-noSplit")
