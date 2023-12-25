import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def none_or_zeros(tensor):
  return None if tensor is None else torch.zeros_like(tensor)

def normalize(tensor, eps=0.00000001):
  norm = torch.norm(tensor)
  return tensor / torch.max(norm, eps*torch.ones_like(norm))

def get_mean_grad_norm(to_diff, input_batch):
  dv_di_batch = torch.autograd.grad(to_diff, input_batch, create_graph=False, allow_unused=False, retain_graph=True)[0]
  return torch.norm(dv_di_batch, dim=-1).mean()

def get_samples_by_task(sample):
  state_batch, belief_batch, task_batch, \
  actions_batch, latent_sample_batch, latent_mean_batch, latent_logvar_batch, value_preds_batch, \
  return_batch, old_action_log_probs_batch, adv_targ = sample

  # assign task id to each item in batch based on which distinct task was used
  task_id_for_item = []
  distinct_tasks_so_far = []
  assert task_batch is not None
  for i in range(len(task_batch)):
    task = task_batch[i]
    # see if task matches previous task id
    task_id = None
    for i, prev_task in enumerate(distinct_tasks_so_far):
      if torch.allclose(task, prev_task):
        task_id = i
        break
    # no match, so assign new id
    if task_id is None:
      distinct_tasks_so_far.append(task)
      task_id = len(distinct_tasks_so_far)-1
    task_id_for_item.append(task_id)
  assert len(task_id_for_item) == len(task_batch), (len(task_id_for_item), len(task_batch))

  # create new samples split up by task_id
  samples = []
  task_ids = range(len(distinct_tasks_so_far))
  for task_id in task_ids:
    bool_mask = [(item_id == task_id) for item_id in task_id_for_item] # find all elements in sample for this task_id=specific sample
    samples.append(tuple((None if tensor is None else tensor[bool_mask]) for tensor in sample))
  assert len(samples) == len(task_ids), (len(samples), len(task_ids))

  return samples

class PPO:
    def __init__(self,
                 args,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 policy_optimiser,
                 policy_anneal_lr,
                 train_steps,
                 optimiser_vae=None,
                 lr=None,
                 clip_param=0.2,
                 ppo_epoch=5,
                 num_mini_batch=5,
                 eps=None,
                 use_huber_loss=True,
                 use_clipped_value_loss=True,
                 reset_parameters_ac=None,
                 ):
        self.args = args

        # the model
        self.actor_critic = actor_critic
        self.reset_parameters_ac = reset_parameters_ac # A second set of initial parameters used if args.hyper_onehot_hard_reset

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.update_index = 0

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.use_clipped_value_loss = use_clipped_value_loss
        self.use_huber_loss = use_huber_loss

        # optimizer
        if policy_optimiser == 'adam':
            self.optimiser = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        elif policy_optimiser == 'rmsprop':
            self.optimiser = optim.RMSprop(actor_critic.parameters(), lr=lr, eps=eps, alpha=0.99)
        self.lr_scheduler_policy = None
        self.lr_scheduler_encoder = None
        self.train_steps = train_steps
        self.policy_anneal_lr = policy_anneal_lr
        if self.policy_anneal_lr:
            lam = lambda f: 1 - f / train_steps
            self.lr_scheduler_policy = optim.lr_scheduler.LambdaLR(self.optimiser, lr_lambda=lam)
        # vae optimizer
        if optimiser_vae is not None:
            self.set_vae_optimizer(optimiser_vae) # can also set later


    def set_vae_optimizer(self, optimiser_vae):
      self.optimiser_vae = optimiser_vae
      if self.policy_anneal_lr and hasattr(self.args, 'rlloss_through_encoder') and self.args.rlloss_through_encoder:
        lam = lambda f: 1 - f / self.train_steps
        self.lr_scheduler_encoder = optim.lr_scheduler.LambdaLR(self.optimiser_vae, lr_lambda=lam)


    def update(self,
               policy_storage,
               encoder=None,  # variBAD encoder
               rlloss_through_encoder=False,  # whether or not to backprop RL loss through encoder
               compute_vae_loss=None  # function that can compute the VAE loss
               ):

        # if doing multi-task pre-training for hypernet, with a reset of params, then no reason to train some components, e.g. VAE
        hyper_reset = False
        ti_reset = False
        warming_up = False
        freeze_vae_during_warmup = False
        num_warmups = self.args.task_num_warmup if self.args.task_num_warmup is not None else self.args.hyper_onehot_num_warmup
        if num_warmups is not None:
          warming_up = self.update_index < num_warmups
          hyper_reset = self.args.hyper_onehot_hard_reset and (self.args.hyper_onehot_num_warmup is not None)
          ti_reset = self.args.ti_coeff is not None and self.args.ti_hard_reset
          freeze_vae_during_warmup = (hyper_reset or ti_reset or self.args.freeze_vae_during_warmup) and warming_up


        # -- get action values --
        advantages = policy_storage.returns[:-1] - policy_storage.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        # if this is true, we will update the VAE at every PPO update
        # otherwise, we update it after we update the policy
        if rlloss_through_encoder and not freeze_vae_during_warmup:
            # recompute embeddings (to build computation graph)
            utl.recompute_embeddings(policy_storage, encoder, sample=False, update_idx=0,
                                     detach_every=self.args.tbptt_stepsize if hasattr(self.args,
                                                                                      'tbptt_stepsize') else None)

        # update the normalisation parameters of policy inputs before updating
        self.actor_critic.update_rms(args=self.args, policy_storage=policy_storage)

        self.encoder = encoder # This can be changed

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        loss_epoch = 0
        state_grad_norm, task_grad_norm, latent_grad_norm, state_norm, latent_norm = 0, 0, 0, 0, 0
        for e in range(self.ppo_epoch):

            data_generator = policy_storage.feed_forward_generator(advantages, self.num_mini_batch)
            for sample in data_generator:

                loss, value_loss, action_loss, dist_entropy, values, state_batch, task_batch, latent_batch = self.get_loss(sample, rlloss_through_encoder, compute_vae_loss, return_all_losses=True)

                # Compute gradient info for logging
                if self.args.compute_grad_info: 
                  # Compute some gradient norms for logging
                  if self.args.pass_state_to_policy:
                    state_grad_norm += get_mean_grad_norm(values.sum(), state_batch)
                    state_norm += torch.norm(state_batch, dim=-1).mean()
                  if task_batch is not None:
                    task_grad_norm += get_mean_grad_norm(values.sum(), task_batch)
                  if latent_batch is not None:
                    latent_grad_norm += get_mean_grad_norm(values.sum(), latent_batch)
                    latent_norm += torch.norm(latent_batch, dim=-1).mean()
                else:
                  state_batch, task_batch, latent_batch = None, None, None

                # compute gradients (will attach to all networks involved in this computation)
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.args.policy_max_grad_norm)
                if (encoder is not None) and rlloss_through_encoder and not freeze_vae_during_warmup:
                    nn.utils.clip_grad_norm_(encoder.parameters(), self.args.policy_max_grad_norm)

                # update
                self.optimiser.step()
                if rlloss_through_encoder and not freeze_vae_during_warmup:
                    self.optimiser_vae.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                loss_epoch += loss.item()

                if rlloss_through_encoder and not freeze_vae_during_warmup:
                    # recompute embeddings (to build computation graph)
                    utl.recompute_embeddings(policy_storage, encoder, sample=False, update_idx=e + 1,
                                             detach_every=self.args.tbptt_stepsize if hasattr(self.args,
                                                                                              'tbptt_stepsize') else None)

        if (not rlloss_through_encoder) and (self.optimiser_vae is not None) and not freeze_vae_during_warmup:
            # Note: if rlloss_through_encoder, vae loss is already computed: it is computed above in self.get_loss
            if self.args.ti_coeff is not None and self.args.ti_target == "base_net" and not self.args.ti_target_stop_grad:
              # pass policy optimizer to vae if using hypernetwork with base_net target and no stop grad
              policy_opt_to_vae = self.optimiser
            else:
              policy_opt_to_vae = None
            for _ in range(self.args.num_vae_updates):
                compute_vae_loss(update=True, encoded_task_func=self.get_encoded_task_func(), policy_optimiser=policy_opt_to_vae)

        if self.lr_scheduler_policy is not None:
            self.lr_scheduler_policy.step()
        if self.lr_scheduler_encoder is not None and not freeze_vae_during_warmup:
            self.lr_scheduler_encoder.step()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        loss_epoch /= num_updates
        #
        state_grad_norm /= num_updates
        task_grad_norm /= num_updates
        latent_grad_norm /= num_updates
        state_norm /= num_updates
        latent_norm /= num_updates
        
        self.update_index += 1

        # reset correct params after warm up
        if warming_up and (self.update_index == num_warmups): # just finished warm up
          if hyper_reset:
            self.actor_critic.reset_hyper_parameters(self.reset_parameters_ac)
          if ti_reset:
            self.actor_critic.reset_nontask_parameters(self.reset_parameters_ac)

        return (value_loss_epoch, action_loss_epoch, dist_entropy_epoch, loss_epoch, \
          state_grad_norm, task_grad_norm, latent_grad_norm, state_norm, latent_norm)

    def get_encoded_task_func(self):
      encoded_task_func = None
      if self.args.ti_coeff is not None: # pass encoded_task_func if doing task inference
        assert self.args.decode_task, "When using task inference with learned embedding (args.ti_coeff) and vae, please set args.decode_task to True"
        if self.args.ti_target == "task":
          def task_target_encoding(task_to_encode):
            state_shape = task_to_encode.shape[:-1] + (self.actor_critic.dim_state,)
            _, _, _, encoded_task_batch, _, _ = self.actor_critic.get_encoded_inputs(torch.zeros(state_shape).to(device), None, None, task_to_encode) # Note state should really be ignored here
            return encoded_task_batch
          encoded_task_func = task_target_encoding
        else:
          assert self.args.ti_target == "base_net"
          encoded_task_func = self.actor_critic.task_bn_params
      return encoded_task_func

    def act(self, state, latent, belief, task, deterministic=False, training=False):
        return self.actor_critic.act(state=state, latent=latent, belief=belief, task=task, deterministic=deterministic, training=training, update_num=self.update_index)

    def get_loss(self, sample, rlloss_through_encoder, compute_vae_loss, return_all_losses=False, include_ti_loss=True):
      state_batch, belief_batch, task_batch, \
      actions_batch, latent_sample_batch, latent_mean_batch, latent_logvar_batch, value_preds_batch, \
      return_batch, old_action_log_probs_batch, adv_targ = sample

      if not rlloss_through_encoder:
          state_batch = state_batch.detach()
          if latent_sample_batch is not None:
              latent_sample_batch = latent_sample_batch.detach()
              latent_mean_batch = latent_mean_batch.detach()
              latent_logvar_batch = latent_logvar_batch.detach()

      latent_batch = utl.get_latent_for_policy(args=self.args, latent_sample=latent_sample_batch,
                                               latent_mean=latent_mean_batch,
                                               latent_logvar=latent_logvar_batch
                                               )

      # set up state and task for grad logging
      if self.args.compute_grad_info:
        state_batch = torch.autograd.Variable(state_batch, requires_grad=True)
        if task_batch is not None:
          task_batch = torch.autograd.Variable(task_batch, requires_grad=True)
        if latent_batch is not None and not rlloss_through_encoder:
          latent_batch = torch.autograd.Variable(latent_batch, requires_grad=True)

      # Reshape to do in a single forward pass for all steps
      values, action_log_probs, dist_entropy, action_mean, action_logstd, info = \
          self.actor_critic.evaluate_actions(state=state_batch, latent=latent_batch,
                                             belief=belief_batch, task=task_batch,
                                             action=actions_batch, return_action_mean=True,
                                             return_info=True, training=True, update_num=self.update_index
                                             )

      ratio = torch.exp(action_log_probs -
                        old_action_log_probs_batch)
      surr1 = ratio * adv_targ
      surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
      action_loss = -torch.min(surr1, surr2).mean()

      if self.use_huber_loss and self.use_clipped_value_loss:
          value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                      self.clip_param)
          value_losses = F.smooth_l1_loss(values, return_batch, reduction='none')
          value_losses_clipped = F.smooth_l1_loss(value_pred_clipped, return_batch, reduction='none')
          value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
      elif self.use_huber_loss:
          value_loss = F.smooth_l1_loss(values, return_batch)
      elif self.use_clipped_value_loss:
          value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                      self.clip_param)
          value_losses = (values - return_batch).pow(2)
          value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
          value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
      else:
          value_loss = 0.5 * (return_batch - values).pow(2).mean()

      # zero out the gradients
      self.optimiser.zero_grad()
      if rlloss_through_encoder:
          self.optimiser_vae.zero_grad()

      # compute policy loss and backprop
      loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef

      # Task Inference Loss (optional)
      if include_ti_loss and (self.args.decode_task or self.args.ti_coeff is not None) and rlloss_through_encoder: # Note: if not rlloss_through_encoder, this loss will be computed in vae_loss() instead
        loss = self.add_ti_loss(loss, state_batch, latent_batch, belief_batch, task_batch, actions_batch)

      # compute vae loss and backprop
      if rlloss_through_encoder:
          loss += self.args.vae_loss_coeff * compute_vae_loss()

      if return_all_losses:
        if "one_hot_task" in info: # return the one-hot version policy used so we can calculate its gradient
          task_batch = info["one_hot_task"]
        return loss, value_loss, action_loss, dist_entropy, values, state_batch, task_batch, latent_batch
      return loss

    def add_ti_loss(self, loss, state_batch, latent_batch, belief_batch, task_batch, actions_batch):
      # add loss for task inference
      assert self.args.ti_target in ["task", "base_net"]  
      if self.args.ti_coeff is not None:
        assert self.args.task_chance is not None, "end-to-end task inference requires task multi-task training."

      coeff = self.args.ti_coeff
      
      if self.args.ti_coeff is None:
        # embedding is not learned end-to-end from multi-task setting
        assert self.args.decode_task
        assert self.args.policy_latent_embedding_dim == task_batch.shape[-1], (self.args.policy_latent_embedding_dim, task_batch.shape[-1])
        coeff = self.args.task_loss_coeff
        _, latent, _, _, _, _ = self.actor_critic.get_encoded_inputs(state_batch, latent_batch, belief_batch, task_batch) 
        task_target = torch.detach(task_batch)
        latent_target = latent
      elif self.args.ti_target == "task":
        state, latent, belief, task, one_hot_task, info = self.actor_critic.get_encoded_inputs(state_batch, latent_batch, belief_batch, task_batch) 
        task_target = task
        latent_target = latent
      else:
        assert self.args.use_hypernet, "inferring base net requires hypernet."
        _, _, _, _, _, base_params_task = \
            self.actor_critic.evaluate_actions(state=state_batch, latent=latent_batch,
                                               belief=belief_batch, task=task_batch,
                                               action=actions_batch, return_action_mean=True,
                                               return_base_params=True, # Need the base net params
                                               return_info=False, training=True, update_num=self.update_index, force_task_input=True, # force task
                                               )
        _, _, _, _, _, base_params_latent = \
            self.actor_critic.evaluate_actions(state=state_batch, latent=latent_batch,
                                               belief=belief_batch, task=task_batch,
                                               action=actions_batch, return_action_mean=True,
                                               return_base_params=True, # Need the base net params
                                               return_info=False, training=True, update_num=self.update_index, force_latent_input=True, # force latent
                                               )
        task_target = torch.cat(base_params_task,-1)
        latent_target = torch.cat(base_params_latent,-1)

      if self.args.ti_target_stop_grad:
        task_target = torch.detach(task_target)
      
      ti_loss = ((task_target-latent_target)**2).mean()
      loss = loss + coeff * ti_loss
      return loss


    def get_grad_with_enc(self, to_diff, create_graph=True):
      # for some reason it does not work to do this: params_to_opt = self.actor_critic.parameters(), then use params_to_opt instead
      # but this does seem to work:
      all_params = list(self.actor_critic.parameters())+list(self.encoder.parameters())
      g = torch.autograd.grad(to_diff, all_params, create_graph=create_graph, allow_unused=True)
      g = torch.cat([(torch.zeros(param.shape) if param_grad is None else param_grad).reshape((-1,)).to(device) for param_grad, param in zip(g, all_params)])
      
      # # confirmed by:
      # ac_g = torch.autograd.grad(to_diff, self.actor_critic.parameters(), create_graph=create_graph, allow_unused=True)
      # ac_g = torch.cat([(torch.zeros(param.shape) if param_grad is None else param_grad).reshape((-1,)).to(device) for param_grad, param in zip(ac_g, self.actor_critic.parameters())])
      # enc_g = torch.autograd.grad(to_diff, self.encoder.parameters(), create_graph=create_graph, allow_unused=True)
      # enc_g = torch.cat([(torch.zeros(param.shape) if param_grad is None else param_grad).reshape((-1,)).to(device) for param_grad, param in zip(enc_g, self.encoder.parameters())])
      # combined_g = torch.cat((ac_g, enc_g))
      # assert torch.allclose(combined_g, g)

      return g

    def get_grad(self, to_diff, create_graph=True):
      # for some reason it does not work to do this: params_to_opt = self.actor_critic.parameters(), then use params_to_opt instead
      g = torch.autograd.grad(to_diff, self.actor_critic.parameters(), create_graph=create_graph, allow_unused=True)
      g = torch.cat([(torch.zeros(param.shape) if param_grad is None else param_grad).reshape((-1,)).to(device) for param_grad, param in zip(g, self.actor_critic.parameters())])
      return g

      return loss
