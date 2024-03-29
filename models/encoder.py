import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import helpers as utl
from models.aggregator import Aggregator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RNNEncoder(nn.Module):
    def __init__(self,
                 args,
                 ):
        super(RNNEncoder, self).__init__()

        self.args = args
        self.layers_before_gru=self.args.encoder_layers_before_gru
        self.hidden_size=self.args.encoder_gru_hidden_size
        self.layers_after_gru=self.args.encoder_layers_after_gru
        self.layers_after_agg=self.args.encoder_layers_after_agg
        self.encoder_type=self.args.encoder_enc_type
        self.skip_type=self.args.encoder_skip_type
        self.agg_type=self.args.encoder_agg_type
        self.st_estimator=self.args.encoder_st_estimator
        self.max_init_low=self.args.encoder_max_init_low
        self.latent_dim=self.args.latent_dim
        self.action_dim=self.args.action_dim
        self.action_embed_dim=self.args.action_embedding_size
        self.state_dim=self.args.state_dim
        self.state_embed_dim=self.args.state_embedding_size
        self.reward_embed_size=self.args.reward_embedding_size

        self.reparameterise = self._sample_gaussian

        # embed action, state, reward
        self.state_encoder = utl.FeatureExtractor(self.state_dim, self.state_embed_dim, F.relu)
        self.action_encoder = utl.FeatureExtractor(self.action_dim, self.action_embed_dim, F.relu)
        self.reward_encoder = utl.FeatureExtractor(1, self.reward_embed_size, F.relu)

        if self.args.full_transitions:
            self.agg_input_dim = self.action_embed_dim + (2 * self.state_embed_dim) + self.reward_embed_size
        else:
            self.agg_input_dim = self.action_embed_dim + self.state_embed_dim + self.reward_embed_size

        self.agg = Aggregator(args, self.agg_input_dim, self.hidden_size, self.latent_dim, self.state_embed_dim, 
            skip_type=self.skip_type, encoder_type=self.encoder_type, agg_type=self.agg_type, st_estimator=self.st_estimator, 
            max_init_low=self.max_init_low, layers_before_enc=self.layers_before_gru, layers_after_enc=self.layers_after_gru, 
            layers_after_agg=self.layers_after_agg)


    def _sample_gaussian(self, mu, logvar, num=None):
        if num is None:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            raise NotImplementedError  # TODO: double check this code, maybe we should use .unsqueeze(0).expand((num, *logvar.shape))
            std = torch.exp(0.5 * logvar).repeat(num, 1)
            eps = torch.randn_like(std)
            mu = mu.repeat(num, 1)
            return eps.mul(std).add_(mu)

    def reset_hidden(self, hidden_state, done):
        """ Reset the hidden state where the BAMDP was done (i.e., we get a new task) """
        if hidden_state.dim() != done.dim():
            if done.dim() == 2:
                done = done.unsqueeze(0)
            elif done.dim() == 1:
                done = done.unsqueeze(0).unsqueeze(2)
        if len(hidden_state.shape) == 2:
            batch_size = hidden_state.shape[0]
            reset_state = self.agg.init_state(batch_size).squeeze(0)
        else:
            assert len(hidden_state.shape) == 3, hidden_state.shape
            batch_size = hidden_state.shape[1]
            reset_state = self.agg.init_state(batch_size)
        hidden_state = hidden_state*(1 - done) + reset_state*(done)
        return hidden_state

    def prior(self, batch_size, sample=True):

        # TODO: add option to incorporate the initial state (does this mean initial obs/inputs?)

        hidden_state = self.agg.init_state(batch_size) # init aggregator state
        agg_out, _ = self.agg(torch.zeros((1,batch_size,self.agg_input_dim)).to(device), hidden_state) # TODO: Should we replace the hidden state or start again at 0?
        latent_mean, latent_logvar = torch.chunk(agg_out, 2, dim=-1) 

        if sample:
            latent_sample = self.reparameterise(latent_mean, latent_logvar)
        else:
            latent_sample = latent_mean

        return latent_sample, latent_mean, latent_logvar, hidden_state

    def forward(self, actions, states, rewards, prev_states, hidden_state, return_prior, sample=True, detach_every=None, unpadded_lens=None, return_all_hidden=True):
        """
        Actions, states, rewards should be given in form [sequence_len * batch_size * dim].
        For one-step predictions, sequence_len=1 and hidden_state!=None.
        For feeding in entire trajectories, sequence_len>1 and hidden_state=None.
        In the latter case, we return embeddings of length sequence_len+1 since they include the prior.
        If unpadded_lens is not None, return hidden state at time t in each batch, for each t in unpadded_lens
        If not return_all_hidden, just return final hidden state
        """

        # shape should be: sequence_len x batch_size x hidden_size
        actions = actions.reshape((-1, *actions.shape[-2:]))
        states = states.reshape((-1, *states.shape[-2:]))
        prev_states = prev_states.reshape((-1, *prev_states.shape[-2:])).to(device)
        rewards = rewards.reshape((-1, *rewards.shape[-2:]))
        if hidden_state is None:
            assert return_prior
        else:
            assert hidden_state.shape[-1] == self.agg.state_size, (hidden_state.shape, self.agg.state_size, self.agg.enc_state_size)
            # if the sequence_len is one, this will add a dimension at dim 0 (otherwise will be the same)
            hidden_state = hidden_state.reshape((-1, *hidden_state.shape[-2:]))

        if return_prior:
            # if hidden state is none, start with the prior
            assert hidden_state is None
            _, prior_mean, prior_logvar, prior_hidden_state = self.prior(actions.shape[1])
            hidden_state = prior_hidden_state

        # extract features for actions, rewards, next_states 
        # Note: This order is required by aggregator file
        ha = self.action_encoder(actions)
        hr = self.reward_encoder(rewards)
        hs_next = self.state_encoder(states) # these are next states, after actions and rewards
        hs_prev = self.state_encoder(prev_states) # these are previous states, before actions and rewards
        # print("ha", ha)
        # print("hr", hr)
        # print("hs", hs_next)
        if self.args.full_transitions:
            h = torch.cat((hs_prev, ha, hr, hs_next), dim=2)
        else:
            h = torch.cat((ha, hr, hs_next), dim=2)

        agg_outputs, hidden_states = None, None # compute these1 step at a time to save on memory)
        if unpadded_lens is None:
            # # Alternatively, letting self.agg run all steps at once may be slightly faster, but need to modify somehow to deal with detach:
            # agg_outputs, hidden_states = self.agg(h, hidden_state.clone(), return_all_hidden_states=return_all_hidden)
            # if return_prior:
            #     prior_agg_out = torch.cat((prior_mean, prior_logvar), dim=-1)
            #     agg_outputs = torch.cat((prior_agg_out, agg_outputs), dim=0)
            #     if return_all_hidden:
            #         hidden_states = torch.cat((prior_hidden_state, hidden_states), dim=0)
            # assert detach_every is None, "Only full tbptt is supported now with args.tbptt_stepsize is None, for speed of above code. To truncate, use code below."
            seq_len = actions.shape[0]
            if seq_len != 1 and return_all_hidden:
                print("Warning: seq_len={}; requesting all hidden states for long sequences may cause memory issues.".format(seq_len))
            if return_prior:
                agg_outputs = [torch.cat((prior_mean, prior_logvar), dim=-1)]
                hidden_states = [prior_hidden_state] if return_all_hidden else []
            else:
                agg_outputs, hidden_states = [], []
            for t in range(seq_len):
                to_agg = h[t:t+1, :, :] # input to aggregator at time t, leaving time dimension
                agg_out, hidden_state = self.agg(to_agg, hidden_state.clone(), return_all_hidden_states=False)
                agg_outputs.append(agg_out)
                if return_all_hidden or t == seq_len-1:
                    hidden_states.append(hidden_state)
                if detach_every and (t%detach_every==0) and t!=0:
                    hidden_state = hidden_state.detach()
            agg_outputs = torch.cat(agg_outputs, dim=0)
            hidden_states = torch.cat(hidden_states, dim=0)
        else:
            assert return_prior
            assert return_all_hidden
            unpadded_lens = [l-1 for l in unpadded_lens] # make a new copy where -1 is prior and 0 is first output
            assert len(unpadded_lens) == states.shape[1] # make sure batch sizes agree
            seq_len, batch_size = actions.shape[0], actions.shape[1]
            agg_outputs, hidden_states = [None for _ in range(batch_size)], [None for _ in range(batch_size)]
            for t in range(-1, seq_len, 1): # -1 is prior
                if t == -1:
                    agg_out, hidden_state = torch.cat((prior_mean, prior_logvar), dim=-1), hidden_state
                else:
                    to_agg = h[t:t+1, :, :] # input to aggregator at time t, leaving time dimension
                    agg_out, hidden_state = self.agg(to_agg, hidden_state.clone(), return_all_hidden_states=False)
                if detach_every and (t%detach_every==0) and t!=0 and t!=-1:
                    hidden_state = hidden_state.detach()
                for b in range(batch_size):
                    if unpadded_lens[b] == t: # save hidden state at end of padding
                        agg_outputs[b] = agg_out[:,b,:]
                        hidden_states[b] = hidden_state[:,b,:]
            agg_outputs = torch.stack(agg_outputs, dim=1)
            hidden_states = torch.stack(hidden_states, dim=1)
            assert agg_outputs.shape[:2] == (1, batch_size), (agg_outputs.shape, (1, batch_size))
            assert hidden_states.shape[:2] == (1, batch_size), (agg_outputs.shape, (1, batch_size))

        latent_mean, latent_logvar = torch.chunk(agg_outputs, 2, dim=-1)

        if sample:
            latent_sample = self.reparameterise(latent_mean, latent_logvar)
        else:
            latent_sample = latent_mean

        if latent_mean.shape[0] == 1: # batch size 1, get rid of batch dim
            latent_sample, latent_mean, latent_logvar = latent_sample[0], latent_mean[0], latent_logvar[0]

        return latent_sample, latent_mean, latent_logvar, hidden_states
