'''
Generalizes the aggregation 
in PEARL (Paper: https://arxiv.org/abs/1903.08254) 
and AMRL
( Github: https://github.com/jacooba/AMRL-ICLR2020/
  Paper: https://iclr.cc/virtual_2020/poster_Bkl7bREtDr.html )
'''

import numpy as np

import torch
import torch.nn as nn
from torch.nn import MultiheadAttention, LayerNorm
from torch.nn import functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Aggregator(nn.Module):
    def __init__(self, args, input_size, enc_size, latent_dim, s_embed_size,
        skip_type="enc", encoder_type="gru", agg_type="max", st_estimator=True, max_init_low=False,
        layers_before_enc=[], layers_after_enc=[], layers_after_agg=[]):
        """
        Implements aggregation as in AMRL and PEARL.
        Args:
            input_size (int):       Size of input vectors
            enc_size (int):         Size of encoder (e.g. gru) output
            latent_dim (int):       Size of output mean and size of output variance for latent distribution
            s_embed_size (int):     Size of state encoding assumed to be at start of input
            skip_type (str):        What neurons are unaggregated. If None, all neurons are aggregate.
                                        Note: currently neurons from sequential encoder are only option for skip
                                        since the state can always be skipped around the aggregation 
                                        in the config by using --pass_state_to_policy
            encoder_type (str):     Type of sequential encoder (e.g. gru). (Options below.)
            agg_type (str):         Type of aggregation. (Options below.)
            st_estimator (bool):    Whether to use a "straight_through" gradient approximation.
            max_init_low (bool):    Whether the init state of the max aggregator should start at an arbitrarily small constant. 
                                        Note: This will also remove the activation before the max aggregator (if using layers_after_enc),
                                        otherwise, relu will cause this feature to be useless. (Alternatively, use tanh, as the output
                                        of the encoders already do.)
            layers_before_enc:      List of sizes for FC layers before encoder
            layers_after_enc:       List of sizes for FC layers after encoder
            layers_after_agg:       List of sizes for FC layers after aggregator
        """
        super().__init__()
        self.args = args
        self.input_size = input_size
        self.enc_size = enc_size
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type
        self.agg_type = agg_type
        self.skip_type = skip_type
        self.max_init_low = max_init_low
        self.learn_init_state = self.args.learn_init_state
        assert skip_type in ["enc", None], skip_type
        assert encoder_type in ["gru", "lstm", "transition", "transformer", None], encoder_type
        assert agg_type in ["max", "max_norm", "sum", "avg", "am", "weighted_avg", "gauss", "softmax", None], agg_type

        if encoder_type == "transformer":
            assert not self.learn_init_state, "Transformer uses state just to record observations. Trying to learn the state will cause NaNs."

        if self.learn_init_state:
            self.learn_enc_state = True
            self.learn_agg_state = True 
        else:
            self.learn_enc_state = False
            self.learn_agg_state = False 

        if encoder_type == "transition":
            print("Warning: Transition encoder left for backward compatibility. Consider using --full_transitions instead.")
            assert not args.full_transitions, "Cannot have --full_transitions and transition encoder."

        # Combine layers_after_agg and latent dim so that we output 2*latent_dim (mean and variance)
        layers_after_agg.append(2*latent_dim)

        # FC layers before the encoder
        curr_dim = input_size
        if encoder_type in ["gru", "lstm"]:
            self.fc_before_enc = nn.ModuleList([])
            for i in range(len(layers_before_enc)):
                self.fc_before_enc.append(nn.Linear(curr_dim, layers_before_enc[i]))
                curr_dim = layers_before_enc[i]
        else:
            # For no encoder, it is equivalent adding layer before or after,
            # And, the masking (or ar) in TransitionEncoder is incorrect if adding before,
            # Thus we move these layers after...
            self.fc_before_enc = ()
            layers_after_enc = layers_before_enc + layers_after_enc
        self.size_before_enc = curr_dim

        # Special fix for weighted_avg
        if agg_type == "weighted_avg": 
            # Weighted avg reduces number of neurons by half, so multiply input 2x
            # Note: This is fair in terms of activations after aggregation, but not number of parameters
            # Alternatively, remove this line and ensure that size_before_agg%2 == 0 later
            if layers_after_enc:
                layers_after_enc[-1] = 2*layers_after_enc[-1]
            else:
                self.enc_size *= 2
            # Note that this doesn't work with skip connection for now, however. (that skips too many neurons around.)

        # Define (sequential) encoder
        self.enc_state_size = None # Set below.
        if encoder_type == "gru":
            self.enc_state_size = self.enc_size
            self.encoder = nn.GRU(input_size=self.size_before_enc,
                            hidden_size=self.enc_size,
                            num_layers=1,
                            )
            self.init_rnn()
        elif encoder_type == "lstm":
            self.enc_state_size = 2*self.enc_size
            self.encoder = nn.LSTM(input_size=self.size_before_enc,
                            hidden_size=self.enc_size,
                            num_layers=1,
                            )
            self.init_rnn()
        elif encoder_type == "transformer":
            self.encoder = TransformerEncoder(self.size_before_enc, args.transformer_num_layers, 
                args.transformer_num_heads, max_seq_len=args.max_trajectory_len, output_sz=enc_size,
                memory_size=args.transformer_mem_sz, token_size=args.transformer_token_sz)
            self.enc_state_size = self.encoder.memory_size
        elif encoder_type == "transition": # sars' (or a r s a r s) defines aggregates
            self.enc_state_size = self.size_before_enc # only need to store prior input to get transition
            self.encoder = TransitionEncoder(input_size=self.size_before_enc, 
                            encoding_size=self.enc_size,
                            s_embed_size=s_embed_size)
        else: # ars' defines aggregates, just use a FF layer
            assert encoder_type is None, encoder_type
            self.enc_state_size = 0 # Not used
            self.encoder = LinearEncoder(self.size_before_enc, self.enc_size)
        curr_dim = self.enc_size
        self.size_after_enc = curr_dim

        # FC layers after the encoder
        if agg_type == "gauss": # Gauss cannot have any layers after agg, so move them here
            layers_after_enc = layers_after_enc + layers_after_agg
            layers_after_agg = ()
        self.fc_after_enc = nn.ModuleList([])
        for i in range(len(layers_after_enc)):
            self.fc_after_enc.append(nn.Linear(curr_dim, layers_after_enc[i]))
            curr_dim = layers_after_enc[i]
        self.size_before_agg = curr_dim

        # Define aggregator
        if agg_type is None: # no aggregator: just use enc output
            self.agg = None
            self.skip_type = None
            self.agg_state_size = 0
            self.size_after_agg = self.size_before_agg
        else: # aggregator:
            # define aggregator function
            self.agg = AggregatorFunction(agg_type=agg_type, st_estimator=st_estimator, wavg_temp=args.wavg_temp,
                softmax_temp=args.softmax_temp, learn_temps=args.learn_temps,)
            # compute size of output
            if agg_type == "weighted_avg":
                assert self.size_before_agg%2 == 0, self.size_before_agg # must be divisible if splitting encoding
                if skip_type == "enc":
                    original_sz = self.size_before_agg//2
                    self.size_after_agg = original_sz + (original_sz//2) # half neurons not output, but rest skip around. this is 1.5x given fix above.
                else:
                    self.size_after_agg = self.size_before_agg//2 # half of the neurons (computing weights) are not output
            else:
                self.size_after_agg = self.size_before_agg
            # compute size of aggregator memory
            if skip_type == "enc":
                assert agg_type != "gauss", "Change argument: Gauss cannot have skip between aggregator and latent since aggregator directly outputs latent."
                assert self.size_before_agg%2 == 0, self.size_before_agg # must be divisible if splitting encoding
                self.agg_state_size = self.size_before_agg//2 # only half of neurons from encoding are aggregated
            else:
                self.agg_state_size = self.size_before_agg
            # add additional storage to aggregators if needed
            if agg_type in ["avg", "am"]:
                self.agg_state_size += 1 # Need to store count
            elif agg_type == "softmax":
                self.agg_state_size = 2*self.agg_state_size # Need to store running sum and running sum of softmax weights
        curr_dim = self.size_after_agg

        # Combined state size for memory of both enc and aggregator
        self.state_size = self.enc_state_size + self.agg_state_size

        # FC layers after the aggregator
        self.fc_after_agg = nn.ModuleList([])
        for i in range(len(layers_after_agg)):
            assert agg_type != "gauss", agg_type
            self.fc_after_agg.append(nn.Linear(curr_dim, layers_after_agg[i]))
            curr_dim = layers_after_agg[i]
        assert curr_dim == 2*latent_dim, (curr_dim, latent_dim)

        if self.learn_enc_state:
            self.learned_init_enc_state = nn.Parameter(self.init_state_fixed(1)[:, :, :self.enc_state_size])
        if self.learn_agg_state:
            self.learned_init_agg_state = nn.Parameter(self.init_state_fixed(1)[:, :, self.enc_state_size:])
        if self.agg:
            self.agg.init_state = self.init_state(1).detach()[:, :, self.enc_state_size:] # pass a copy of initial state to aggregator


    def init_state_fixed(self, batch_size, max_low_const=-999999):
        # Get enc state
        enc_state = torch.zeros((1, batch_size, self.enc_state_size), requires_grad=True)
        # Get aggregator 
        if (self.agg_type in ["max" , "max_norm"]) and self.max_init_low: # max's state should start at -inf (approximated by -9999) not 0 
            agg_state = max_low_const*torch.ones((1, batch_size, self.agg_state_size), requires_grad=True)
        else:
            agg_state = torch.zeros((1, batch_size, self.agg_state_size), requires_grad=True) 
        # Combine
        init_state = torch.cat((enc_state.to(device), agg_state.to(device)), dim=-1)
        assert init_state.shape == (1, batch_size, self.state_size), init_state.shape
        return init_state.to(device)


    def init_state(self, batch_size):
        fixed_st = self.init_state_fixed(batch_size)
        enc_state = self.learned_init_enc_state.clone().expand((1, batch_size, -1)) if self.learn_enc_state else fixed_st[:, :, :self.enc_state_size]
        agg_state = self.learned_init_agg_state.clone().expand((1, batch_size, -1)) if self.learn_agg_state else fixed_st[:, :, self.enc_state_size:]
        return torch.cat((enc_state.to(device), agg_state.to(device)), dim=-1)

    def init_rnn(self, eps=0.0001):
        for name, param in self.encoder.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name and name != "LN.weight": # LayerNorm initial weight should remain 1.
                nn.init.orthogonal_(param)
        if self.args.invariant_init: 
            assert self.encoder_type in ["lstm"], "Invariant init only implemented for LSTM"
            # Calculate biases values needed using inverse of sigmoid to attain gate value of 1-eps
            large_bias_value = np.log((1-eps) / eps)
            # Calculate the size of one-fourth of the biases and weights. (There are four gates in the LSTM.)
            bias_size = self.encoder.bias_ih_l0.size(0) // 4
            weight_size = self.encoder.weight_ih_l0.size(0) // 4
            # Modify the weights and biases for the forget gate (first fourth) to be 1-eps
            self.encoder.bias_ih_l0.data[:bias_size] = large_bias_value
            self.encoder.bias_hh_l0.data[:bias_size] = large_bias_value
            if self.args.invariant_init_adjusts_weights:
                self.encoder.weight_ih_l0.data[:weight_size] = 0
                self.encoder.weight_hh_l0.data[:weight_size] = 0
            # Modify the weights and biases for the input gate (second fourth) to be 1-eps
            self.encoder.bias_ih_l0.data[bias_size:2*bias_size] = large_bias_value
            self.encoder.bias_hh_l0.data[bias_size:2*bias_size] = large_bias_value
            if self.args.invariant_init_adjusts_weights:
                self.encoder.weight_ih_l0.data[bias_size:2*bias_size] = 0
                self.encoder.weight_hh_l0.data[bias_size:2*bias_size] = 0
            # Set recurrent weights to 0 for the cell and output gates
            if self.args.invariant_init_adjusts_weights:
                self.encoder.weight_hh_l0.data[2*weight_size:3*weight_size] = 0
                self.encoder.weight_hh_l0.data[3*weight_size:] = 0


    def forward(self, sequences, state, return_all_hidden_states=False):
        # shape of sequences should be: sequence_len x batch_size x input_size
        assert len(sequences.shape) == 3, len(sequences.shape)
        assert sequences.shape[-1] == self.input_size, sequences.shape
        sequence_len, batch_size, self.input_size = sequences.shape
        # shape of state should be: 1 x batch_size x self.state_size
        assert len(state.shape) == 3, len(state.shape)
        assert state.shape == (1,batch_size,self.state_size), (state.shape, (1,batch_size,self.state_size))

        x = sequences

        # FC
        for i in range(len(self.fc_before_enc)):
            x = F.relu(self.fc_before_enc[i](x))
        
        # split memory between encoder and aggregator
        enc_state = state[:, :, :self.enc_state_size].contiguous()
        agg_state = state[:, :, self.enc_state_size:]
        assert agg_state.shape[-1] == self.agg_state_size, (agg_state.shape[-1], self.agg_state_size)
        
        # encode x
        if self.encoder_type == "gru":
            if return_all_hidden_states:
                x, _ = self.encoder(x, enc_state) # gru only returns last state, but gru outputs also equals its hidden states
                enc_states = x
            else:
                x, enc_state = self.encoder(x, enc_state)
        elif self.encoder_type == "lstm":
            assert not return_all_hidden_states, "Not supported for LSTM"
            h_lstm, c_lstm = torch.chunk(enc_state, 2, dim=-1) 
            x, (h_lstm, c_lstm) = self.encoder(x, (h_lstm.contiguous(), c_lstm.contiguous()))
            enc_state = torch.cat((h_lstm, c_lstm), dim=-1)
        else:
            if return_all_hidden_states:
                x, enc_states = self.encoder(x, enc_state, return_all_hidden_states=True)
            else:
                x, enc_state = self.encoder(x, enc_state, return_all_hidden_states=False)

        # FC 
        # last layer should be linear if Gauss aggregator or weighted avg, since neurons need to be negative
        # last layer is linear if using max aggregator and max_init_low (see doc string for reason.)
        # last layer is linear if using self.args.hyper_agg
        # so for consistency, just make this last layer always linear:
        last_layer_is_linear = True
        # alternatively do:
            # gauss_or_avg = self.agg_type in ["weighted_avg", "gauss"] 
            # max_and_low = (self.agg_type in ["max", "max_norm"]) and self.max_init_low
            # last_layer_is_linear = gauss_or_avg or max_and_low
        x = apply_fc_layers(x, self.fc_after_enc, last_layer_is_linear)

        # aggregate x
        if self.skip_type == "enc":
            skip_neurons, x = torch.chunk(x, 2, dim=-1) 
        if self.agg is not None:
            aggregates, agg_states = [], []
            for t in range(x.shape[0]):
                # get encoded vector to aggregate
                to_agg = x[t:t+1, :, :] # input to aggregator at time t, leaving time dimension
                agg_out, agg_state = self.agg(to_agg, agg_state)
                aggregates.append(agg_out)
                if return_all_hidden_states:
                    agg_states.append(agg_state)
            x = torch.cat(aggregates, dim=0)
            if return_all_hidden_states:
                agg_states = torch.cat(agg_states, dim=0)

        # Skip connections around aggregator and FC, if not gauss
        if self.agg_type != "gauss": # Gauss cannot have FC or skip between aggregator and latent since aggregator directly outputs latent
            # Skip
            if self.skip_type == "enc":
                x = torch.cat((skip_neurons,x), dim=-1)
            assert x.shape == (sequence_len, batch_size, self.size_after_agg), (x.shape, (sequence_len, batch_size, self.size_after_agg))
            # FC
            x = apply_fc_layers(x, self.fc_after_agg, True)
            assert x.shape == (sequence_len, batch_size, 2*self.latent_dim), x.shape
        else: # Just shape checks
            assert x.shape == (sequence_len, batch_size, self.size_after_agg), x.shape
            assert x.shape == (sequence_len, batch_size, 2*self.latent_dim), x.shape

        # rejoin memory
        if return_all_hidden_states:
            new_states = torch.cat((enc_states,agg_states), dim=-1) if self.agg else enc_states
            assert new_states.shape[1:] == state.shape[1:], (new_states.shape, state.shape)
            assert x.shape[:-1] == new_states.shape[:-1], (x.shape, new_states.shape)
            return x, new_states
        else:
            new_state = torch.cat((enc_state,agg_state), dim=-1)
            assert new_state.shape == state.shape, (new_states.shape, state.shape)
            return x, new_state

        
class BatchFirstMultiheadAttention(nn.Module):
     # Wraps MultiheadAttention to allow batch to be passed as the first dimension
     def __init__(self, embed_dim, num_heads, **kwargs):
         super().__init__()
         self.mha = MultiheadAttention(embed_dim, num_heads, **kwargs)

     def forward(self, query, key, value, **kwargs):
         query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
         output, weights = self.mha(query, key, value, **kwargs)
         return output.transpose(0, 1), weights


class TransformerEncoder(nn.Module):
    def __init__(self, input_size, num_layers, num_heads, max_seq_len, output_sz, memory_size=None, token_size=None):
        """
        If slot_size is None, the slot size will be computed so that the transformer uses memory_size in storing previous tokens
        If memory_size is None, then the memory needed will be set based on slot_size.
        """
        super().__init__()
        self.input_size = input_size
        self.memory_size = memory_size
        self.token_size = token_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.output_size = output_sz

        assert not (memory_size is None and token_size is None), "token_size or memory_size must be specified."
        assert not (memory_size is not None and token_size is not None), "Only one of token_size and memory_size can be specified."

        if token_size is None:
            self.token_size = self.memory_size//self.max_seq_len
            assert self.token_size >= 2, "Not enough memory for given sequence length"
            self.memory_size = self.token_size * self.max_seq_len
        else:
            self.memory_size = token_size * self.max_seq_len
            self.token_size = token_size

        self.memory_size += 1 # Add one to store the step count

        self.output_layer = nn.Linear(self.token_size, self.output_size)
        self.input_layer = nn.Linear(self.input_size, self.token_size)
        self.transformer_layers = nn.ModuleList([TransformerBlock(self.token_size, self.num_heads) for _ in range(self.num_layers)])
        

    def forward(self, sequences, h_state, return_all_hidden_states=False):
        # shape of sequences should be: sequence_len x batch_size x input_size
        assert len(sequences.shape) == 3, len(sequences.shape)
        assert sequences.shape[-1] == self.input_size, sequences.shape
        sequence_len, batch_size, input_size = sequences.shape
        # shape of h_state should be: 1 x batch_size x memory_size
        assert h_state.shape == (1, batch_size, self.memory_size), h_state.shape

        outputs, new_h_states = [], []
        x = self.input_layer(sequences) # encoded inputs of correct size
        h = h_state.squeeze(0)
        cur_idx = h[:,-1].detach().long() # the current index of each item in the batch is saved at the end of memory
        h = h[:,:-1] # remove the current index
        h = h.reshape(batch_size, self.max_seq_len, self.token_size) # reshape hidden states
        # Shrink h to the max sequence length in this batch to save on compute
        max_idx = torch.max(cur_idx)
        max_len_here = max_idx + sequence_len
        original_padding = h[:,max_len_here:,:]
        h = h[:,:max_len_here,:]
        
        for t in range(sequence_len):
            x_t = x[t, :, :] # input to for first slot at time t
            assert x_t.shape == (batch_size, self.token_size), x_t.shape
            assert h.shape == (batch_size, max_len_here, self.token_size), h.shape

            # Update the hidden state by adding x_t at cur_idx
            h = h.clone()
            h[torch.arange(batch_size), cur_idx, :] = x_t

            # Create attention mask using cur_idx
            range_tensor = torch.arange(max_len_here).unsqueeze(0).expand(batch_size, -1).to(device)
            expanded_cur_idx = cur_idx.unsqueeze(-1).expand(-1, max_len_here)
            # Create the attention mask where positions <= cur_idx are False (allowed to attend) and others are True (masked)
            self_attn_mask = range_tensor > expanded_cur_idx
            # Create a mask for the general (not self) attention. Start with the mask set to False (allowed to attend)
            attn_mask = torch.zeros((batch_size, max_len_here, max_len_here)).bool().to(device)
            # Mask in both sequence dimensions
            attn_mask = attn_mask | self_attn_mask.unsqueeze(1) | self_attn_mask.unsqueeze(2)
            # Repeat for the number of heads in attention layer
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).reshape((batch_size * self.num_heads, max_len_here, max_len_here))
            # First position in each sequence must be allowed to attend, so that no sequence has no attention
            attn_mask[:, :, 0] = False

            # Apply transformer layers
            for transformer_layer in self.transformer_layers:
                h = transformer_layer(h, mask=attn_mask)
            assert not torch.isnan(h).any(), h

            # Get output
            output = h[torch.arange(batch_size), cur_idx, :]
            output = self.output_layer(output).reshape(batch_size, self.output_size)
            outputs.append(output)

            # Update index
            cur_idx = cur_idx + 1

            # Get new state for output
            h_state_with_padding = torch.cat((h, original_padding), dim=1)
            new_h_state = h_state_with_padding.reshape(batch_size, self.memory_size-1)
            new_cur_idx = cur_idx.reshape(batch_size, 1).float()
            h_with_indx = torch.cat((new_h_state,new_cur_idx.float()), dim=-1)
            new_h_states.append(h_with_indx)

        outputs = torch.stack(outputs, dim=0)
        new_h_states = torch.stack(new_h_states, dim=0)
        if not return_all_hidden_states:
            new_h_states = new_h_states[-1:, :, :]
        return outputs, new_h_states

class TransformerBlock(nn.Module):
    def __init__(self, token_size, num_heads, norm_before_sum=False):
        super().__init__()
        self.attention = BatchFirstMultiheadAttention(token_size, num_heads)
        self.norm1 = LayerNorm(token_size)
        self.norm2 = LayerNorm(token_size)
        self.ff = nn.Linear(token_size, token_size)
        self.norm_before_sum = norm_before_sum

    def forward(self, x, mask=None):
        if self.norm_before_sum:
            normd_x = self.norm1(x)
            x_attended, _ = self.attention(normd_x, normd_x, normd_x, attn_mask=mask)
            x = x + x_attended
            normd_x = self.norm2(x)
            x_ff = self.ff(normd_x)
            x = x + x_ff
        else:
            x_attended, _ = self.attention(x, x, x, attn_mask=mask)
            x = x + x_attended
            x = self.norm1(x)
            x_ff = self.ff(x)
            x = x + x_ff
            x = self.norm2(x)
        return x

class LinearEncoder(nn.Module):
    def __init__(self, size_before_enc, enc_size, activation=torch.tanh):
        # Note tanh is the default so that activations are similar to that from gru output
        super().__init__()
        self.activation = activation
        self.layer = nn.Linear(size_before_enc, enc_size)

    def forward(self, sequences, ignored_state, return_all_hidden_states=False):
        encoded = self.activation(self.layer(sequences))
        st = ignored_state.repeat((sequences.shape[0],1,1)) if return_all_hidden_states else ignored_state
        return encoded, st


class TransitionEncoder(nn.Module):
    def __init__(self, input_size, encoding_size, s_embed_size, activation=torch.tanh, ignore_prev_ar=True):
        """ Take a sequence of hidden states output sequence of encoded transition pairs """
        # Note: if ignore_prev_ar is True, then the prior action and reward at the previous timestep (ar)
        # will be ignored. Thus collectively, we encode (s,ars') instead of (ars,ars), which includes a and r from the prior transition.
        # Note tanh is the default so that activations are similar to that from gru output
        super().__init__()
        self.ignore_prev_ar = ignore_prev_ar
        self.input_size = input_size 
        self.encoding_size = encoding_size
        self.s_embed_size = s_embed_size # size of state embedding, assumed to be first
        self.combined_size = (input_size + s_embed_size) if ignore_prev_ar else 2*input_size
        self.layer = nn.Linear(self.combined_size, encoding_size)
        self.activation = activation

    def forward(self, sequences, initial_ars, return_all_hidden_states=False):
        # shape of sequences should be: sequence_len x batch_size x input_size
        assert len(sequences.shape) == 3, len(sequences.shape)
        assert sequences.shape[-1] == self.input_size, sequences.shape
        sequence_len, batch_size, input_size = sequences.shape
        # shape of initial_ars should be: 1 x batch_size x input_size
        if initial_ars is None:
            initial_ars = torch.zeros_like(sequences[0:1,:,:])
        assert len(initial_ars.shape) == 3, len(initial_ars.shape)
        assert initial_ars.shape == (1,batch_size,input_size), (initial_ars.shape, (1,batch_size,input_size))
        # shift states by one to get prior states:
        if sequences.shape[0] > 1: # sequence length > 1, need prior state for each
            prior_seqs = sequences[:-1,:,:] # remove last state
            prior_seqs = torch.cat((initial_ars, prior_seqs), dim=0) # put initial_ars at start
        else: # sequence length 1, need only 1 prior state
            prior_seqs = initial_ars
        assert prior_seqs.shape == sequences.shape, (prior_seqs.shape, sequences.shape)
        # mask off prior action and reward if necessary (the action and reward encoded are already from the last timestep)
        if self.ignore_prev_ar:
            prior_seqs = prior_seqs[:,:,-self.s_embed_size:]
        # combine transitions:
        combined_states = torch.cat((prior_seqs, sequences), dim=-1)
        assert len(combined_states.shape) == 3, len(combined_states.shape)
        assert combined_states.shape[-1] == (self.combined_size), (combined_states.shape, sequences.shape)
        encoded_transitions = self.activation(self.layer(combined_states))
        assert encoded_transitions.shape[:-1] == sequences.shape[:-1], (encoded_transitions.shape, sequences.shape)
        if return_all_hidden_states:
            new_initial_ars = sequences
        else:
            new_initial_ars = sequences[-1:, :, :] # grab last state as next initial_ars
        return encoded_transitions, new_initial_ars


# Max operation with "Straight Through" grad
class MaxST(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor1, tensor2):
        return torch.max(tensor1, tensor2)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output

# Division operation with "Straight Through" grad
class DivST(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, constant):
        return tensor/constant
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class AggregatorFunction(nn.Module):
    def __init__(self, agg_type, st_estimator, learn_temps, softmax_temp, wavg_temp, init_state = None): # See Aggregator for arguments
        super().__init__()
        self.agg_type = agg_type
        self.st_estimator = st_estimator
        self.init_state = init_state
        self.wavg_temp = wavg_temp
        self.softmax_temp = softmax_temp
        self.learn_temps = learn_temps
        if self.learn_temps:
            self.softmax_temp = nn.Parameter(torch.tensor([self.softmax_temp])).to(device)
            if self.wavg_temp is not None:
                self.wavg_temp = nn.Parameter(torch.tensor([self.wavg_temp])).to(device)

    def forward(self, curr_input, agg_state, eps=0.1):
        assert len(curr_input.shape) == 3, len(curr_input.shape)
        assert curr_input.shape[:-1] == agg_state.shape[:-1], (curr_input.shape, agg_state.shape)
        if self.agg_type == "sum":
            assert curr_input.shape[-1] == agg_state.shape[-1], (curr_input.shape, agg_state.shape)
            new_state = agg_state+curr_input
            output = new_state
        elif self.agg_type in ["max", "max_norm"]:
            assert curr_input.shape[-1] == agg_state.shape[-1], (curr_input.shape, agg_state.shape)
            new_state = MaxST.apply(agg_state, curr_input) if self.st_estimator else torch.max(agg_state, curr_input)
            output = new_state
            if self.agg_type == "max_norm":
                norm = torch.norm(output, p=2, dim=-1, keepdim=True).detach()
                norm = torch.max(norm, eps*torch.ones_like(norm)) # make sure >= eps
                output = DivST.apply(output,norm) if self.st_estimator else output/norm
        elif self.agg_type == "avg":
            # There must be extra neuron in aggregator so that it can store an extra scalar
            assert agg_state.shape[-1] == curr_input.shape[-1]+1, (agg_state.shape[-1], curr_input.shape[-1])
            count = agg_state[:,:,-1:] # count of time steps
            new_count = count + 1
            new_enc_sum = agg_state[:,:,:-1] + curr_input # compute sum of encodings
            new_state = torch.cat((new_enc_sum, new_count), dim=-1)
            output = DivST.apply(new_enc_sum, new_count) if self.st_estimator else new_enc_sum/new_count
        elif self.agg_type == "am": # Compute half average and half max
            # There must be extra neuron in aggregator so that it can store an extra scalar
            assert agg_state.shape[-1] == curr_input.shape[-1]+1, (agg_state.shape[-1], curr_input.shape[-1])
            avg_in, max_in = torch.chunk(curr_input, 2, dim=-1)
            count = agg_state[:,:,-1:] # count of time steps
            new_count = count + 1
            old_sum, old_max = torch.chunk(agg_state[:,:,:-1], 2, dim=-1)
            new_sum = old_sum + avg_in
            new_max = MaxST.apply(old_max, max_in) if self.st_estimator else torch.max(old_max, max_in)
            new_state = torch.cat((new_sum, new_max, new_count), dim=-1)
            max_out = new_max
            avg_out = DivST.apply(new_sum, new_count) if self.st_estimator else new_sum/new_count
            output = torch.cat((avg_out, max_out), dim=-1)
        elif self.agg_type == "weighted_avg":
            # split input into 2 quantities:
            # 1) x
            # 2) w
            x, w = torch.chunk(curr_input, 2, -1)
            if self.wavg_temp is None:
                w = F.softplus(w) # weights need to be positive in weighted avg
            else: # softmax weights to interpolate between avg and max
                w = torch.exp(w/self.wavg_temp - 2/self.wavg_temp) # -2/t is identical and prevents overflow
                eps = torch.exp(torch.tensor([-3/self.wavg_temp]).detach()).to(device)
                assert eps[0] != 0, "wavg_temp too small"
            # split memory into 2 quantities: 
            # A) x1*w1 + ... + xn*wn
            # B) w1    + ... + wn
            x_sum, w_sum = torch.chunk(agg_state, 2, dim=-1)
            # Update A and B
            new_x_sum = x_sum + (x*w) # add weighed value
            new_w_sum = w_sum + w # add weight
            # Combine A and B to get new state
            new_state = torch.cat((new_x_sum, new_w_sum), dim=-1)
            # Define output
            new_w_sum = torch.max(new_w_sum, eps*torch.ones_like(new_w_sum)) # make sure >= eps
            avg = DivST.apply(new_x_sum, new_w_sum) if self.st_estimator \
                else new_x_sum/new_w_sum
            output = avg
            assert not torch.isnan(output).any(), output # This can happen if using torch.exp for the weights
        elif self.agg_type == "softmax":
            x = curr_input
            w = torch.exp(x/self.softmax_temp - 2/self.softmax_temp) # -2/t is identical and prevents overflow
            eps = torch.exp(torch.tensor([-3/self.softmax_temp]).detach()).to(device)
            assert eps[0] != 0, "softmax_temp too small"
            x_sum, w_sum = torch.chunk(agg_state, 2, dim=-1)
            new_x_sum = x_sum + (x*w) # add weighed value
            new_w_sum = w_sum + w # add weight
            new_state = torch.cat((new_x_sum, new_w_sum), dim=-1)
            new_w_sum = torch.max(new_w_sum, eps*torch.ones_like(new_w_sum)) # make sure >= eps
            avg = DivST.apply(new_x_sum, new_w_sum) if self.st_estimator \
                else new_x_sum/new_w_sum
            output = avg
            assert not torch.isnan(output).any(), output
        else: # gauss
            assert self.agg_type == "gauss"
            # split input into 2 quantities:
            # 1) mu_i        =   m^2
            # 2) 1/sigma_i^2 = 1/s^2
            mu, one_div_s2 = torch.chunk(curr_input, 2, -1) # predict mu and 1/s2 directly
            one_div_s2 = F.softplus(one_div_s2) # variance needs to be positive
            # Alternatively, predict s2 then take reciprocal
            # Alternatively, predict 1/s and square
            # softplus was chosen to stay in line with PEARL, predicting reciprocal directly was chosen to 
            # stay in line with weighted_avg
            #
            # Now, split memory into 2 quantities: 
            # A) m1/s1^2 + ... + mn/sn^2
            # B) 1/s1^2  + ... +  1/sn^2
            weighted_mu_sum, reciprocal_s2_sum = torch.chunk(agg_state, 2, dim=-1)
            # Update A and B
            new_weighted_mu_sum = weighted_mu_sum + (mu*one_div_s2) # add m/s^2
            new_reciprocal_s2_sum = reciprocal_s2_sum + one_div_s2 # add 1/s^2
            # Combine A and B to get new state
            new_state = torch.cat((new_weighted_mu_sum, new_reciprocal_s2_sum), dim=-1)
            # Define output mu', s2'
            # mu' = A / B (i.e. weighted average)
            new_reciprocal_s2_sum = torch.max(new_reciprocal_s2_sum, eps*torch.ones_like(new_reciprocal_s2_sum)) # make sure >= eps
            new_mu = DivST.apply(new_weighted_mu_sum, new_reciprocal_s2_sum) if self.st_estimator \
                else new_weighted_mu_sum/new_reciprocal_s2_sum
            # s2' = 1 / B
            new_s2 = torch.reciprocal(new_reciprocal_s2_sum)
            # Combine to get output
            output = torch.cat((new_mu, torch.log(new_s2)), dim=-1) # it is assumed that log-var is output
        
        return output, new_state

def apply_fc_layers(x, layers, last_layer_is_linear):
    linear_layer_reached = False
    for i in range(len(layers)):
        x = layers[i](x)
        if last_layer_is_linear and (i == len(layers)-1):
            # if on last later, it must be linear
            linear_layer_reached = True
            break
        x = F.relu(x)
    assert linear_layer_reached or (not last_layer_is_linear) or (not layers), "Indexing in loop above likely incorrect."
    return x


def unit_tests():
    #### Test TransitionEncoder() ####
    
    input_size, encoding_size, s_embed_size = 4, 1, 3
    tran_enc = TransitionEncoder(input_size, encoding_size, s_embed_size, ignore_prev_ar=False)
    for name, param in tran_enc.named_parameters(): # set weights = 1, bias = 0
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.constant_(param, 1)

    seq = torch.tensor([[[.01,.02,.03,.04]],
                        [[.04,.05,.06,.07]]])
    expected_ans = torch.tensor([[[torch.tanh(torch.tensor(0.+0.+0.+0.+.01+.02+.03+.04))]],
                                 [[torch.tanh(torch.tensor(.01+.02+.03+.04+.04+.05+.06+.07))]]])
    ans, out_state = tran_enc(seq,None,return_all_hidden_states=False)
    assert torch.all(ans == expected_ans), (ans, expected_ans)
    assert torch.all(out_state == torch.tensor([[[.04,.05,.06,.07]]])), (out_state)
    
    # Redo with ignore_prev_ar=True
    tran_enc = TransitionEncoder(input_size, encoding_size, s_embed_size, ignore_prev_ar=True)
    for name, param in tran_enc.named_parameters(): # set weights = 1, bias = 0
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.constant_(param, 1)
    seq2 = torch.tensor([[[.05,.02,.03,.04]], #.01 is too small to mask off
                         [[.04,.05,.06,.07]]])
    expected_ans = torch.tensor([[[torch.tanh(torch.tensor(0.+0.+0.+0.+.05+.02+.03+0.04))]],
                                 [[torch.tanh(torch.tensor(0.+.02+.03+.04+.04+.05+.06+.07))]]])
    ans, out_state = tran_enc(seq2,None,return_all_hidden_states=False)
    assert torch.all(ans == expected_ans), (ans, expected_ans)
    assert torch.all(out_state == torch.tensor([[[.04,.05,.06,.07]]])), (out_state)
    
    # Redo with different input state
    tran_enc = TransitionEncoder(input_size, encoding_size, s_embed_size, ignore_prev_ar=False)
    for name, param in tran_enc.named_parameters(): # set weights = 1, bias = 0
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.constant_(param, 1)
    input_state = torch.tensor([[[.01,.01,.02,.02]]])
    seq = torch.tensor([[[.01,.02,.03,.04]],
                        [[.04,.05,.06,.07]]])
    expected_ans = torch.tensor([[[torch.tanh(torch.tensor(.01+.01+.02+.02+.01+.02+.03+.04))]],
                                 [[torch.tanh(torch.tensor(.01+.02+.03+.04+.04+.05+.06+.07))]]])
    ans, out_state = tran_enc(seq,input_state,return_all_hidden_states=False)
    assert torch.all(ans == expected_ans), (ans, expected_ans)
    assert torch.all(out_state == torch.tensor([[[.04,.05,.06,.07]]])), (out_state)
    
    # Redo one step at a time
    tran_enc = TransitionEncoder(input_size, encoding_size, s_embed_size, ignore_prev_ar=False)
    for name, param in tran_enc.named_parameters(): # set weights = 1, bias = 0
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.constant_(param, 1)

    seq1 = torch.tensor([[[.01,.02,.03,.04]]])
    seq2 = torch.tensor([[[.04,.05,.06,.07]]])
    expected_ans1 = torch.tensor([[[torch.tanh(torch.tensor(0.+0.+0.+0.+.01+.02+.03+.04))]]])
    expected_ans2 = torch.tensor([[[torch.tanh(torch.tensor(.01+.02+.03+.04+.04+.05+.06+.07))]]])
    ans1, out_state1 = tran_enc(seq1,None,return_all_hidden_states=False)
    ans2, out_state2 = tran_enc(seq2,out_state1,return_all_hidden_states=False)
    assert torch.all(ans1 == expected_ans1), (ans1, expected_ans1)
    assert torch.all(ans2 == expected_ans2), (ans2, expected_ans2)
    assert torch.all(out_state1 == seq1), (out_state1, seq1)
    assert torch.all(out_state2 == seq2), (out_state2, seq2)




    #### Test AggregatorFunction for output, memory, grad approximation ####
    import math
    
    # Sum: output and memory
    for st in [True, False]:
        sum_agg = AggregatorFunction("sum", st_estimator=st)
        init_st = torch.zeros((1,2,3,), requires_grad=True)
        to_agg = torch.tensor([[[1.,2.,3.],
                                [4.,5.,6.]]], requires_grad=True)
        out, new_st = sum_agg(to_agg, init_st)
        assert torch.all(out == to_agg), (out, to_agg)
        assert torch.all(new_st == to_agg), (new_st, to_agg)
        out, new_st = sum_agg(to_agg, new_st)
        assert torch.all(out == 2*to_agg), (out, to_agg)
        assert torch.all(new_st == 2*to_agg), (new_st, to_agg)
        out, new_st = sum_agg(to_agg, new_st)
        assert torch.all(out == 3*to_agg), (out, to_agg)
        assert torch.all(new_st == 3*to_agg), (new_st, to_agg)
    # grad
    for st in [True, False]:
        sum_agg = AggregatorFunction("sum", st_estimator=st)
        init_st = torch.zeros((1,2,3,), requires_grad=True)

        # Aggregate 1
        to_agg = torch.tensor([[[1.,2.,3.],
                                [4.,5.,6.]]], requires_grad=True)
        out, new_st = sum_agg(to_agg, init_st)
        #
        expected_grad_batch_1 = torch.tensor([[[1., 1., 1.],
                                               [0., 0., 0.]]])
        grad_batch_1 = torch.autograd.grad(torch.sum(out[0,0,:]), to_agg)[0] # just sum first batch
        assert torch.all(grad_batch_1 == expected_grad_batch_1), (grad_batch_1, expected_grad_batch_1)
        expected_grad_batch_2 = torch.tensor([[[0., 0., 0.],
                                               [1., 1., 1.]]])
        grad_batch_2 = torch.autograd.grad(torch.sum(out[0,1,:]), to_agg)[0]
        assert torch.all(grad_batch_2 == expected_grad_batch_2), (grad_batch_2, expected_grad_batch_2)

        # Aggregate 2
        to_agg = torch.tensor([[[55.,2.,6.],
                                [1.,3.,1.]]], requires_grad=True)
        out, new_st = sum_agg(to_agg, new_st)
        #
        expected_grad_batch_1 = torch.tensor([[[1., 1., 1.],
                                               [0., 0., 0.]]])
        grad_batch_1 = torch.autograd.grad(torch.sum(out[0,0,:]), to_agg)[0]
        assert torch.all(grad_batch_1 == expected_grad_batch_1), (grad_batch_1, expected_grad_batch_1)
        expected_grad_batch_2 = torch.tensor([[[0., 0., 0.],
                                               [1., 1., 1.]]])
        grad_batch_2 = torch.autograd.grad(torch.sum(out[0,1,:]), to_agg)[0]
        assert torch.all(grad_batch_2 == expected_grad_batch_2), (grad_batch_2, expected_grad_batch_2)

    # TODO: compare full jacobian to that of sum for avg and max... should be fine but good to check.
    #       I am having issues flattening the jacobian into a comparable form
    # def get_jacobian_sum(to_agg):
    #     st = torch.zeros_like(to_agg)
    #     sum_agg = AggregatorFunction("sum", False)
    #     return torch.autograd.functional.jacobian(sum_agg, (to_agg, st))

    # Avg: output, memory, grad
    for st in [True, False]:
        avg_agg = AggregatorFunction("avg", st_estimator=st)
        init_st = torch.zeros((1,2,4,), requires_grad=True)
        in_1 = torch.tensor([[[2.,2.,2.],
                              [10.,10.,10.]]], requires_grad=True)
        out, new_st = avg_agg(in_1, init_st)
        expected_out = torch.tensor([[[2.,2.,2.],
                                     [10.,10.,10.]]])
        expected_st = torch.tensor([[[2.,2.,2.,1.],
                                    [10.,10.,10.,1.]]])
        assert torch.all(out == expected_out), (out, expected_out)
        assert torch.all(new_st == expected_st), (new_st, expected_st)
        # compare jacobian to sum... nested tuple making issues...
        # jacobian = flatten_jacobian(torch.autograd.functional.jacobian(avg_agg, (in_1, init_st)))
        # sum_jacobian = flatten_jacobian(get_jacobian_sum(in_1))
        # expected_jacobian = sum_jacobian if st else sum_jacobian/1 # first timestep, so same
        in_2 = torch.tensor([[[4.,4.,4.],
                              [10.,10.,10.]]], requires_grad=True)
        out, new_st = avg_agg(in_2, new_st)
        expected_out = torch.tensor([[[3.,3.,3.],
                                     [10.,10.,10.]]])
        expected_st = torch.tensor([[[6.,6.,6.,2.],
                                    [20.,20.,20.,2.]]])
        assert torch.all(out == expected_out), (out, expected_out)
        assert torch.all(new_st == expected_st), (new_st, expected_st)
        # a grad check (Should be same as sum for ST and sum divided by count otherwise)
        if st:
            expected_grad = torch.tensor([[[1., 1., 1.],
                                           [0., 0., 0.]]])
        else:
            expected_grad = torch.tensor([[[1., 1., 1.],
                                           [0., 0., 0.]]])/2.
        grad = torch.autograd.grad(torch.sum(out[0,0,:]), in_2)[0]
        assert torch.all(grad == expected_grad), (grad, expected_grad)
        # now try reseting the count of just one batch
        new_st[0,0,-1] = 0
        in_3 = torch.tensor([[[2.,2.,2.],
                              [2.,2.,2.]]], requires_grad=True)
        out, new_st = avg_agg(in_3, new_st)
        expected_out = torch.tensor([[[8.,8.,8.],
                                     [22/3,22/3,22/3]]])
        expected_st = torch.tensor([[[8.,8.,8.,1.],
                                    [22.,22.,22.,3.]]])
        assert torch.all(out == expected_out), (out, expected_out)
        assert torch.all(new_st == expected_st), (new_st, expected_st)

    # Max: output, memory, grad
    for st in [True, False]:
        max_agg = AggregatorFunction("max", st_estimator=st)
        init_st = -999*torch.ones((1,2,3,), requires_grad=True)
        in_1 = torch.tensor([[[2.,2.,2.],
                              [10.,10.,10.]]], requires_grad=True)
        out, new_st = max_agg(in_1, init_st)
        expected_out = torch.tensor([[[2.,2.,2.],
                                     [10.,10.,10.]]])
        expected_st = expected_out
        assert torch.all(out == expected_out), (out, expected_out)
        assert torch.all(new_st == expected_st), (new_st, expected_st)
        in_2 = torch.tensor([[[1.,3.,4.],
                              [-2.,-2.,20.]]], requires_grad=True)
        out, new_st = max_agg(in_2, new_st)
        expected_out = torch.tensor([[[2.,3.,4.],
                                     [10.,10.,20.]]])
        expected_st = expected_out
        assert torch.all(out == expected_out), (out, expected_out)
        assert torch.all(new_st == expected_st), (new_st, expected_st)
        # a grad check (Should be same as sum for ST and sum but masked where less than st otherwise)
        if st:
            expected_grad = torch.tensor([[[1., 1., 1.],
                                           [0., 0., 0.]]])
        else:
            expected_grad = torch.tensor([[[0, 1., 1.],
                                           [0., 0., 0.]]])
        grad = torch.autograd.grad(torch.sum(out[0,0,:]), in_2)[0]
        assert torch.allclose(grad, expected_grad), (grad, expected_grad)
        #

    # max_norm: output, memory, grad
    for st in [True, False]:
        max_agg = AggregatorFunction("max_norm", st_estimator=st)
        init_st = -999*torch.ones((1,2,3,), requires_grad=True)
        in_1 = torch.tensor([[[2.,2.,1.],
                              [6.,0,0]]], requires_grad=True)
        out, new_st = max_agg(in_1, init_st)
        expected_st = torch.tensor([[[2.,2.,1.],
                                    [6.,0.,0.]]])
        expected_out = torch.tensor([[[2./3,2./3,1./3],
                                     [6/6,0,0]]])
        assert torch.allclose(out, expected_out), (out, expected_out)
        assert torch.allclose(new_st, expected_st), (new_st, expected_st)
        in_2 = torch.tensor([[[0.,0.,4.],
                              [-2.,-2.,18]]], requires_grad=True)
        out, new_st = max_agg(in_2, new_st)
        expected_st = torch.tensor([[[2.,2.,4.],
                                    [6.,0.,18.]]])
        expected_out = torch.tensor([[[2./24**.5,2./24**.5,4./24**.5],
                                    [6./360**.5,0.,18./360**.5]]])
        assert torch.allclose(out, expected_out), (out, expected_out)
        assert torch.allclose(new_st, expected_st), (new_st, expected_st)
        # a grad check (Should be same as sum for ST and sum but masked where less than st otherwise)
        if st:
            expected_grad = torch.tensor([[[1., 1., 1.],
                                           [0., 0., 0.]]])
        else:
            expected_grad = torch.tensor([[[0, 0., 1.],
                                           [0., 0., 0.]]])/24**.5
        grad = torch.autograd.grad(torch.sum(out[0,0,:]), in_2)[0]
        assert torch.allclose(grad, expected_grad), (grad, expected_grad)

    # weighted_avg: check output, memory, grad
    for st in [True, False]:
        avg_agg = AggregatorFunction("weighted_avg", st_estimator=st)
        init_st = torch.zeros((1,2,4,), requires_grad=True)
        in_1 = torch.tensor([[[4.,2.,1,1],
                              [1.,1.,2,1]]], requires_grad=True)
        out, new_st = avg_agg(in_1, init_st)
        expected_st = torch.tensor([[[4.,2.,1,1],
                                    [2.,1.,2,1]]])
        expected_out = torch.tensor([[[4.,2.],
                                     [1.,1.]]])
        # This is approximate since we use softplus on the weights
        assert torch.allclose(out, expected_out, atol=2), (out, expected_out)
        assert torch.allclose(new_st, expected_st, atol=2), (new_st, expected_st)
        in_2 = torch.tensor([[[3.,2.,2,2],
                              [1.,1.,1,1]]], requires_grad=True)
        out, new_st = avg_agg(in_2, new_st)
        expected_st = torch.tensor([[[10.,6.,3,3],
                                    [3., 2.,3,2]]])
        expected_out = torch.tensor([[[10./3,6./3],
                                     [3./3,1./1]]])
        assert torch.allclose(out, expected_out, atol=2), (out, expected_out)
        assert torch.allclose(new_st, expected_st, atol=2), (new_st, expected_st)
        # a grad check (Should be same as sum for ST and sum but masked where less than st otherwise)
        if st:
            expected_grad = torch.tensor([[[2., 2., 3., 2.],
                                           [0., 0., 0., 0.]]])
            grad = torch.autograd.grad(torch.sum(out[0,0,:]), in_2)[0]
            assert torch.allclose(grad, expected_grad, atol=.5), (grad, expected_grad)
        else:
            pass # TODO, but should be fine since we know DivST works

    def softplus(x):
        return F.softplus(torch.tensor(float(x))).item()

    def log_reciprocal_softplus(x):
        import math
        return math.log(1/softplus(x))

    # gauss: check output, memory, grad
    #        also, compare to "weighed_avg"
    for st in [True, False]:
        gaus_agg = AggregatorFunction("gauss", st_estimator=st)
        wavg_agg = AggregatorFunction("weighted_avg", st_estimator=st) # for comparison
        init_st = torch.zeros((1,2,4,), requires_grad=True)
        in_1 = torch.tensor([[[4.,2.,1,1],
                              [1.,1.,2,1]]], requires_grad=True)
        out, new_st = gaus_agg(in_1, init_st)
        expected_st = torch.tensor([[[4.,2.,1,1],
                                    [2.,1.,2,1]]])
        expected_out = torch.tensor([[[4.,2.,log_reciprocal_softplus(1),log_reciprocal_softplus(1)],
                                     [1.,1.,log_reciprocal_softplus(2),log_reciprocal_softplus(1)]]])
        wavg_out, wavg_st = wavg_agg(in_1, init_st)
        # expected st is approximate since we use softplus on the weights
        assert torch.allclose(out, expected_out), (out, expected_out)
        assert torch.allclose(new_st, expected_st, atol=1.5), (new_st, expected_st)
        assert torch.all(wavg_st == new_st), (wavg_st, new_st)
        assert torch.all(wavg_out == out[:,:,:-2]), (wavg_out, out)
        in_2 = torch.tensor([[[3.,2.,2,2],
                              [1.,1.,1,1]]], requires_grad=True)
        out, new_st = gaus_agg(in_2, new_st)
        wavg_out, wavg_st = wavg_agg(in_2, wavg_st)
        assert torch.all(wavg_st == new_st), (wavg_st, new_st)
        assert torch.all(wavg_out == out[:,:,:-2]), (wavg_out, out)
        # a grad check (Should be same as sum for ST and sum but masked where less than st otherwise)
        grad_gauss = torch.autograd.grad(torch.sum(out[:,:,:-2]), in_2)[0]
        grad_wavg = torch.autograd.grad(torch.sum(wavg_out), in_2)[0]
        assert torch.all(grad_gauss == grad_wavg), (grad_gauss, grad_wavg)



if __name__ == "__main__":
    unit_tests()

