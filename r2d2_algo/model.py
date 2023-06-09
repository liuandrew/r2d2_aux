import torch
from torch import nn


'''
Class definitions for R2D2 network
'''
    
class ResettingGRU(nn.Module):
    '''
    Modification to GRU that can take dones on the forward call to tell when
    state in the middle of a batched sequence forward call should be reset
    '''
    def __init__(self, **kwargs):
        super().__init__()
        self.gru = nn.GRU(**kwargs)
        
        self.hidden_size = self.gru.hidden_size
        self.batch_first = self.gru.batch_first
        
    def get_rnn_hxs(self, num_batches=1):
        '''
        Get torch.zeros hidden states to start off with
        '''
        if num_batches == 1:
            return torch.zeros(1, self.hidden_size)
        else:
            return torch.zeros(1, num_batches, self.hidden_size)
        
    
    def forward(self, x, hidden_state, dones=None):
        if dones == None:
            return self.gru(x, hidden_state)
        
        #Unfortunately need to split up the batch here
        if self.batch_first == False:
            raise NotImplementedError        
        
        def single_gru_row(x_i, rnn_hx, done):
            breakpoints = (done == 1).argwhere()
            cur_idx = 0
            output_row = torch.zeros(x_i.shape[0], self.hidden_size)
            
            for breakpoint in breakpoints:
                if breakpoint == 0:
                    rnn_hx = self.get_rnn_hxs()
                    continue
                out, out_hx = self.gru(x_i[cur_idx:breakpoint], rnn_hx)
                output_row[cur_idx:breakpoint] = out
                rnn_hx = self.get_rnn_hxs()
                cur_idx = breakpoint
                
            if cur_idx < x_i.shape[0]:
                out, out_hx = self.gru(x_i[cur_idx:], rnn_hx)
                output_row[cur_idx:] = out
            return output_row, out_hx
        
        if hidden_state.dim() == 2:
            output, output_hx = single_gru_row(x, hidden_state, dones)
            
        else:
            num_batches = hidden_state.shape[1]
            
            # Output will have shape [N, L, hidden_size]
            full_out = torch.zeros((x.shape[0], x.shape[1], self.hidden_size))
            # hidden_state output has shape [1, N, hidden_size]
            full_hx_out = torch.zeros((1, x.shape[0], self.hidden_size))
            
            batchable_rows = (dones == 0).all(dim=1)
            individual_rows = (~batchable_rows).argwhere().reshape(-1)
            
            # First batch all computations that have no dones in them
            out, out_hx = self.gru(x[batchable_rows], hidden_state[:, batchable_rows, :])
            full_out[batchable_rows] = out
            full_hx_out[:, batchable_rows, :] = out_hx
            
            for i in individual_rows:
                d = dones[i]
                x_i = x[i]
                rnn_hx = hidden_state[:, i, :]
                
                output_row, output_hx_row = single_gru_row(x_i, rnn_hx, d)
                full_out[i] = output_row
                full_hx_out[:, i, :] = output_hx_row
            
        
        return full_out, full_hx_out
            
            
            
            
            
            
class ResettingGRUBatched(nn.Module):
    '''
    Modification to GRU that can take dones on the forward call to tell when
    state in the middle of a batched sequence forward call should be reset
    
    This version batches the forward call by creating a modified
      input matrix, appending rows and padding them to handle batching.
    We get the same output as expected, however due to the padding and inability
      to do early stopping of GRU, the hidden_state we get is incorrect when batching occurs.
    Since this forward call with dones is only used during training to generate Q-values,
      it doesn't affect the output. But be careful using this if the next_rnn_hxs are desired
      and want to use the forward dones functionality. 
    '''
    def __init__(self, **kwargs):
        super().__init__()
        
        
        self.gru = nn.GRU(**kwargs)
        
        self.hidden_size = self.gru.hidden_size
        self.batch_first = self.gru.batch_first
        
        #Forward pass splits and recombined batches so its easier
        #  to index with batch_first. The batch_first == False version
        #  is in commented lines in forward function
        if self.batch_first == False:
            raise NotImplementedError

        
    def get_rnn_hxs(self, num_batches=1):
        '''
        Get torch.zeros hidden states to start off with
        '''
        if num_batches == 1:
            return torch.zeros(1, self.hidden_size)
        else:
            return torch.zeros(1, num_batches, self.hidden_size)
        
    
    def forward(self, x, hidden_state, dones=None, masks=None):
        '''
        For batched forward pass
            x: (N, L, H_in)
            hidden_state: (1, L, hidden_size)
            dones: (N, L)
            
        For unbatched
            x: (L, H_in)
            hidden_state: (L, hidden_size)
            dones: (L,)
            
        Note that in unbatched, we will actually turn it into a batch
            hence we have a line of unsqueezes
        hidden_state has first dimension of 1 indicating 1 layer and not bi-directional
        
        Output: full_out (N, L, hidden_size), full_hx_out (1, N, hidden_size)
            Note that full_hx_out is NOT GENERALLY the same as the rnn_hidden_state
            you would expect from standard gru unit, since we sometimes pad the ending
            More correct to generally just take the last step of full_out
        '''
        
        if dones is None:
            if masks is None:
                return self.gru(x, hidden_state)
            else:
                return self.gru(x, hidden_state * masks)    
        
        if hidden_state.dim() == 2:
            hidden_state = hidden_state.unsqueeze(0)
            x = x.unsqueeze(0)
            dones = dones.unsqueeze(0)

        num_batches = x.shape[0]
            
        extra_rows = int(dones.sum().item())
        
        #Generate a new padded x and rnn_hxs vector to batch forward pass
        padded_x = torch.zeros((num_batches + extra_rows, x.shape[1], x.shape[2]))
        padded_rnn_hxs = torch.zeros((hidden_state.shape[0], num_batches + extra_rows, hidden_state.shape[2]))

        batchable_rows = (dones == 0).all(dim=1)
        num_batchable_rows = batchable_rows.sum().item()
        individual_rows = (~batchable_rows).argwhere().reshape(-1)

        #First N rows will be taken from rows with no dones
        padded_x[:num_batchable_rows] = x[batchable_rows]
        padded_rnn_hxs[:, :num_batchable_rows, :] = hidden_state[:, batchable_rows, :]
        # padded_rnn_hxs[:num_batchable_rows] = hidden_state[batchable_rows]

        #Remaining rows will be filled out in order of rows with dones
        cur_row_idx = num_batchable_rows
        for i in individual_rows:
            breakpoints = (dones[i] == 1).argwhere()
            cur_idx = 0
            rnn_hx = hidden_state[:, i, :]

            for breakpoint in breakpoints:
                if breakpoint == 0:
                    rnn_hx = self.get_rnn_hxs()
                    continue
                padded_x[cur_row_idx, :breakpoint-cur_idx, :] = x[i, cur_idx:breakpoint, :]
                padded_rnn_hxs[:, cur_row_idx, :] = rnn_hx
                # padded_rnn_hxs[cur_row_idx] = rnn_hx

                rnn_hx = self.get_rnn_hxs()
                cur_idx = breakpoint
                cur_row_idx += 1

            if cur_idx < len(dones[i]):
                padded_x[cur_row_idx, :len(dones[i])-cur_idx, :] = x[i, cur_idx:, :]
                padded_rnn_hxs[:, cur_row_idx, :] = rnn_hx
                # padded_rnn_hxs[cur_row_idx] = rnn_hx
                cur_row_idx += 1

        #Perform forward pass on new batched
        output, output_hx = self.gru(padded_x, padded_rnn_hxs)

        #Fill out the expected output by reversing the whole process
        full_out = torch.zeros((x.shape[0], x.shape[1], self.hidden_size))
        full_hx_out = torch.zeros((1, x.shape[0], self.hidden_size))

        full_out[batchable_rows] = output[:num_batchable_rows]
        full_hx_out[:, batchable_rows, :] = output_hx[:, :num_batchable_rows, :]
        # full_hx_out[batchable_rows] = output_hx[:num_batchable_rows]

        cur_row_idx = num_batchable_rows
        for i in individual_rows:
            breakpoints = (dones[i] == 1).argwhere()
            cur_idx = 0

            for breakpoint in breakpoints:
                if breakpoint == 0:
                    continue
                full_out[i, cur_idx:breakpoint, :] = output[cur_row_idx, :breakpoint-cur_idx, :]

                cur_idx = breakpoint
                cur_row_idx += 1

            if cur_idx < len(dones[i]):
                full_out[i, cur_idx:, :] = output[cur_row_idx, :len(dones[i])-cur_idx, :]
                
                #Remember only the last rnn hidden state gets returned
                full_hx_out[:, i, :] = output_hx[:, cur_row_idx, :]
                # full_hx_out[i] = output_hx[cur_row_idx]
                cur_row_idx += 1
            else:
                #If we did not have remaining steps to fill, then we must have had a done
                # on the last step, so the final rnn_hx should be zeros to return
                full_hx_out[:, i, :] = self.get_rnn_hxs()
                # full_hx_out[i] = self.get_rnn_hxs()

        return full_out, full_hx_out
            


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def linear_schedule(start_e, end_e, duration, t):
    slope = (end_e - start_e) / duration
    return max(slope*t + start_e, end_e)


class RNNQNetwork(nn.Module):
    '''RNN Network using ResettingGRU that outputs Q-values'''
    
    def __init__(self, env, hidden_size):
        super(RNNQNetwork, self).__init__()
        self.hidden_size = hidden_size
        state_size = env.observation_space.shape[0]
        
        self.relu = nn.ReLU()
        # self.gru = ResettingGRU(input_size=state_size, hidden_size=hidden_size,
        #                        batch_first=True)
        self.gru = ResettingGRUBatched(input_size=state_size, hidden_size=hidden_size,
                                       batch_first=True)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))
        

        self.fc0 = nn.Linear(hidden_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, env.action_space.n)
        
    def forward(self, state, hidden_state, dones=None, masks=None):
        '''
        Forward pass of GRU network. 
            hidden_state should have size [1, hidden_size] for unbatched or [1, N, hidden_size] for batched
            state should have size [L, input_size] for unbatched or [N, L, input_size] for batched
        return (unbatched)
            q_values [L, 1], gru_out [L, hidden_size]
        return (batched)
            q_values [N, L, 1], gru_out[N, L, hidden_size]
        '''        
        gru_out, next_hidden_state = self.gru(state, hidden_state, 
                                              dones=dones, masks=masks)
        
        x = self.relu(gru_out)
        x = self.fc0(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        q_values = x
                        
        return q_values, gru_out, next_hidden_state
    
    def get_rnn_hxs(self, num_batches=1):
        '''
        Get torch.zeros hidden states to start off with
        '''
        # if num_batches == 1:
        #     return torch.zeros(1, self.hidden_size)
        # else:
        return torch.zeros(1, num_batches, self.hidden_size)