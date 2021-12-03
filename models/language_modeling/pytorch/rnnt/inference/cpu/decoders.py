# Copyright (c) 2018 Intel Corporation
# Copyright (c) 2019, Myrtle Software Limited. All rights reserved.
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

import torch.nn.functional as F
from model_rnnt import label_collate

import time

class TransducerDecoder:
    """Decoder base class.

    Args:
        alphabet: An Alphabet object.
        blank_symbol: The symbol in `alphabet` to use as the blank during CTC
            decoding.
        model: Model to use for prediction.
    """

    def __init__(self, blank_index, model):
        self._model = model
        self._SOS = -1   # start of sequence
        self._blank_id = blank_index

    def _pred_step(self, label, hidden, device):
        if label == self._SOS:
            return self._model.predict(None, hidden, add_sos=False, device=device)
        if label > self._blank_id:
            label -= 1
        label = label_collate([[label]]).to(device)
        return self._model.predict(label, hidden, add_sos=False, device=device)

    def _joint_step(self, enc, pred, log_normalize=False):
        logits = self._model.joint(enc, pred)[:, 0, 0, :]
        if not log_normalize:
            return logits

        probs = F.log_softmax(logits, dim=len(logits.shape) - 1)

        return probs

    def _get_last_symb(self, labels):
        return self._SOS if labels == [] else labels[-1]

def _update_batch(dim, max_lens, max_symbols, _SOS, blankness, blank_vec, x, hidden_prime, hidden, label_col, label_row, label_tensor, symbols_added, time_idxs, f, k):
    # ones = torch.ones_like(label_col)
    
    symbols_added *= blankness.logical_not()
    tmp_blank_vec = blank_vec.logical_or(blankness)

    # If for sample blankid already encountered, then stop
    # update hidden values until input from next time step.
    # So we would mix value of hidden and hidden_prime together,
    # keep values in hidden where blank_vec[i] is true

    if hidden == None:
        hidden = [torch.zeros_like(hidden_prime[0]), torch.zeros_like(hidden_prime[1])]

    not_blank = tmp_blank_vec.eq(0)
    
    idx = (not_blank).nonzero(as_tuple=True)[0]

    hidden[0][:, idx, :] = hidden_prime[0][:, idx, :]
    hidden[1][:, idx, :] = hidden_prime[1][:, idx, :]

    label_col += not_blank
    label_tensor.index_put_([label_row, label_col], (k-_SOS)*not_blank, accumulate=True)

    symbols_added += not_blank

    need_add = symbols_added.ge(max_symbols)

    time_idxs += need_add
    blankness.logical_or_(need_add)
    symbols_added *= symbols_added.lt(max_symbols)

    # update f if necessary
    # if at least one id in blankness is blank them time_idx is updated
    # and we need to update f accordingly

    # TODO nonzero()
    # if blankness.count_nonzero() > 0:
    if blankness.nonzero().size(0) > 0:
        fetch_time_idxs = time_idxs.min(max_lens)
        # select tensor along second dim of x
        # implement something like --> f = x[:, fetch_time_idxs, :].unsqueeze(1)
        # for example, if all elements in fetch_time_idxs = n, then
        # this is equivelent to f = x[:, n, :].unsqueeze(1)
        f = x[list(range(x.size(0))), fetch_time_idxs, :].unsqueeze(1)

    return hidden, label_tensor, label_col, f, time_idxs, symbols_added

class RNNTGreedyDecoder(TransducerDecoder):
    """A greedy transducer decoder.

    Args:
        blank_symbol: See `Decoder`.
        model: Model to use for prediction.
        max_symbols_per_step: The maximum number of symbols that can be added
            to a sequence in a single time step; if set to None then there is
            no limit.
        cutoff_prob: Skip to next step in search if current highest character
            probability is less than this.
    """
    def __init__(self, blank_index, model, max_symbols_per_step=30):
        super().__init__(blank_index, model)
        assert max_symbols_per_step is None or max_symbols_per_step > 0
        self.max_symbols = max_symbols_per_step

    def decode(self, x, out_lens, args, conf=None):
        """Returns a list of sentences given an input batch.

        Args:
            x: A tensor of size (batch, channels, features, seq_len)
                TODO was (seq_len, batch, in_features).
            out_lens: list of int representing the length of each sequence
                output sequence.

        Returns:
            list containing batch number of sentences (strings).
        """
        if args.ipex:
            import intel_extension_for_pytorch as ipex
        with torch.no_grad():
            if args.print_time:
                t0 = time.time()
            # Apply optional preprocessing
            logits, out_lens = self._model.encode((x, out_lens))

            # output = []
            # for batch_idx in range(logits.size(0)):
            #     inseq = logits[batch_idx, :, :].unsqueeze(1)
            #     logitlen = out_lens[batch_idx]
            #     sentence = self._greedy_decode(inseq, logitlen)
            #     output.append(sentence)
            
            if args.print_time:
                t1 = time.time()

            if args.ipex:
                output = self._greedy_decode_batch(logits, out_lens, args.ipex)
            else:
                output = self._greedy_decode_batch_origin(logits, out_lens, args.ipex)

            if args.print_time:
                print("encoder time: {0}, decoder time: {1}".format(t1 - t0, time.time() - t1))

        return output

    def _greedy_decode(self, x, out_len):
        training_state = self._model.training
        self._model.eval()

        device = x.device

        hidden = None
        label = []
        for time_idx in range(out_len):
            f = x[time_idx, :, :].unsqueeze(0)

            not_blank = True
            symbols_added = 0

            while not_blank and (
                    self.max_symbols is None or
                    symbols_added < self.max_symbols):
                g, hidden_prime = self._pred_step(
                    self._get_last_symb(label),
                    hidden,
                    device
                )
                logp = self._joint_step(f, g, log_normalize=False)[0, :]

                # get index k, of max prob
                v, k = logp.max(0)
                k = k.item()

                if k == self._blank_id:
                    not_blank = False
                else:
                    label.append(k)
                    hidden = hidden_prime
                symbols_added += 1

        self._model.train(training_state)
        return label

    def _greedy_decode_batch_origin(self, x, out_lens, ipex):
        batch_size = x.size(0)
        hidden = None
        max_len = out_lens.max()
        max_lens = torch.tensor([max_len-1 for i in range(batch_size)], dtype=torch.int64)
        # pos 0 of label_tensor is set to _SOS to simplify computation
        # real label start from pos 1
        label_tensor = torch.tensor([self._SOS]).repeat(batch_size, max_len*self.max_symbols)
        # (row, col) of current labels end
        label_row = torch.tensor([i for i in range(batch_size)])
        label_col = torch.tensor([0 for i in range(batch_size)])
        # this list will be used to return labels to caller
        label_copy = [0 for i in range(batch_size)]
        # initially time_idx is 0 for all input
        # then advance time_idx for each 'track' when needed and update f
        f = x[:, 0, :].unsqueeze(1)
        time_idxs = torch.tensor([0 for i in range(batch_size)], dtype=torch.int64)

        not_blank = True
        blank_vec = torch.tensor([False for i in range(batch_size)])
        symbols_added = torch.tensor([0 for i in range(batch_size)], dtype=torch.int64)

        while True:
            g, hidden_prime = self._pred_step_batch(
                label_tensor.gather(1, label_col.unsqueeze(1)),
                hidden,
                batch_size,
                ipex
            )
            logp = self._joint_step_batch(f, g, log_normalize=False)

            # get index k, of max prob
            v, k = logp.max(1)

            # if any of the output is blank, pull in the next time_idx for next f
            # tmp_blank_vec is the vect used to mix new hidden state with previous hidden state
            # blank_vec is the baseline of blank_vec, it turns to blank only when run out of time_idx
            blankness = k.eq(self._blank_id)
            time_idxs = time_idxs + blankness
            # it doesn't matter if blank_vec is update now or later,
            # tmp_blank_vec always get correct value for this round
            blank_vec = time_idxs.ge(out_lens)

            # TODO count_nonzero()
            # if blank_vec.count_nonzero() == batch_size:
            if blank_vec.nonzero().size(0) == batch_size:
                # all time_idxs processed, stop
                break
            else:
                # if ipex:
                #     import intel_extension_for_pytorch as ipex
                #     _update_batch = ipex.rnnt_update_batch

                hidden, label_tensor, label_col, f, time_idxs, symbols_added = _update_batch(
                    f.size()[2], 
                    max_lens,
                    self.max_symbols,
                    self._SOS,
                    blankness,
                    blank_vec,
                    x,
                    hidden_prime,
                    hidden,
                    label_col,
                    label_row,
                    label_tensor,
                    symbols_added,
                    time_idxs,
                    f,
                    k)

        for i in range(batch_size):
            label_copy[i]=label_tensor[i][1:label_col[i]+1].tolist()
        return label_copy        

    def _greedy_decode_batch(self, x, out_lens, ipex):
        batch_size = x.size(0)
        hidden = None
        max_len = out_lens.max()
        not_blank = True
        # initially time_idx is 0 for all input
        # then advance time_idx for each 'track' when needed and update f
        f = x[:, 0, :].unsqueeze(1)
        # this list will be used to return labels to caller
        label_copy = [0 for i in range(batch_size)]

        # pos 0 of label_tensor is set to _SOS to simplify computation
        # real label start from pos 1
        label_tensor = torch.empty((batch_size, max_len*self.max_symbols), dtype=torch.long).fill_(self._SOS)
        # TODO: use custom kernel to init tensors
        # label_tensor = torch.empty((batch_size, max_len*self.max_symbols), dtype=torch.long)
        # torch.ops.torch_ipex.rnnt_tensor_init(label_tensor, self._SOS, batch_size*max_len*self.max_symbols)
        
        # (row, col) of current labels end
        label_col = torch.zeros((batch_size), dtype=torch.int)
        time_idxs = torch.zeros((batch_size), dtype=torch.int)
        blank_vec = torch.zeros((batch_size), dtype=torch.int)
        symbols_added = torch.zeros((batch_size), dtype=torch.int)
        blankness = torch.zeros((batch_size), dtype=torch.int)
        not_blank = torch.zeros((batch_size), dtype=torch.int)
        label_to_put = torch.zeros((batch_size), dtype=torch.long)

        label_for_next_loop = torch.tensor([self._SOS for i in range(batch_size)], dtype=torch.long)
        while True:
            g, hidden_prime = self._pred_step_batch(
                label_for_next_loop.unsqueeze(1),
                hidden,
                batch_size,
                ipex
            )
            logp = self._joint_step_batch(f, g, log_normalize=False)

            # get index k, of max prob
            v, k = logp.max(1)

            if hidden == None:
                hidden = [torch.zeros_like(hidden_prime[0]), torch.zeros_like(hidden_prime[1])]

            # out_lens dtype int32 or int64?
            finished = torch.ops.torch_ipex.rnnt_update_batch(
                k,
                out_lens,
                label_col,
                symbols_added,
                time_idxs,
                blankness,
                blank_vec,
                not_blank,
                label_to_put,
                label_tensor,
                label_for_next_loop,
                hidden[0],
                hidden[1],
                hidden_prime[0],
                hidden_prime[1],
                x,
                f,
                self.max_symbols,
                self._blank_id,
                batch_size,
                self._SOS,
                max_len)


            if finished:
                # all time_idxs processed, stop
                break

        for i in range(batch_size):
            label_copy[i]=label_tensor[i][1:label_col[i]+1].tolist()
        return label_copy

    def _pred_step_batch(self, label, hidden, batch_size, ipex):
        # not really need this line, _blank_id is the last id of dict
        #label = label - label.gt(self._blank_id).int()
        if ipex:
            result = self._model.predict_batch(label, hidden, batch_size)
        else:
            result = self._model.predict_batch_origin(label, hidden, batch_size)
        return result

    def _joint_step_batch(self, enc, pred, log_normalize=False):
        logits = self._model.joint_inference(enc, pred)
        # In inference, logits is 3d: (B, T=1, H)
        logits = logits[:, 0, :]
        if not log_normalize:
            return logits

        probs = F.log_softmax(logits, dim=len(logits.shape) - 1)

        return probs
