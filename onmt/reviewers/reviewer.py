import torch
import torch.nn as nn

from onmt.utils.rnn_factory import rnn_factory
from onmt.modules import GlobalAttention


class ReviewerBase(nn.Module):

    def __init__(self, review_step, review_type, rnn_type, num_layers,
                 hidden_size, dropout, attn_type="general", attn_func="softmax"):
        super(ReviewerBase, self).__init__()
        self.review_step = review_step
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

        self.state = {}

        # TODO: might not be necessary
        if review_type == 'output':
            self.linear = nn.Linear(hidden_size, hidden_size)

        self.rnn = self._build_rnn(rnn_type,
                                   input_size=self._input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   dropout=dropout)

        self.attn = GlobalAttention( # no coverage
            hidden_size, coverage=False,
            attn_type=attn_type, attn_func=attn_func
        )

    @classmethod
    def from_opt(cls, opt):
        return cls(
            opt.review_steps,
            opt.review_type,
            opt.rnn_type,
            opt.review_layers,
            opt.rnn_size,
            opt.dropout[0] if type(opt.dropout) is list
            else opt.dropout,
            opt.global_attention,
            opt.global_attention_function)

    def _build_rnn(self, rnn_type, **kwargs):
        rnn, _ = rnn_factory(rnn_type, **kwargs)
        return rnn

    def init_state(self, encoder_final):
        """Initialize reviewer state with last state of the encoder."""

        if isinstance(encoder_final, tuple):  # LSTM
            self.state["hidden"] = encoder_final
        else:  # GRU
            self.state["hidden"] = (encoder_final, )

    def forward(self, memory_bank, memory_lengths=None, step=None):
        """
        Args:
            memory_bank (FloatTensor): vectors from the encoder
                 ``(src_len, batch, hidden)``.
            memory_lengths (LongTensor): the padded source lengths
                ``(batch,)``.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * dec_outs: output from the decoder (after attn)
              ``(tgt_len, batch, hidden)``.
            * attns: distribution over src at each tgt
              ``(tgt_len, batch, src_len)``.
        """

        review_state, review_outs, attns = self._run_forward_pass(
            memory_bank, memory_lengths=memory_lengths)

        # Update the state with the result.
        if not isinstance(review_state, tuple):
            review_state = (review_state,)
        self.state["hidden"] = review_state

        # Concatenates sequence of tensors along a new dimension.
        # NOTE: v0.3 to 0.4: dec_outs / attns[*] may not be list
        #       (in particular in case of SRU) it was not raising error in 0.3
        #       since stack(Variable) was allowed.
        #       In 0.4, SRU returns a tensor that shouldn't be stacke
        if type(review_outs) == list:
            review_outs = torch.stack(review_outs).squeeze(1)

            for k in attns:
                if type(attns[k]) == list:
                    attns[k] = torch.stack(attns[k])
        return review_outs, attns


class InputReviewer(ReviewerBase):

    def _run_forward_pass(self, memory_bank, memory_lengths=None):
        review_outs = []
        attns = {}
        attns["std"] = []

        review_state = self.state["hidden"]
        for review_t in range(self.review_step):
            attn_out, p_attn = self.attn(
                review_state[0][-1],
                memory_bank.transpose(0, 1),
                memory_lengths=memory_lengths)
            attns["std"].append(p_attn)

            review_input = attn_out.unsqueeze(0)
            review_output, review_state = self.rnn(review_input, review_state)

            review_outs += [self.dropout(review_output)]
        return review_state, review_outs, attns

    @property
    def _input_size(self):
        return self.hidden_size

class OutputReviewer(ReviewerBase):

    def _run_forward_pass(self, memory_bank, memory_lengths=None):
        attns = {}
        # memory bank (src_len, batch, size)
        src_len, batch, _ = memory_bank.size()
        zero_input = torch.zeros(batch, self.review_step, 0)

        if isinstance(self.rnn, nn.GRU):
            review_output, review_state = self.rnn(zero_input, self.state["hidden"][0])
        else:
            review_output, review_state = self.rnn(zero_input, self.state["hidden"])

        review_outs, p_attn = self.attn(
            review_output.transpose(0, 1).contiguous(),
            memory_bank.transpose(0, 1),
            memory_lengths=memory_lengths
        )
        attns["std"] = p_attn

        review_outs = self.dropout(review_outs) # (review_step, batch, dim)
        review_outs = review_output + self.linear(review_outs)
        return review_state, review_outs, attns

    # FIXME: this seems to raise CUDNN BAD PARAM error
    @property
    def _input_size(self):
        return 0
