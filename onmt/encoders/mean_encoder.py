"""Define a minimal encoder."""
from onmt.encoders.encoder import EncoderBase
from onmt.utils.misc import sequence_mask
import torch
import torch.nn as nn
from functools import partial


class MeanEncoder(EncoderBase):
    """A trivial non-recurrent encoder. Simply applies mean pooling.

    Args:
       num_layers (int): number of replicated layers
       embeddings (onmt.modules.Embeddings): embedding module to use
    """

    def __init__(self, num_layers, embeddings):
        super(MeanEncoder, self).__init__()
        self.num_layers = num_layers
        self.embeddings = embeddings

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            embeddings)

    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        emb = self.embeddings(src)
        _, batch, emb_dim = emb.size()

        if lengths is not None:
            # we avoid padding while mean pooling
            mask = sequence_mask(lengths).float()
            mask = mask / lengths.unsqueeze(1).float()
            mean = torch.bmm(mask.unsqueeze(1), emb.transpose(0, 1)).squeeze(1)
        else:
            mean = emb.mean(0)

        mean = mean.expand(self.num_layers, batch, emb_dim)
        memory_bank = emb
        encoder_final = (mean, mean)
        return encoder_final, memory_bank, lengths

class EntityMeanEncoder(EncoderBase):
    def __init__(self, num_layers, embeddings, num_entity):
        super(EntityMeanEncoder, self).__init__()
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.linear = nn.Linear(embeddings.word_vec_size * num_entity,
                                embeddings.word_vec_size)

    @classmethod
    def from_opt(cls, opt, embeddings, num_entity=28):
        """Alternate constructor. 26 players + 2 teams"""
        return cls(
            opt.enc_layers,
            embeddings,
            num_entity)

    def forward(self, src, lengths=None):
        """last hidden is a mean pool by entitiy"""
        self._check_args(src, lengths)

        emb = self.embeddings(src)
        _, batch, emb_dim = emb.size()

        # 26 players with 22 records each
        # 2 teams with 16 records each
        entity_idx = [22] * 26 + [15] * 2
        emb_by_entity = emb.split(entity_idx, dim=0) # tuple of len 28
        t_mean = partial(torch.mean, dim=0)
        mean_pool = map(t_mean, emb_by_entity)
        entity_mean = torch.cat(tuple(mean_pool), dim=1) # (64, 16800)
        entity_mean = self.linear(entity_mean) # (64, 600)
        entity_mean = entity_mean.expand(self.num_layers, batch, emb_dim)

        memory_bank = emb
        encoder_final = (entity_mean, entity_mean)
        return encoder_final, memory_bank, lengths

# TODO: player도 team별로 나누기?
class BiEntityMeanEncoder(EncoderBase):
    def __init__(self, num_layers, embeddings, num_player, num_team):
        super(BiEntityMeanEncoder, self).__init__()
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.num_entity_player = num_player
        self.num_entity_team = num_team
        self.player_linear = nn.Linear(embeddings.word_vec_size * num_player,
                                embeddings.word_vec_size)
        self.team_linear = nn.Linear(embeddings.word_vec_size * num_team,
                                     embeddings.word_vec_size)

    @classmethod
    def from_opt(cls, opt, embeddings, num_player=26, num_team=2):
        """Alternate constructor. 26 players + 2 teams"""
        return cls(
            opt.enc_layers,
            embeddings,
            num_player,
            num_team)

    def forward(self, src, lengths=None):
        """last hidden is a mean pool by team and player"""
        self._check_args(src, lengths)

        memory_bank = {}
        emb = self.embeddings(src)
        _, batch, emb_dim = emb.size()

        # 26 players with 22 records each
        # 2 teams with 16 records each
        entity_idx = [22] * 26 + [15] * 2
        emb_by_entity = emb.split(entity_idx, dim=0) # tuple of len 28
        t_mean = partial(torch.mean, dim=0)
        mean_pool = map(t_mean, emb_by_entity)

        player_mean = tuple(mean_pool)[:self.num_entity_player]
        player_mean = torch.cat(player_mean, dim=1) # (64, 16800)
        player_mean = self.player_linear(player_mean) # (64, 600)

        team_mean = tuple(mean_pool)[-self.num_entity_team:]
        team_mean = torch.cat(team_mean, dim=1)
        team_mean = self.team_linear(team_mean) # (64, 600)

        entity_mean = torch.max(torch.stack(player_mean, team_mean),
                                dim=0)[0] # max pooling
        entity_mean = entity_mean.expand(self.num_layers, batch, emb_dim)

        memory_bank['emb'] = emb
        memory_bank['player'] = player_mean
        memory_bank['team'] = team_mean
        encoder_final = (entity_mean, entity_mean)
        return encoder_final, memory_bank, lengths
