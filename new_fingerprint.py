import axelrod as axl
import numpy as np
import pandas as pd
from itertools import product
from axelrod.strategy_transformers import JossAnnTransformer, DualTransformer
from axelrod.interaction_utils import compute_final_score_per_turn as cfspt
from collections import OrderedDict


class Fingerprint():
    def __init__(self):
        pass

    def plot(self):
        pass


class AshlockFingerprint(Fingerprint):
    def __init__(self, strategy, probe):
        self.strategy = strategy
        self.probe = probe

    def create_probe_coords(self, gran):
        coordinates = list(product(np.arange(0, 1, gran),
                                   np.arange(0, 1, gran)))

        probe_coords = OrderedDict.fromkeys(coordinates)
        return probe_coords

    def create_probes(self, probe, granularity):
        """

        """
        probe_dict = self.create_probe_coords(granularity)
        for coord in probe_dict.keys():
            x, y = coord
            if x + y > 1:
                probe_dict[coord] = JossAnnTransformer((1 - y, 1 - x))(probe)()
            else:
                probe_dict[coord] = JossAnnTransformer((x, y))(probe)()
        return probe_dict

    def create_edges(self, probe_players):
        edges = []
        for index, coord in enumerate(probe_players.keys()):
            if sum(coord) > 1:
                edge = (1, index + 2)
            else:
                edge = (0, index + 2)
            edges.append(edge)
        return edges

    def fingerprint(self, granularity):
        self.probe_players = self.create_probes(self.probe, granularity)
        self.edges = self.create_edges(self.probe_players)
        original = self.strategy()
        dual = DualTransformer()(self.strategy)()
        probes = self.probe_players.values()
        tourn_players = [original, dual] + list(probes)
        spatial_tourn = axl.SpatialTournament(tourn_players, turns=2,
                                              repetitions=2, edges=self.edges)
        self.results = spatial_tourn.play(keep_interactions=True)

    def plot(self):
        # edge_scores = {key: cfspt(value[0]) for key, value in
        #                self.results.interactions.items()}

        # ser = pd.Series(list(dict_sim_scores.values()),
        #           index=pd.MultiIndex.from_tuples(dict_sim_scores.keys()))
        # df = ser.unstack().fillna(0)
        # df.shape


# dual_probes = defaultdict()
# strategies = [axl.Cooperator, axl.Defector]
# probe_strat = axl.TitForTat
# for coords in dual_coords:
#     x, y = coords
#     probe_strategy = MixedTransformer((1 - y, 1 - x), strategies)(probe_strat)
#     dual_probes[coords] = probe_strategy

# standard_probes = []
# players = []
# edges = []
# spatial_tournament = axl.SpatialTournament(players, edges)
