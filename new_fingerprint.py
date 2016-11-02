import axelrod as axl
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

    def fingerprint(self, turns=50, repetitions=10, granularity=0.01, cores=None):
        self.probe_players = self.create_probes(self.probe, granularity)
        self.edges = self.create_edges(self.probe_players)
        original = self.strategy()
        dual = DualTransformer()(self.strategy)()
        probes = self.probe_players.values()
        tourn_players = [original, dual] + list(probes)
        spatial_tourn = axl.SpatialTournament(tourn_players, turns=turns,
                                              repetitions=repetitions,
                                              edges=self.edges)
        print("Begin Spatial Tournament")
        self.results = spatial_tourn.play(processes=cores,
                                          keep_interactions=True)
        print("Spatial Tournament Finished")

    def generate_data(self, results, probe_coords):
        edge_scores = {key: cfspt(value[0])[0] for key, value in
                       results.interactions.items()}

        coord_scores = OrderedDict.fromkeys(probe_coords)
        for index, coord in enumerate(coord_scores.keys()):
            if sum(coord) > 1:
                edge = (1, index + 2)
            else:
                edge = (0, index + 2)
            coord_scores[coord] = edge_scores[edge]

        ser = pd.Series(list(coord_scores.values()),
                        index=pd.MultiIndex.from_tuples(coord_scores.keys()))
        df = ser.unstack().fillna(0)
        df.shape
        return df

    def plot(self):
        self.data = self.generate_data(self.results, self.probe_players.keys())
        sns.heatmap(self.data)
        plt.show()



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
