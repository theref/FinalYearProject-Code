import axelrod as axl
import numpy as np
from itertools import product
from axelrod.strategy_transformers import MixedTransformer
from collections import defaultdict


class Fingerprint():
    def __init__(self):
        pass

    def plot(self):
        pass


class AshlockFingerprint(Fingerprint):
    def __init__(self, strategy, probe, granularity=0.01):
        self.strategy = strategy
        self.probe = probe
        self.granularity = granularity
        self.dual_coords, self.standard_coords = self.get_coordinates()

    def get_coordinates(self):
        coordinates = list(product(np.arange(0, 1, self.granularity),
                                   np.arange(0, 1, self.granularity)))
        standard_coords = [x for x in coordinates if sum(x) <= 1]
        dual_coords = [x for x in coordinates if sum(x) > 1]
        return dual_coords, standard_coords

    def create_players():
        # create a list of the standard player, dual player,
        # and all the probes
        pass

    def create_edges():
        # create edges to connect dual/standard to correct probes
        pass

    def compute_resutls(self):
        self.create_players()
        self.create_edges()
        # compute the results
        pass

    def plot(self):
        # plot self.results
        pass

dual_coords, standard_coords = get_coordinates(granularity)
dual_probes = defaultdict()
strategies = [axl.Cooperator, axl.Defector]
probe_strat = axl.TitForTat
for coords in dual_coords:
    x, y = coords
    probe_strategy = MixedTransformer((1 - y, 1 - x), strategies)(probe_strat)
    dual_probes[coords] = probe_strategy

standard_probes = []
players = []
edges = []
spatial_tournament = axl.SpatialTournament(players, edges)
