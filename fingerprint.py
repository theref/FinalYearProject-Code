from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool
from itertools import product
import numpy as np

import axelrod as axl
from axelrod.strategy_transformers import MixedTransformer

states = [('C', 'C'), ('C', 'D'), ('D', 'C'), ('D', 'D')]
scores = {('C', 'C'): 3,
          ('C', 'D'): 0,
          ('D', 'C'): 5,
          ('D', 'D'): 1}


def expected_value(fingerprint_strat, probe_strat, turns, coords, repetitions=50, start_seed=0):
    strategies = [axl.Cooperator, axl.Defector]
    probe_strategy = MixedTransformer(coords, strategies)(probe_strat)
    players = (fingerprint_strat(), probe_strategy())
    scores = []

    for seed in range(start_seed, start_seed + repetitions):  # Repeat to deal with expectation
        axl.seed(seed)
        match = axl.Match(players, turns)  # Need to create a new match because of caching
        match.play()
        scores.append(match.final_score_per_turn()[0])

    # dist = dict(match.normalised_state_distribution())
    # try:
    #     score = sum([scores[state] * dist[state] for state in states])
    # except:
    #     score = 0

    return coords, np.mean(scores)


def fingerprint(fingerprint_strat, probe_strat, granularity, cores, turns=50, name=None):
    if name is None:
        name = fingerprint_strat.name + ".pdf"

    coordinates = list(product(np.arange(0, 1, granularity), np.arange(0, 1, granularity)))
    p = Pool(cores)

    func = partial(expected_value, fingerprint_strat, probe_strat, turns)
    scores = p.map(func, coordinates)
    scores.sort()

    xs = set([i[0][0] for i in scores])
    ys = set([i[0][1] for i in scores])
    values = np.array([i[1] for i in scores])

    clean_data = np.array(values).reshape(len(xs), len(ys))

    sns.heatmap(clean_data, xticklabels=False, yticklabels=False)
    plt.savefig(name)


fingerprint(axl.TitForTat, axl.TitForTat, 0.1, 4, 50)
