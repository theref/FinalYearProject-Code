from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool
from itertools import product
import numpy as np

import axelrod as axl
from axelrod.strategy_transformers import MixedTransformer
from axelrod.interaction_utils import compute_final_score_per_turn as cfspt
from axelrod.interaction_utils import compute_normalised_state_distribution as cnsd


def expected_value(fingerprint_strat, probe_strat, turns, repetitions=50, warmup=0, start_seed=0, coords=None):
    """
    Calculate the expected score for a strategy and probe strategy for (x, y)
    """
    strategies = [axl.Cooperator, axl.Defector]
    probe_strategy = MixedTransformer(coords, strategies)(probe_strat)
    players = (fingerprint_strat(), probe_strategy())
    scores = []

    for seed in range(start_seed, start_seed + repetitions):  # Repeat to deal with expectation
        axl.seed(seed)
        match = axl.Match(players, turns)  # Need to create a new match because of caching
        match.play()
        interactions = match.result[warmup:]
        scores.append(cfspt(interactions)[0])
        distribution = cnsd(interactions)

    return coords, np.mean(scores), distribution


def AnalyticalWinStayLoseShift(coords):
    """
    The analytical function for Win-Stay-Lose-Shift (Pavlov) being probed by
    Tit-For-Tat
    """
    x = coords[0]
    y = coords[1]
    numerator = 3 * x * (x - 1) + y * (x - 1) + 5 * y * (y - 1)
    denominator = (2 * y * (x - 1) + x * (x - 1) + y * (y - 1))
    value = numerator / denominator
    return coords, value


def fingerprint(fingerprint_strat, probe_strat, granularity, cores,
                turns=50, name=None, repetitions=50, warmup=0, start_seed=0):
    """
    Produces a fingerprint plot for a strategy and probe strategy
    """
    if name is None:
        name = fingerprint_strat.name + ".pdf"

    coordinates = list(product(np.arange(0, 1, granularity), np.arange(0, 1, granularity)))
    p = Pool(cores)

    func = partial(expected_value, fingerprint_strat, probe_strat, turns, repetitions, warmup, start_seed)
    scores = p.map(func, coordinates)
    scores.sort()

    xs = set([i[0][0] for i in scores])
    ys = set([i[0][1] for i in scores])
    values = np.array([i[1] for i in scores])
    clean_data = np.array(values).reshape(len(xs), len(ys))
    sns.heatmap(clean_data, xticklabels=False, yticklabels=False)
    plt.savefig(name)


def analytical_fingerprint(granularity, cores, name=None):
    coordinates = list(product(np.arange(0, 1, granularity), np.arange(0, 1, granularity)))

    p = Pool(cores)

    scores = p.map(AnalyticalWinStayLoseShift, coordinates)
    scores.sort()

    xs = set([i[0][0] for i in scores])
    ys = set([i[0][1] for i in scores])
    values = np.array([i[1] for i in scores])
    clean_data = np.array(values).reshape(len(xs), len(ys))
    sns.heatmap(clean_data, xticklabels=False, yticklabels=False)
    plt.savefig(name)


fingerprint(axl.WinStayLoseShift, axl.TitForTat, granularity=0.01, cores=4, turns=50,
            repetitions=10, warmup=0)

# analytical_fingerprint(0.01, 4, "AnalyticalWinStayLoseShift.pdf")
# print(expected_value(axl.WinStayLoseShift, axl.TitForTat, 500, (0.5, 0.5), repetitions=50, start_seed=0))
