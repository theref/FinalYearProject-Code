from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool
from itertools import product
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import axelrod as axl
from axelrod.strategy_transformers import MixedTransformer
from axelrod.interaction_utils import compute_final_score_per_turn as cfspt
from axelrod.interaction_utils import compute_normalised_state_distribution as cnsd


def expected_value(fingerprint_strat, probe_strat, turns, repetitions=50,
                   warmup=0, start_seed=0, coords=None):
    """
    Calculate the expected score for a strategy and probe strategy for (x, y)
    """
    strategies = [axl.Cooperator, axl.Defector]
    probe_strategy = MixedTransformer(coords, strategies)(probe_strat)
    players = (fingerprint_strat(), probe_strategy())
    scores = []
    distribution = defaultdict(int)  # If you access the defaultdict using a key, and the key is not
                                     # already in the defaultdict, the key is automatically added 
                                     # with a default value. (stackoverflow)

    for seed in range(start_seed, start_seed + repetitions):  # Repeat to deal with expectation
        axl.seed(seed)
        match = axl.Match(players, turns)  # Need to create a new match because of caching
        match.play()
        interactions = match.result[warmup:]
        scores.append(cfspt(interactions)[0])  # compute_final_score_per_turn
        match_distribution = cnsd(interactions)  # compute_normalised_state_distribution

        for key, value in match_distribution.items():
            distribution[key] += value

    factor = 1.0 / sum(distribution.values())
    for k in distribution:  # normalize the new dictionary
        distribution[k] = distribution[k] * factor

    mean_score = np.mean(scores)

    return coords, mean_score, distribution


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

    func = partial(expected_value, fingerprint_strat, probe_strat, turns,
                   repetitions, warmup, start_seed)
    scores = p.map(func, tqdm(coordinates))
    scores.sort()

    xs = set([i[0][0] for i in scores])
    ys = set([i[0][1] for i in scores])
    values = np.array([i[1] for i in scores])
    clean_data = np.array(values).reshape(len(xs), len(ys))
    sns.heatmap(clean_data, xticklabels=False, yticklabels=False)
    plt.savefig(name)


def analytical_distribution_wsls(coords):
    x = coords[0]
    y = coords[1]
    distribution = {('C', 'C'): x * (1 - x),
                    ('D', 'C'): y * (1 - y),
                    ('D', 'D'): y * (1 - x),
                    ('C', 'D'): y * (1 - x)
                    }

    factor = 1.0 / (2 * y * (1 - x) + x * (1 - x) + y * (1 - y))
    for k in distribution:
        distribution[k] = distribution[k] * factor

    return coords, distribution


def state_distribution_comparison(fingerprint_strat, probe_strat, granularity, cores,
                                  turns=50, repetitions=50, warmup=0, start_seed=0):

    coordinates = list(product(np.arange(0, 1, granularity), np.arange(0, 1, granularity)))
    p = Pool(cores)

    func = partial(expected_value, fingerprint_strat, probe_strat, turns,
                   repetitions, warmup, start_seed)
    sim_results = p.map(func, tqdm(coordinates))
    sim_results.sort()

    q = Pool(cores)
    analytical_dist = q.map(analytical_distribution_wsls, coordinates)
    analytical_dist.sort()

    final_results = []
    for i, an in enumerate(analytical_dist):
        coordinates = sim_results[i][0]
        sim_dist = sim_results[i][2]
        ana_dist = an[1]

        new_dict = defaultdict(tuple)
        for state, value in sim_dist.items():
            new_dict[state] = (round(value, ndigits=3), round(ana_dist[state], ndigits=3))

        final_results.append((coordinates, dict(new_dict)))

    xs = set([i[0][0] for i in final_results])
    ys = set([i[0][1] for i in final_results])
    values = np.array([i[1] for i in final_results])
    clean_data = np.array(values).reshape(len(xs), len(ys))

    table = " & ".join(str(e) for e in xs) + "\n"
    for i, row in enumerate(clean_data):
        table += "{} & ".format(list(xs)[i])
        table += " & ".join(map(str, row))
        table += "\n"

    with open("test.txt", 'w') as outfile:
        outfile.write(table)


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


# fingerprint(axl.WinStayLoseShift, axl.TitForTat, granularity=0.01, cores=4, turns=50,
            # repetitions=10, warmup=0)

# analytical_fingerprint(0.01, 4, "AnalyticalWinStayLoseShift.pdf")

state_distribution_comparison(axl.WinStayLoseShift, axl.TitForTat, granularity=0.25, cores=4,
                              turns=50, repetitions=10, warmup=0)
