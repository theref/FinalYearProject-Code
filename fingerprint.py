from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool
from itertools import product
import numpy as np
from collections import defaultdict

import axelrod as axl
from axelrod.strategy_transformers import MixedTransformer, dual
from axelrod.interaction_utils import compute_final_score_per_turn as cfspt
from axelrod.interaction_utils import compute_normalised_state_distribution as cnsd

states = [('C', 'C'), ('C', 'D'), ('D', 'C'), ('D', 'D')]


def expected_value(fingerprint_strat, probe_strat, turns, repetitions=50,
                   warmup=0, start_seed=0, coords=None):
    """
    Calculate the expected score for a strategy and probe strategy for (x, y)
    """
    strategies = [axl.Cooperator, axl.Defector]
    probe_strategy = MixedTransformer(coords, strategies)(probe_strat)
    players = (fingerprint_strat, probe_strategy())
    scores = []
    distribution = defaultdict(int)  # If you access the defaultdict using a key, and the key is
                                    # not already in the defaultdict, the key is automatically added
                                    # with a default value. (stackoverflow)

    for seed in range(start_seed, start_seed + repetitions):  # Repeat to deal with expectation
        axl.seed(seed)
        match = axl.Match(players, turns)  # Need to create a new match because of caching
        match.play()
        interactions = match.result[warmup:]
        scores.append(cfspt(interactions)[0])  # compute_final_score_per_turn
        match_distribution = cnsd(interactions)  # compute_normalised_state_distribution

        for st in states:
            distribution[st] += match_distribution[st]

    factor = 1.0 / sum(distribution.values())
    for k in distribution:  # normalize the new dictionary
        distribution[k] = distribution[k] * factor

    mean_score = np.mean(scores)

    return coords, mean_score, distribution


def get_coordinates(granularity):
    coordinates = list(product(np.arange(0, 1, granularity), np.arange(0, 1, granularity)))
    original_coords = [x for x in coordinates if sum(x) <= 1]
    dual_coords = [x for x in coordinates if sum(x) > 1]
    return dual_coords, original_coords



def get_results(fingerprint_strat, probe_strat, granularity, cores,
                turns=50, repetitions=10, warmup=0, start_seed=0):

    dual_coords, original_coords = get_coordinates(granularity)
    p = Pool(cores)

    func = partial(expected_value, fingerprint_strat(), probe_strat, turns, repetitions, warmup,
                   start_seed)
    sim_results = p.map(func, original_coords)
    p.close()
    p.join()
    q = Pool(cores)
    dual_strat = dual(fingerprint_strat())
    dual_func = partial(expected_value, dual_strat, probe_strat, turns, repetitions, warmup,
                        start_seed)
    dual_results = q.map(dual_func, dual_coords)
    q.close()
    q.join()

    results = sim_results + dual_results
    results.sort()
    return results


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

    scores = get_results(fingerprint_strat, probe_strat, granularity, cores, turns, repetitions,
                         warmup, start_seed)

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

    results = get_results(fingerprint_strat, probe_strat, granularity, cores, turns, repetitions,
                          warmup, start_seed)

    dual_coords, original_coords = get_coordinates(granularity)
    coordinates = dual_coords + original_coords
    q = Pool(cores)
    analytical_dist = q.map(analytical_distribution_wsls, coordinates)
    analytical_dist.sort()

    final_results = []
    for i, an in enumerate(analytical_dist):
        coordinates = results[i][0]
        sim_dist = results[i][2]
        ana_dist = an[1]

        new_dict = defaultdict(tuple)
        for state, value in sim_dist.items():
            new_dict[state] = (float("%.2f" % value), float("%.2f" % ana_dist[state]))

        final_results.append((coordinates, dict(new_dict)))

    xs = sorted(list(set([i[0][0] for i in final_results])))

    table = "& "
    table += " & ".join(str(e) for e in xs) + " \\\ \n"

    for coord1 in xs:
        table += "\hline \n"
        table += "{0:.1f}".format(coord1)
        for st in states:
            for coord2 in xs:
                table += " & "
                table += "({0})".format(", ".join(str(i) for i in st))
                table += ": "
                sim_val = [dict(element[2])[st] for element in results if element[0] == (coord1, coord2)]
                ana_val = [element[1][st] for element in analytical_dist if element[0] == (coord1, coord2)]
                table += "({0:.2f}, {1:.2f})".format(sim_val[0], ana_val[0])
            table += " \\\ \n"

    info = """%% fingerprint strat - {}
              %% probe strat - {}
              %% granularity - {}
              %% cores - {}
              %% turns - {}
              %% repetitions - {}
              %% warmup - {}
              %% start seed - {}""".format(fingerprint_strat, probe_strat, granularity, cores,
                                        turns, repetitions, warmup, start_seed)

    with open("comparison.txt", 'w') as outfile:
        outfile.write(table)


def analytical_fingerprint(granularity, cores, name=None):
    dual_coords, original_coords = get_coordinates(granularity)
    coordinates = dual_coords + original_coords

    p = Pool(cores)

    scores = p.map(AnalyticalWinStayLoseShift, coordinates)
    scores.sort()

    xs = set([i[0][0] for i in scores])
    ys = set([i[0][1] for i in scores])
    values = np.array([i[1] for i in scores])
    clean_data = np.array(values).reshape(len(xs), len(ys))
    sns.heatmap(clean_data, xticklabels=False, yticklabels=False)
    plt.savefig(name)


def plot_sum_squares(fingerprint_strat, probe_strat, granularity, cores,
                     turns=50, name=None, repetitions=50, start_seed=0):

    if name is None:
        plot_name = "sum_squares.pdf"
    else:
        plot_name = name

    dual_coords, original_coords = get_coordinates(granularity)
    coordinates = dual_coords + original_coords
    warmup=0

    cc_errors = []
    cd_errors = []
    dc_errors = []
    dd_errors = []

    q = Pool(cores)
    analytical_dist = q.map(analytical_distribution_wsls, coordinates)
    analytical_dist.sort()
    for t in range(1, turns):
        results = get_results(fingerprint_strat, probe_strat, granularity, cores, t, repetitions,
                              warmup, start_seed)

        cc_e = []
        cd_e = []
        dc_e = []
        dd_e = []

        for index, value in enumerate(results):
            sim_dist = value[2]
            ana_dist = analytical_dist[index][1]
            errors = [(val - ana_dist[key])**2 for key, val in sim_dist.items()]
            cc_e.append(errors[0])
            cd_e.append(errors[1])
            dc_e.append(errors[2])
            dd_e.append(errors[3])

        cc_errors.append(np.mean([x for x in cc_e if x > 0]))
        cd_errors.append(np.mean([x for x in cd_e if x > 0]))
        dc_errors.append(np.mean([x for x in dc_e if x > 0]))
        dd_errors.append(np.mean([x for x in dd_e if x > 0]))

    plt.plot(cc_errors, label='CC errors')
    plt.plot(cd_errors, label='CD errors')
    plt.plot(dc_errors, label='DC errors')
    plt.plot(dd_errors, label='DD errors')
    plt.ylim(0, 0.01)
    plt.xlabel('Turns')
    plt.ylabel('Mean Errors Squared')
    plt.legend()
    plt.savefig(plot_name)

# fingerprint(axl.WinStayLoseShift, axl.TitForTat,
#             granularity=0.01, cores=4, turns=50, repetitions=5, warmup=0)

# analytical_fingerprint(0.01, 4, "AnalyticalWinStayLoseShift.pdf")

# state_distribution_comparison(axl.WinStayLoseShift, axl.TitForTat, granularity=0.2, cores=4,
                              # turns=200, repetitions=20, warmup=100)

plot_sum_squares(axl.WinStayLoseShift, axl.TitForTat, granularity=0.05, cores=4,
                 turns=100, repetitions=5, name="large_errors_plot.pdf")
