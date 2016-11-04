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
        """
        Parameters
        ----------
        strategy : class
            A class that must be descended from axelrod.strategies.
        probe : class
            A class that must be descended from axelrod.strategies
        """
        self.strategy = strategy
        self.probe = probe

    def create_probe_coords(self, granularity):
        """Creates a set of coordinates over a 1x1 grid.

        Construncts (x, y) coordinates that are separated by a step equal to
        `granularity`. The coordinates are over a 1x1 grid which implies that
        the number of points created will be 1/`granularity`^2.

        Parameters
        ----------
        granularity : float
            The seperation between each coordinate. Smaller granularity will
            produce more coordinates that will be closer together.

        Returns
        ----------
        probe_coords : ordered dictionary
            An Ordered Dictionary where the keys are tuples representing each
            coordinate, eg. (x, y). The value is automatically set to `None`.
        """
        coordinates = list(product(np.arange(0, 1, granularity),
                                   np.arange(0, 1, granularity)))

        probe_coords = OrderedDict.fromkeys(coordinates)
        return probe_coords

    def create_probes(self, probe, granularity):
        """Creates a set of probe strategies over a 1x1 grid.

        Construncts probe strategies that correspond to (x, y) coordinates. The
        precision of the coordinates is determined by `granularity`. The probes
        are created using the `JossAnnTransformer`.

        Parameters
        ----------
        granularity : float
            The seperation between each coordinate. Smaller granularity will
            produce more coordinates that will be closer together.
        probe : class
            A class that must be descended from axelrod.strategies.

        Returns
        ----------
        probe_dict : ordered dictionary
            An Ordered Dictionary where the keys are tuples representing each
            coordinate, eg. (x, y). The value is a `JossAnnTransformer` with
            parameters that correspond to (x, y).
        """
        probe_dict = self.create_probe_coords(granularity)
        for coord in probe_dict.keys():
            x, y = coord
            if x + y > 1:
                probe_dict[coord] = JossAnnTransformer((1 - y, 1 - x))(probe)()
            else:
                probe_dict[coord] = JossAnnTransformer((x, y))(probe)()
        return probe_dict

    def create_edges(self, probe_dict):
        """Creates a set of edges for a spatial tournament.

        Construncts edges that correspond to the probes in `probe_dict`. Probes
        whose coordinates sum to less/more than 1 will have edges that link them
        to 0/1 correspondingly.

        Parameters
        ----------
        probe_dict : ordered dictionary
            An Ordered Dictionary where the keys are tuples representing each
            coordinate, eg. (x, y). The value is a `JossAnnTransformer` with
            parameters that correspond to (x, y).

        Returns
        ----------
        edges : list of tuples
            A list containing tuples of length 2. All tuples will have either 0
            or 1 as the first element. The second element is the index of the
            corresponding probe (+2 to allow for including the Strategy and it's
            Dual).
        """
        edges = []
        for index, coord in enumerate(probe_dict.keys()):
            #  Add 2 to the index because we will have to allow for the Strategy
            #  and it's Dual
            if sum(coord) > 1:
                edge = (1, index + 2)
            else:
                edge = (0, index + 2)
            edges.append(edge)
        return edges

    def fingerprint(self, turns=50, repetitions=10, granularity=0.01, cores=None):
        """Build and play a spatial tournament.

        Creates the probes and their edges then builds a spatial tournament
        where the original strategy only plays probes whose coordinates sum to
        less than 1 (or equal). Probes whose coordinates sum to more than 1 play
        the Dual Strategy.

        Parameters
        ----------
        turns : integer, optional
            The number of turns per match
        repetitions : integer, optional
            The number of times the round robin should be repeated
        granularity : float, optional
            The seperation between each coordinate. Smaller granularity will
            produce more coordinates that will be closer together.
        cores : integer, optional
            The number of processes to be used for parallel processing
        """
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

    def _generate_data(self, results, probe_coords):
        """Generates useful data from a spatial tournament.

        Matches interactions from `results` to their corresponding coordinate in
        `probe_coords`.

        Parameters
        ----------
        results : axelrod.result_set.ResultSetFromFile
            A results set for a spatial tournament.
        probe_coords : list of tuples
            A list of tuples of length 2, where each tuple represents a
            coordinate, eg. (x, y).

        Returns
        ----------
        data_frame : pandas.core.frame.DataFrame
            A pandas DataFrame object where the row and column headers
            correspond to coordinates. The cell values are the score of the
            original/dual strategy playing the probe with parameters that match
            the coordinate.
        """
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
        data_frame = ser.unstack().fillna(0)
        data_frame.shape
        return data_frame

    def plot(self):
        self.data = self._generate_data(self.results, self.probe_players.keys())
        sns.heatmap(self.data)
        plt.show()
