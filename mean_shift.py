import numpy as np
import random


class MeanShift(object):
    """ Simple implementation of the Mean Shift algorithm

    Extra addition is that the cluster center history is stored, which is very
    useful for visualisation.

    For a more complete implementation of Mean Shift see Sklearn:
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/cluster/
        mean_shift_.py
    """
    def __init__(self, bandwidth=0.1, n_seeds=None, max_iter=100):
        """ Set up the parameters for clustering

        :param bandwidth: Width of the kernel used (float)
        :param n_seeds: Amount of initial cluster centers (int). Default is
            None, in which case n_seeds == points in training set
        :param max_iter: Maximum amount of iterations for the algorithm (int)

        self.cluster_center_history is a list of numpy arrays. Each iteration
        the location of the cluster centers is stored.
        """
        self.bandwidth = bandwidth
        self.n_seeds = n_seeds
        self.max_iter = max_iter

        self.cluster_centers = None
        self.cluster_center_history = []

    def fit(self, x):
        """ Fit clusters to data provided

        :param x: Data to fit clusters to (float numpy array)
        :return: List of cluster center locations (float numpy array)
        """
        # TODO: Perform more input checks
        n_points, n_dimensions = x.shape
        if self.n_seeds > n_points:
            msg = "n_seeds ({}) cannot be higher than the amount of points in" \
                  "the dataset ({})"
            msg = msg.format(str(self.n_seeds), str(n_points))
            raise ValueError(msg)

        # Seed centers
        if self.n_seeds is None:
            self.cluster_centers = x
        else:
            cluster_centers_idx = random.sample(range(n_points), self.n_seeds)
            self.cluster_centers = x[cluster_centers_idx]
        self.cluster_center_history.append(self.cluster_centers)

        # Iterate until stable: update centers
        for iteration in range(1, self.max_iter+1):
            new_centers = []
            for center in self.cluster_centers:
                # For each center: determine points within bandwidth
                within_bandwidth = np.sum((x - center) ** 2, axis=1) \
                    < self.bandwidth ** 2

                # the mean of those points is the new center
                new_center = np.mean(x[within_bandwidth], axis=0)
                new_centers.append(new_center)

            # copy centers to history
            new_centers = np.array(new_centers)
            self.cluster_centers = new_centers
            self.cluster_center_history.append(self.cluster_centers)

            print "Iteration: ", iteration
            if self._is_converged():
                print "Clusters converged"
                break
        else:
            print "Max iterations reached"

        # TODO: Optional: remove near identical centers

        return self.cluster_centers

    def _is_converged(self):
        """ Helper function to determine if the clusters are converged

        :return: If clusters are stable or not (bool)
        """
        same_per_dim = (self.cluster_center_history[-1] ==
                        self.cluster_center_history[-2])
        return same_per_dim.all()
