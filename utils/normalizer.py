import numpy as np

class RunningStatNormalizer:
    """
    Implements running statistics (mean, std) normalization using Welford's algorithm.
    This is used to normalize observations to have a mean of ~0 and a std dev of ~1.
    """
    def __init__(self, shape, epsilon=1e-8):
        """
        Initializes the normalizer.

        Args:
            shape (tuple): The shape of the data to be normalized (e.g., env.observation_space.shape).
            epsilon (float): A small value to add to the standard deviation to prevent division by zero.
        """
        self.shape = shape
        self._epsilon = epsilon

        # Running statistics
        self.n = 0  # Count of data points seen
        self.mean = np.zeros(shape, dtype=np.float64)
        self.M2 = np.zeros(shape, dtype=np.float64) # Sum of squares of differences from the mean
        self.std = np.ones(shape, dtype=np.float64) # Standard deviation

    def update(self, x: np.ndarray):
        """
        Updates the running mean and standard deviation with a new data point.
        Uses Welford's algorithm for numerical stability.

        Args:
            x (np.ndarray): A new data point (e.g., a new observation).
        """
        x = np.asarray(x)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean # The new mean
        self.M2 += delta * delta2

        # Update the standard deviation after enough samples have been seen
        if self.n > 1:
            variance = self.M2 / self.n
            self.std = np.sqrt(variance)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Normalizes a data point using the current running statistics.

        Args:
            x (np.ndarray): The data point to normalize.

        Returns:
            np.ndarray: The normalized data point.
        """
        # The formula for standardization (Z-score)
        return (x - self.mean) / (self.std + self._epsilon)

    def state_dict(self):
        """Returns the internal state of the normalizer."""
        return {
            'n': self.n,
            'mean': self.mean,
            'M2': self.M2,
            'std': self.std
        }

    def load_state_dict(self, state_dict):
        """Restores the internal state."""
        self.n = state_dict['n']
        self.mean = state_dict['mean']
        self.M2 = state_dict['M2']
        self.std = state_dict['std']
