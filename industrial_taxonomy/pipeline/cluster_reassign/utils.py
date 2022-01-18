import numpy as np


class Groupby:
    """Performs groupby and apply

    Args:
        keys (np.array): 1-D array of key values.
        values (np.array): 1-D array of values. Must be same length as `keys`.
    """

    def __init__(self, keys, values):
        self.keys = keys
        self.values = values

    def groupby_apply(self, function):
        """Groups values by each key according to their position and applies
        `function` to each group.

        Args:
            function: A function to apply to the values corresponding to each
                key. For example, `np.mean`, would return the mean of values
                for each key.

        Returns:
            2-D array containing unique keys in the first column and the
            corresponding aggregated values in the second column.
        """
        unique_keys = np.unique(self.keys)
        agg = []
        for k in unique_keys:
            v = self.values[self.keys == k]
            agg.append(function(v))
        agg = np.array([unique_keys, np.array(agg)])
        return agg
