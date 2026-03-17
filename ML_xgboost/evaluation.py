import pandas as pd
import numpy as np

class BinaryClassifierEvaluator:
    """Evaluator for binary classification models.

    Computes discrimination metrics — KS statistic, Gini coefficient, and AUC —
    commonly used in credit risk scorecard development.

    Examples
    --------
    >>> evaluator = BinaryClassifierEvaluator()
    >>> df = BinaryClassifierEvaluator.gen_test_data()
    >>> cdf, auc, gini, ks = evaluator.gen_metrics(df, label='y', scr='x')
    """

    @staticmethod
    def gen_test_data(n=10000, seed=42):
        """Generate synthetic binary classification data for testing.

        Simulates a logistic model where the predictor ``x`` is drawn from
        N(2, 1) and the binary label ``y`` is sampled from the resulting
        Bernoulli probabilities.

        Parameters
        ----------
        n : int, default=10000
            Number of observations to generate. Must be between 10 and
            10,000,000; falls back to 10000 if out of range.
        seed : int, default=42
            Random seed for reproducibility. Must be a non-negative integer
            no greater than 2**32 - 1; falls back to 42 if invalid.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns:

            - ``x`` : float, the continuous predictor.
            - ``y`` : int (0 or 1), the binary target label.

        Examples
        --------
        >>> df = BinaryClassifierEvaluator.gen_test_data(n=5000, seed=0)
        >>> df.shape
        (5000, 3)
        >>> df.columns.tolist()
        ['x', 'y', 'weight']
        """
        if not isinstance(seed, int) or not (0 <= seed <= 2 ** 32 - 1):
            seed = 42
        if not isinstance(seed, int) or not (10 <= n <= 10_000_000):
            n = 10000
        np.random.seed(seed)
        x = np.random.normal(2, 1, n)
        prob = 1 / (1 + np.exp(-(-4.2 + x)))
        y = np.random.binomial(1, prob, n)
        df = pd.DataFrame({'x': x, 'y': y})
        df['weight'] = np.random.uniform(0.5, 2.0, len(df))
        return df

    @staticmethod
    def _valid_df(df, label, scr, w=None):
        """Validate inputs before computing metrics.

        Parameters
        ----------
        df : pandas.DataFrame
            Input data containing at minimum the label and score columns.
        label : str
            Name of the binary target column. Must contain only 0 and 1.
        scr : str
            Name of the score or predicted probability column. Must be numeric.
        w : str or None, default=None
            Name of the sample weight column. Must be numeric if provided.

        Raises
        ------
        TypeError
            If ``df`` is not a :class:`pandas.DataFrame`, or if ``scr`` or
            ``w`` columns are not numeric.
        ValueError
            If required columns are missing, contain null values, or if
            ``label`` is not a binary column with values in {0, 1}.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")

        cols = [label, scr]
        missing_cols = [c for c in cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in df: {missing_cols}.")
        missing_vals = [c for c in cols if df[c].isna().any()]
        if missing_vals:
            raise ValueError(f"Columns contain missing values: {missing_vals}.")

        if not pd.api.types.is_numeric_dtype(df[label]) or set(df[label].unique()) - {0, 1}:
            raise ValueError(f"'{label}' must be binary with values in {{0, 1}}.")
        if not pd.api.types.is_numeric_dtype(df[scr]):
            raise TypeError(f"'{scr}' must be numerical.")

        if w is not None:
            if w not in df.columns:
                raise ValueError(f"Columns not found in df: {w}.")
            if not pd.api.types.is_numeric_dtype(df[w]):
                raise TypeError(f"'{w}' must be numerical.")

    @staticmethod
    def _gen_metrics(df, label, scr, w=None):
        """Compute the CDF table and discrimination metrics.

        Parameters
        ----------
        df : pandas.DataFrame
            Validated input data.
        label : str
            Name of the binary target column.
        scr : str
            Name of the score column (used as the grouping variable).
        w : str or None, default=None
            Name of the sample weight column.

        Returns
        -------
        cdf : pandas.DataFrame
            Cumulative distribution table with columns for class 0 and class 1,
            indexed by score value.
        auc : float
            Area Under the ROC Curve, expressed as a percentage (0–100).
        gini : float
            Gini coefficient, defined as ``2 * |AUC - 50|``.
        ks : float
            Kolmogorov-Smirnov statistic — the maximum separation between the
            cumulative distributions of the two classes, as a percentage (0–100).
        """
        # Create probability distribution table
        if not w:
            w = 'w'
            df[w] = 1
        pdf = pd.crosstab(
            df[scr],
            df[label],
            df[w],
            aggfunc='sum',
            normalize='columns'
        ).rename_axis(None, axis=1).reindex(columns=[0, 1], fill_value=0)
        # Calculate cumulative distribution (needed for KS statistic)
        cdf = pdf.cumsum()
        # Calculate AUC
        auc = ((cdf[1] - pdf[1] / 2) * pdf[0]).sum() * 100
        # Calculate Gini coefficient
        gini = 2 * abs(auc - 50)
        # Calculate KS statistic
        ks = (abs(cdf[1] - cdf[0])).max() * 100
        return cdf, auc, gini, ks

    def gen_metrics(self, df, label, scr, w=None):
        """Compute discrimination metrics for a binary classifier.

        Validates the input data and returns the cumulative distribution
        function (CDF) table used to derive KS, Gini, and AUC.

        Parameters
        ----------
        df : pandas.DataFrame
            Input data containing the label, score, and optionally a weight
            column.
        label : str
            Name of the binary target column. Must contain only values in
            {0, 1} with no missing entries.
        scr : str
            Name of the score or predicted probability column. Must be numeric
            with no missing entries.
        w : str or None, default=None
            Name of the sample weight column. If ``None``, all observations
            are assigned equal weight of 1.

        Returns
        -------
        pandas.DataFrame
            Cumulative distribution table indexed by score value, with columns
            for the non-event (0) and event (1) class cumulative proportions.

        Examples
        --------
        >>> evaluator = BinaryClassifierEvaluator()
        >>> df = BinaryClassifierEvaluator.gen_test_data()
        >>> cdf = evaluator.gen_metrics(df, label='y', scr='x')
        """
        try:
            self._valid_df(df, label, scr, w)
        except (TypeError, ValueError) as e:
            print(f"Error: {e}")
            return

        if w:
            df1 = df[[label, scr, w]]
        else:
            df1 = df[[label, scr]].assign(w=1)
            w = 'w'

        cdf, auc, gini, ks = self._gen_metrics(df1, label, scr, w)
        return cdf, auc, gini, ks


if __name__ == '__main__':
    evaluator = BinaryClassifierEvaluator()
    df = BinaryClassifierEvaluator.gen_test_data()
    cdf, auc, gini, ks = evaluator.gen_metrics(df, label='y', scr='x')
    print(f"AUC  : {auc:.4f}%")
    print(f"Gini : {gini:.4f}%")
    print(f"KS   : {ks:.4f}%")
    print("\nCDF table (first 5 rows):")
    print(cdf.head())
