import pandas as pd
import matplotlib.pyplot as plt
import math
from .evaluation import BinaryClassifierEvaluator

class ScoreDistributionPlotter:


    def __init__(self):
        self.evaluator = BinaryClassifierEvaluator()

    @staticmethod
    def _valid_score(score, label):
        try:
            score = pd.Series(score)
            label_s = pd.Series(label)
        except Exception:
            raise TypeError(f"score and label must be convertible to pd.Series.")

        if len(score) != len(label_s):
            raise ValueError(f"score and label must have the same length.")
        if len(score) <= 3:
            raise ValueError(f"score and label must have length > 3.")
        return True

    @staticmethod
    def _plot_cdf(ax, cdf, auc, gini, ks, global_min, global_max):
        first_row = pd.DataFrame([[0, 0]], columns=[0, 1], index=[global_min])
        last_row  = pd.DataFrame([[1, 1]], columns=[0, 1], index=[global_max])
        cdf1 = pd.concat([first_row, cdf, last_row])

        ax.plot(cdf1.index, cdf1[0], label='CDF|y=0')
        ax.plot(cdf1.index, cdf1[1], label='CDF|y=1')
        ax.set_xlabel('Score')
        ax.set_ylabel('Cumulative Distribution')
        ax.set_title(f"Gini: {gini:.1f}  |  KS: {ks:.1f}")
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend()

    def plot_all_score(self, label, scores, ncols=3, hspace=0.45):
        n = len(scores)
        if not(isinstance(ncols, int) and (1<= n <=20)):
            ncols = 3
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
        fig.subplots_adjust(hspace=hspace)
        axes = axes.flatten()
        stats = []

        global_min = min(s.min() for s in scores)
        global_max = max(s.max() for s in scores)
        for ax, score in zip(axes,scores):
            try:
                valid = self._valid_score(score, label)
                df = pd.DataFrame({'label':label, 'score': score})
                cdf, auc, gini, ks = self.evaluator.gen_metrics(df, 'label', 'score')

                auc = max(100 - auc, auc)
                stats.append({'ax':ax, 'cdf':cdf, 'auc':auc, 'gini':gini,'ks':ks})
                self._plot_cdf(ax, cdf, auc, gini,ks, global_min, global_max)
            except (TypeError, ValueError) as e:
                print(f"Error: {e}")
        fig.suptitle("CDF/Gini of all scores", fontsize=14, fontweight='bold')
        fig.subplots_adjust(top=0.93)   # leave room for the suptitle

        return fig, axes,


sdp = ScoreDistributionPlotter()

df = sdp.evaluator.gen_test_data()
df = pd.DataFrame({'x':[1,2,3,3,3,4,4,5,6], 'y':[1,1,1,1,0,1,0,1,0]})
scrlst = [df['x'], df['x']+2]*3
sdp.plot_all_score(df['y'],scrlst)

cdf, auc, gini, ks=sdp.evaluator.gen_metrics(df, 'y', 'x')

first_row = pd.DataFrame([[0, 0]], columns=[0, 1], index=[cdf.index[0]])
cdf = pd.concat([first_row, cdf])
print(cdf)
