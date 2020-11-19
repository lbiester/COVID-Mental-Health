"""
Utils for statistical tests
"""
from typing import Tuple

import numpy as np
from scipy.stats import pearsonr, norm


def ttest_confidence_interval_autocorr_correction(outside_ci_pre: np.array, outside_ci_post: np.array,
                                                  p0: float) -> Tuple[float, float]:
    """
    t-test on # of values outside of the confidence interval, with correction for 1st order autocorrelation
    :param outside_ci_pre: array of ints (0 or 1/boolean): is this timestamp outside of the confidence interval (pre)?
    :param outside_ci_post: array of ints (0 or 1/boolean): is this timestamp outside of the confidence interval (post)?
    :param p0: the coverage probability of the prediction interval (e.g. 90%, 99%) (we have called this interval width)
    :return: t-test statistic, p-value
    """
    n = outside_ci_post.size
    q = outside_ci_post.mean()
    r = pearsonr(outside_ci_pre[:-1], outside_ci_pre[1:])[0]

    sum_q_n = (1 + r) / (1 - r)

    z = (q - (1 - p0)) / np.sqrt(p0 * (1 - p0) * sum_q_n / n)
    # compute p value: based on https://docs.scipy.org/doc/scipy/reference/tutorial/stats.html#t-test-and-ks-test
    # one sided, so do not multiply by 2
    pval = norm.sf(np.abs(z))
    return z, pval