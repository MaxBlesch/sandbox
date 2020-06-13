"""Processing time of _full_solution() function of `respy` time only."""
import datetime

import numpy as np
from numba import njit
from numba import prange
from respy.shared import aggregate_keane_wolpin_utility
from respy.shared import calculate_expected_value_functions


@njit(parallel=True)
def parralel_range_test(wages, nonpecs, continuation_values, draws, delta):
    n_draws, n_choices = draws.shape
    v = np.repeat(np.nan, n_draws)

    out = 0
    for i in prange(n_draws):

        for j in prange(n_choices):
            out += wages[j] * draws[i, j] + nonpecs[j] + delta * continuation_values[j]

    return out


# @njit(
#     parallel=True
# )
# def execute_parrallel (
#         wages, nonpecs, continuation_values, period_draws_emax_risk, delta, eta
#
# ):
#     out = calculate_expected_value_functions(
#         wages, nonpecs, continuation_values, period_draws_emax_risk, delta, eta
#     )
#     return out


if __name__ == "__main__":

    wages = np.loadtxt("kw_one_wages.txt")[0, :]
    nonpecs = np.loadtxt("kw_one_nonpecs.txt")[0, :]
    continuation_values = np.loadtxt("kw_one_continuation_values.txt")[0, :]
    period_draws_emax_risk = np.loadtxt("kw_one_period_draws_emax_risk.txt")
    delta = 0.95

    parralel_range_test(
        wages, nonpecs, continuation_values, period_draws_emax_risk, delta
    )
    parralel_range_test.parallel_diagnostics(level=4)
    # num_runs = 10
    # run_times = [0.1] * num_runs
    #
    # for j in range(10):
    #     start = datetime.datetime.now()
    #     calc = parralel_range_test(wages,
    #     nonpecs, continuation_values, period_draws_emax_risk, delta)
    #     end = datetime.datetime.now()
    #     time_delta = start - end
    #     run_times[j] = time_delta.microseconds + time_delta.seconds * 100000

    # result_array = np.array(run_times[1:])
    # np.savetxt(f"times_parr_{num_runs}_0.txt", result_array)
