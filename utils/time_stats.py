
import os
import sys

from copy import deepcopy

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.TimeBudgetStats import TimeBudgetStats, TotalStats
from constants.timegroup_constants import *

# ===========================================================================

def calculate_stats_total_time(stats):
    # total_time_analyze == wait_finish time + long_analyze time
    total_time_sec = stats.total_time_analyze_sat + stats.total_time_analyze_unsat
    total_time_sec += stats.path_find_time
    total_time_sec += (stats.time_wait_notfinished_sat + stats.time_wait_notfinished_unsat)

    # strategy overhead
    total_time_sec += stats.strategy_overhead

    # Exclude the wait_notfinished time also in long_analyze (hard pred, pred rank)
    total_time_sec -= (stats.time_used_wait_notfinished_sat + stats.time_used_wait_notfinished_unsat)

    total_time_sec /= 1000 # ms -> sec
    return total_time_sec

def update_percent_orig_time(stats, total_analysis_time, total_find_time):
    total_time_sec = calculate_stats_total_time(stats)
    all_analysis_find_sec = (total_analysis_time + total_find_time) / 1000 # ms ->sec
    stats.percent_orig_time = total_time_sec / all_analysis_find_sec
    return total_time_sec, all_analysis_find_sec

def update_percent_sat_paths(stats, all_analysis_sat_paths):
    stats.percent_sat_paths = (stats.num_sat_analyzed + stats.num_sat_omitted) / float(all_analysis_sat_paths)

def update_time_budget_stats(progress_time_budget, time_budget_ind,
                             stats, total_stats,
                             is_path_percent=False,
                             is_percent=False):
    curr_budget_threshold = None
    if is_path_percent:
        total_sat = stats.num_sat_analyzed + stats.num_sat_omitted

    update_percent_sat_paths(stats, total_stats.num_total_sat_paths)
    total_time_sec, all_analysis_find_sec = update_percent_orig_time(stats, total_stats.total_analysis_time, total_stats.total_find_time)

    if is_path_percent:
        assert(total_stats.num_total_sat_paths > 0)
        ind_length = len(PERCENT_TIME_BUDGET)
    elif is_percent:
        assert(total_stats.total_analysis_time > 0)
        assert(total_stats.total_find_time > 0)
        ind_length = len(PERCENT_TIME_BUDGET)
    else:
        ind_length = len(TIME_BUDGET_LIST)

    curr_budget_threshold = None
    while time_budget_ind < ind_length:
        if is_path_percent:
            pass_threshold_cond = (total_sat >= PERCENT_TIME_BUDGET[time_budget_ind] / 100.0 * total_stats.num_total_sat_paths)
            time_budget_val = PERCENT_TIME_BUDGET[time_budget_ind]
        elif is_percent:
            pass_threshold_cond = (total_time_sec >= PERCENT_TIME_BUDGET[time_budget_ind] / 100.0 * all_analysis_find_sec)
            time_budget_val = PERCENT_TIME_BUDGET[time_budget_ind]
        else:
            pass_threshold_cond = (total_time_sec >= TIME_BUDGET_LIST[time_budget_ind])
            time_budget_val = TIME_BUDGET_LIST[time_budget_ind]

        if pass_threshold_cond:
            curr_budget_threshold = time_budget_val
            time_budget_ind += 1
        else:
            break

    if curr_budget_threshold is not None:
        save_stats = deepcopy(stats)
        save_stats.avg_confidence()
        progress_time_budget[curr_budget_threshold] = save_stats
    return time_budget_ind

def update_final_budget_stats(progress_time_budget, stats, total_stats,
                              is_path_percent=False, is_percent=False):
    update_percent_sat_paths(stats, total_stats.num_total_sat_paths)
    update_percent_orig_time(stats, total_stats.total_analysis_time, total_stats.total_find_time)
    if is_path_percent:
        progress_time_budget["FINAL_PERCENT"] = stats.percent_sat_paths * 100.0
    else:
        progress_time_budget["FINAL_PERCENT"] = stats.percent_orig_time * 100.0

    save_stats = deepcopy(stats)
    save_stats.avg_confidence()
    progress_time_budget["FINAL"] = save_stats

