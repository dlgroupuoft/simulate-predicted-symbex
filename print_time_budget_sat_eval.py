
import os
import sys
import pickle
from copy import copy, deepcopy

sys.path.append( os.path.dirname(os.path.realpath(__file__)) )
from utils.TimeBudgetStats import TimeBudgetStats, TotalStats
from utils.common import divide_none_on_zero
from utils.args import parse_args
from utils.time_stats import calculate_stats_total_time
from constants.timegroup_constants import *
TIME_BUDGET_LIST.append("FINAL")
PERCENT_TIME_BUDGET.append("FINAL")

# ===========================================================================

def get_avg_sat_unsat_stats(sat_unsat_stats):
    avg_stats = []
    for stats_count in sat_unsat_stats:
        stats, count = stats_count
        if count == 0:
            assert(stats is None)
            avg_stats.append(None)
        else:
            avg_stats.append(stats / count)
    return avg_stats

def prc_recall(sat_unsat_counts):
    num_actual_sat = sum(sat_unsat_counts[1])
    num_actual_unsat = sum(sat_unsat_counts[0])

    num_predicted_sat = sat_unsat_counts[0][1] + sat_unsat_counts[1][1]
    num_predicted_unsat = sat_unsat_counts[0][0] + sat_unsat_counts[1][0]

    sat_prc = divide_none_on_zero(sat_unsat_counts[1][1], num_predicted_sat)
    sat_recall = divide_none_on_zero(sat_unsat_counts[1][1], num_actual_sat)
    unsat_prc = divide_none_on_zero(sat_unsat_counts[0][0], num_predicted_unsat)
    unsat_recall = divide_none_on_zero(sat_unsat_counts[0][0], num_actual_unsat)
    return [sat_prc, sat_recall, unsat_prc, unsat_recall]

# ===========================================================================

class PrintStats:
    worklist_max_size = 0
    worklist_curr_size = 0

    percent_sat_paths = 0
    percent_orig_time = 0

    analyzed_confidence = 0
    missed_confidence = 0
    saved_confidence = 0
    wait_finish_confidence = 0
    wait_notfinished_confidence = 0
    long_analyze_confidence = 0

    percent_strategy_overhead_of_progress_time = 0

    percent_sat_analysis_time = 0
    percent_sat_find_time = 0
    percent_sat_analysis_find_time = 0
    percent_find_vs_analysis_sat_time = 0

    percent_sat_omitted_analysis_time_of_progress_time = 0
    percent_covered_sat_analysis_time_of_progress_time = 0 # omitted analysis time + analyzed sat time

    percent_sat_paths_in_analyzed = 0
    percent_sat_analysis_of_progress_time = 0

    percent_sat_analyzed_paths_of_total = 0
    percent_unsat_analyzed_paths_of_total = 0
    percent_missed_paths_of_total = 0
    percent_saved_paths_of_total = 0
    percent_sat_analysis_time_of_total = 0
    percent_missed_analysis_time_of_total = 0
    percent_saved_analysis_time_of_total = 0

    percent_sat_omitted_time_of_total = 0
    percent_covered_sat_analysis_time_of_total = 0 # omitted analysis time + analyzed sat time
    
    percent_sat_paths_of_sat = 0
    percent_unsat_paths_of_unsat = 0
    percent_sat_time_of_sat = 0

    percent_omitted_sat_of_sat = 0
    percent_discarded_unsat_of_unsat = 0
    percent_know_sat_of_sat = 0
    percent_know_unsat_of_unsat = 0

    ranklist_num_paths = 0
    avg_ranklist_path_depth = 0

    avg_popped_sat_depth = 0
    avg_popped_unsat_depth = 0
    avg_sat_liststay_pops = 0
    avg_unsat_liststay_pops = 0
    percent_popped_sat = 0

    timegroup_sat_unsat_prc_recall = None
    total_sat_unsat_prc_recall = None

    def __init__(self):
        self.timegroup_sat_unsat_prc_recall = []
        self.total_sat_unsat_prc_recall = []

    def sum_timegroup_sat_unsat_prc_recall(self, to_add_print_stats):
        add_time_supr_stats = to_add_print_stats.timegroup_sat_unsat_prc_recall
        add_total_supr_stats = to_add_print_stats.total_sat_unsat_prc_recall
        if len(self.timegroup_sat_unsat_prc_recall) == 0:
            self.timegroup_sat_unsat_prc_recall = deepcopy(add_time_supr_stats)

            if len(self.total_sat_unsat_prc_recall) == 0:
                self.total_sat_unsat_prc_recall = deepcopy(add_total_supr_stats)
        else:
            def add_stats(curr_res_stats, add_stats):
                assert(len(curr_res_stats) == len(add_stats))
                assert(len(add_stats) == 4)
                for i, sub_add_stats_count in enumerate(add_stats):
                    assert(len(sub_add_stats_count) == 2)
                    if sub_add_stats_count[1] == 0:
                        assert(sub_add_stats_count[0] is None)
                        continue
                    if curr_res_stats[i][1] == 0:
                        assert(curr_res_stats[i][0] is None)
                        curr_res_stats[i][0] = sub_add_stats_count[0]
                        curr_res_stats[i][1] = sub_add_stats_count[1]
                    else:
                        curr_res_stats[i][0] += sub_add_stats_count[0]
                        curr_res_stats[i][1] += sub_add_stats_count[1]

            #
            assert(len(self.timegroup_sat_unsat_prc_recall) == len(add_time_supr_stats))
            for time_ind, sub_add_supr_stats in enumerate(add_time_supr_stats):
                add_stats(self.timegroup_sat_unsat_prc_recall[time_ind], sub_add_supr_stats)

            #
            assert(len(self.total_sat_unsat_prc_recall) > 0)
            add_stats(self.total_sat_unsat_prc_recall, add_total_supr_stats)

    def get_avg_timegroup_sat_unsat_prc_recall(self):
        avg_timegroup_stats = []
        for time_ind, sat_unsat_stats in enumerate(self.timegroup_sat_unsat_prc_recall):
            avg_stats = get_avg_sat_unsat_stats(sat_unsat_stats)
            avg_timegroup_stats.append(avg_stats)
        return avg_timegroup_stats

    def get_avg_alltime_sat_unsat_prc_recall(self):
        return get_avg_sat_unsat_stats(self.total_sat_unsat_prc_recall)

    def yield_stats_field_attr():
        for attr in dir(PrintStats):
            if attr.startswith("__"):
                continue
            elif attr == "yield_stats_field_attr":
                continue
            elif attr == "timegroup_sat_unsat_prc_recall" or attr == "total_sat_unsat_prc_recall":
                continue
            elif attr == "sum_timegroup_sat_unsat_prc_recall" or attr == "get_avg_timegroup_sat_unsat_prc_recall" or attr == "get_avg_alltime_sat_unsat_prc_recall":
                continue
            yield attr

def print_progress_avg_percent_entry(progress_avg_percent_dict, progress_fmt="{0}"):
    def print_progress_percent_entry(progress, percent_stats):
        progress_str = progress_fmt.format(progress) if type(progress) is not str else progress
        print(progress_str, end="")
        for ele in percent_stats:
            if ele is not None:
                print(", {0:.2f}%".format(ele * 100), end="")
            else:
                print(", None", end="")
        print()

    print("progress, app_num_summed, worklist_max_size, worklist_curr_size")
    for progress, stats_entry in progress_avg_percent_dict.items():
        avg_stats, app_num_summed = stats_entry
        progress_str = progress_fmt.format(progress) if type(progress) is not str else progress
        print("{0}, {1}, {2:.1f}, {3:.1f}".format(progress_str, app_num_summed, avg_stats.worklist_max_size, avg_stats.worklist_curr_size))
    print()

    print("progress, percent_strategy_overhead_of_progress_time")
    for progress, stats_entry in progress_avg_percent_dict.items():
        avg_stats, app_num_summed = stats_entry
        progress_str = progress_fmt.format(progress) if type(progress) is not str else progress
        print("{0}, {1:.4g}%".format(progress_str, avg_stats.percent_strategy_overhead_of_progress_time * 100))
    print()

    print("progress, percent_sat_paths, percent_orig_time")
    for progress, stats_entry in progress_avg_percent_dict.items():
        avg_stats, app_num_summed = stats_entry
        progress_str = progress_fmt.format(progress) if type(progress) is not str else progress
        print("{0}, {1:.2f}%, {2:.2f}%".format(progress_str, avg_stats.percent_sat_paths * 100, avg_stats.percent_orig_time * 100))
    print()

    print("progress, analyzed_confidence, missed_confidence, saved_confidence, wait_finish_confidence, wait_notfinished_confidence, long_analyze_confidence")
    for progress, stats_entry in progress_avg_percent_dict.items():
        avg_stats, app_num_summed = stats_entry
        progress_str = progress_fmt.format(progress) if type(progress) is not str else progress
        confidence_val_list = [avg_stats.analyzed_confidence, avg_stats.missed_confidence, avg_stats.saved_confidence, avg_stats.wait_finish_confidence, avg_stats.wait_notfinished_confidence, avg_stats.long_analyze_confidence]
        confidence_str_list = []
        for ele in confidence_val_list:
            if ele is not None:
                confidence_str_list.append("{0:.4f}".format(ele))
            else:
                confidence_str_list.append("None")

        print(progress_str, end=", ")
        print(", ".join(confidence_str_list))
    print()

    print("progress, percent_sat_analysis_time, percent_sat_find_time, percent_sat_analysis_find_time, percent_find_vs_analysis_sat_time")
    for progress, stats_entry in progress_avg_percent_dict.items():
        avg_stats, app_num_summed = stats_entry
        percent_stats = [avg_stats.percent_sat_analysis_time, avg_stats.percent_sat_find_time, avg_stats.percent_sat_analysis_find_time, avg_stats.percent_find_vs_analysis_sat_time]
        print_progress_percent_entry(progress, percent_stats)
    print()

    print("progress, percent_sat_paths_in_analyzed, percent_sat_analysis_of_progress_time")
    for progress, stats_entry in progress_avg_percent_dict.items():
        avg_stats, app_num_summed = stats_entry
        percent_stats = [avg_stats.percent_sat_paths_in_analyzed, avg_stats.percent_sat_analysis_of_progress_time]
        print_progress_percent_entry(progress, percent_stats)
    print()

    print("progress, percent_sat_analyzed_paths_of_total, percent_unsat_analyzed_paths_of_total, percent_missed_paths_of_total, percent_saved_paths_of_total, percent_sat_analysis_time_of_total, percent_missed_analysis_time_of_total, percent_saved_analysis_time_of_total")
    for progress, stats_entry in progress_avg_percent_dict.items():
        avg_stats, app_num_summed = stats_entry
        percent_stats = [avg_stats.percent_sat_analyzed_paths_of_total, avg_stats.percent_unsat_analyzed_paths_of_total, avg_stats.percent_missed_paths_of_total, avg_stats.percent_saved_paths_of_total, avg_stats.percent_sat_analysis_time_of_total, avg_stats.percent_missed_analysis_time_of_total, avg_stats.percent_saved_analysis_time_of_total]
        print_progress_percent_entry(progress, percent_stats)
    print()

    print("progress, percent_sat_omitted_analysis_time_of_progress_time, percent_covered_sat_analysis_time_of_progress_time(omitted+analyzed), percent_sat_omitted_time_of_total, percent_covered_sat_analysis_time_of_total(omitted+analyzed)")
    for progress, stats_entry in progress_avg_percent_dict.items():
        avg_stats, app_num_summed = stats_entry
        percent_stats = [avg_stats.percent_sat_omitted_analysis_time_of_progress_time, avg_stats.percent_covered_sat_analysis_time_of_progress_time, avg_stats.percent_sat_omitted_time_of_total, avg_stats.percent_covered_sat_analysis_time_of_total]
        print_progress_percent_entry(progress, percent_stats)
    print()


    print("know_sat = sat + omitted_sat")
    print("progress, percent_sat_paths_of_sat, percent_omitted_sat_paths_of_sat, percent_know_sat_paths_of_sat, percent_sat_time_of_sat")
    for progress, stats_entry in progress_avg_percent_dict.items():
        avg_stats, app_num_summed = stats_entry
        percent_stats = [avg_stats.percent_sat_paths_of_sat, avg_stats.percent_omitted_sat_of_sat, avg_stats.percent_know_sat_of_sat, avg_stats.percent_sat_time_of_sat]
        print_progress_percent_entry(progress, percent_stats)
    print()

    print("know_unsat = unsat + omitted_unsat")
    print("progress, percent_unsat_paths_of_unsat, percent_discarded_unsat_paths_of_unsat, percent_know_unsat_paths_of_unsat")
    for progress, stats_entry in progress_avg_percent_dict.items():
        avg_stats, app_num_summed = stats_entry
        percent_stats = [avg_stats.percent_unsat_paths_of_unsat, avg_stats.percent_discarded_unsat_of_unsat, avg_stats.percent_know_unsat_of_unsat]
        print_progress_percent_entry(progress, percent_stats)
    print()

    #############################################
    print("progress, avg_ranklist_path_depth, ranklist_num_paths")
    for progress, stats_entry in progress_avg_percent_dict.items():
        avg_stats, app_num_summed = stats_entry
        avg_ranklist_path_depth = avg_stats.avg_ranklist_path_depth if avg_stats.avg_ranklist_path_depth is not None else "N/A"
        print("{0}, {1}, {2}".format(progress, avg_ranklist_path_depth, avg_stats.ranklist_num_paths))
    print()


    def fmt_or_none(stats, fmt_str, transform_func=None):
        if stats is not None:
            if transform_func is not None:
                stats = transform_func(stats)
            ret_fmt = fmt_str.format(stats)
        else:
            ret_fmt = "None"

    print("progress, avg_popped_sat_depth, avg_sat_liststay_pops, avg_popped_unsat_depth, avg_unsat_liststay_pops, percent_popped_sat")
    for progress, stats_entry in progress_avg_percent_dict.items():
        avg_stats, app_num_summed = stats_entry

        avg_popped_sat_depth_str = fmt_or_none(avg_stats.avg_popped_sat_depth, "{0:.2f}%")
        avg_popped_unsat_depth_str = fmt_or_none(avg_stats.avg_popped_unsat_depth, "{0:.2f}%")
        percent_popped_sat_str = fmt_or_none(avg_stats.percent_popped_sat, "{0:.2f}%", transform_func=lambda x: x * 100)
        print("{0}, {1}, {2}, {3}, {4}, {5}".format(progress, avg_popped_sat_depth_str, avg_stats.avg_sat_liststay_pops, avg_popped_unsat_depth_str, avg_stats.avg_unsat_liststay_pops, percent_popped_sat_str))
    print()
    #############################################

    print("progress, all_time_sat_prc/sat_recall, all_time_unsat_prc/unsat_recall")
    for progress, stats_entry in progress_avg_percent_dict.items():
        avg_stats, app_num_summed = stats_entry
        avg_alltime_prc_recall = avg_stats.get_avg_alltime_sat_unsat_prc_recall()
        assert(len(avg_alltime_prc_recall) > 0)
        avg_alltime_prc_recall = ["None" if ele is None else "{0:.4f}".format(ele) for ele in avg_alltime_prc_recall]

        progress_str = progress_fmt.format(progress) if type(progress) is not str else progress
        sat_prc, sat_recall, unsat_prc, unsat_recall = avg_alltime_prc_recall
        print("{0}, {1}/{2}, {3}/{4}".format(progress_str, sat_prc, sat_recall, unsat_prc, unsat_recall))
    print()

    for progress, stats_entry in progress_avg_percent_dict.items():
        avg_stats, app_num_summed = stats_entry
        progress_str = progress_fmt.format(progress) if type(progress) is not str else progress
        print("=> progress {0}".format(progress_str))
        avg_timegroup_prc_recall = avg_stats.get_avg_timegroup_sat_unsat_prc_recall()
        print("timegroup, sat_prc/sat_recall, unsat_prc/unsat_recall")
        assert(len(avg_timegroup_prc_recall) > 0)
        for time_ind, sat_unsat_prc_recall in enumerate(avg_timegroup_prc_recall):
            sat_unsat_prc_recall = ["None" if ele is None else "{0:.4f}".format(ele) for ele in sat_unsat_prc_recall]
            sat_prc, sat_recall, unsat_prc, unsat_recall = sat_unsat_prc_recall
            print("{0}, {1}/{2}, {3}/{4}".format(TIME_LIST[time_ind], sat_prc, sat_recall, unsat_prc, unsat_recall))
        print()
    print()

# ===========================================================================

def get_percent_stats(tb_stats, total_stats):
    assert(total_stats.total_analysis_time > 0)
    assert(total_stats.total_find_time > 0)
    assert(total_stats.total_sat_analysis_time > 0)
    assert(total_stats.total_sat_find_time > 0)
    assert(total_stats.num_total_paths > 0)
    assert(total_stats.num_total_sat_paths > 0)

    ret_stats = PrintStats()
    ret_stats.worklist_max_size = tb_stats.worklist_max_size
    ret_stats.worklist_curr_size = tb_stats.worklist_curr_size

    ret_stats.analyzed_confidence = tb_stats.analyzed_confidence
    ret_stats.missed_confidence = tb_stats.missed_confidence
    ret_stats.saved_confidence = tb_stats.saved_confidence
    ret_stats.wait_finish_confidence = tb_stats.wait_finish_confidence
    ret_stats.wait_notfinished_confidence = tb_stats.wait_notfinished_confidence
    ret_stats.long_analyze_confidence = tb_stats.long_analyze_confidence

    #
    analysis_time = tb_stats.total_time_analyze_sat + tb_stats.total_time_analyze_unsat
    analysis_find_time = analysis_time + tb_stats.path_find_time
    assert(tb_stats.path_find_time >= 0)
    ret_stats.percent_sat_analysis_time = divide_none_on_zero(tb_stats.total_time_analyze_sat, analysis_time)
    ret_stats.percent_sat_find_time = divide_none_on_zero(tb_stats.path_find_time_sat, tb_stats.path_find_time)
    ret_stats.percent_sat_analysis_find_time = divide_none_on_zero(tb_stats.total_time_analyze_sat + tb_stats.path_find_time_sat, analysis_find_time)
    ret_stats.percent_find_vs_analysis_sat_time = divide_none_on_zero(tb_stats.path_find_time, analysis_find_time)

    #
    total_analyzed = tb_stats.num_sat_analyzed + tb_stats.num_unsat_analyzed
    assert(total_analyzed > 0)
    ret_stats.percent_sat_paths_in_analyzed = divide_none_on_zero(tb_stats.num_sat_analyzed, total_analyzed)

    curr_progress_time = calculate_stats_total_time(tb_stats)
    curr_progress_time *= 1000 # sec -> ms
    ret_stats.percent_sat_analysis_of_progress_time = divide_none_on_zero(tb_stats.total_time_analyze_sat, curr_progress_time)
    ret_stats.percent_strategy_overhead_of_progress_time = divide_none_on_zero(tb_stats.strategy_overhead, curr_progress_time)

    ret_stats.percent_sat_omitted_analysis_time_of_progress_time = divide_none_on_zero(tb_stats.total_time_sat_analysis_omitted, curr_progress_time)
    covered_sat_analysis_time = tb_stats.total_time_sat_analysis_omitted + tb_stats.total_time_analyze_sat
    ret_stats.percent_covered_sat_analysis_time_of_progress_time = divide_none_on_zero(covered_sat_analysis_time, curr_progress_time)

    #
    ret_stats.percent_sat_paths = tb_stats.percent_sat_paths
    ret_stats.percent_orig_time = tb_stats.percent_orig_time

    #
    ret_stats.percent_sat_analyzed_paths_of_total = divide_none_on_zero(tb_stats.num_sat_analyzed, total_stats.num_total_paths)
    ret_stats.percent_unsat_analyzed_paths_of_total = divide_none_on_zero(tb_stats.num_unsat_analyzed, total_stats.num_total_paths)
    ret_stats.percent_missed_paths_of_total = divide_none_on_zero(tb_stats.num_sat_missed, total_stats.num_total_paths)
    ret_stats.percent_saved_paths_of_total = divide_none_on_zero(tb_stats.num_unsat_saved, total_stats.num_total_paths)

    ret_stats.percent_sat_analysis_time_of_total = divide_none_on_zero(tb_stats.total_time_analyze_sat, total_stats.total_analysis_time)
    ret_stats.percent_missed_analysis_time_of_total = divide_none_on_zero(tb_stats.time_missed_sat, total_stats.total_analysis_time)
    ret_stats.percent_saved_analysis_time_of_total = divide_none_on_zero(tb_stats.time_saved_unsat, total_stats.total_analysis_time)

    ret_stats.percent_sat_omitted_time_of_total = divide_none_on_zero(tb_stats.total_time_sat_analysis_omitted, total_stats.total_analysis_time)
    covered_sat_analysis_time = tb_stats.total_time_sat_analysis_omitted + tb_stats.total_time_analyze_sat
    ret_stats.percent_covered_sat_analysis_time_of_total = divide_none_on_zero(covered_sat_analysis_time, total_stats.total_analysis_time)

    #
    ret_stats.percent_sat_paths_of_sat = divide_none_on_zero(tb_stats.num_sat_analyzed, total_stats.num_total_sat_paths)
    ret_stats.percent_omitted_sat_of_sat = divide_none_on_zero(tb_stats.num_sat_omitted, total_stats.num_total_sat_paths)
    ret_stats.percent_know_sat_of_sat = divide_none_on_zero(tb_stats.num_sat_analyzed + tb_stats.num_sat_omitted, total_stats.num_total_sat_paths)
    ret_stats.percent_sat_time_of_sat = divide_none_on_zero(tb_stats.total_time_analyze_sat, total_stats.total_sat_analysis_time)

    #
    num_total_unsat_paths = total_stats.num_total_paths - total_stats.num_total_sat_paths
    ret_stats.percent_unsat_paths_of_unsat = divide_none_on_zero(tb_stats.num_unsat_analyzed, num_total_unsat_paths)
    ret_stats.percent_discarded_unsat_of_unsat = divide_none_on_zero(tb_stats.num_unsat_omitted, num_total_unsat_paths)
    ret_stats.percent_know_unsat_of_unsat = divide_none_on_zero(tb_stats.num_unsat_analyzed + tb_stats.num_unsat_omitted, num_total_unsat_paths)

    #
    if tb_stats.depth_filter is not None:
        ret_stats.avg_ranklist_path_depth = divide_none_on_zero(tb_stats.ranklist_path_depth_sum, tb_stats.ranklist_num_paths)
        ret_stats.ranklist_num_paths = tb_stats.ranklist_num_paths

    ret_stats.avg_popped_sat_depth = divide_none_on_zero(tb_stats.total_popped_sat_depth, tb_stats.num_popped_sat)
    ret_stats.avg_popped_unsat_depth = divide_none_on_zero(tb_stats.total_popped_unsat_depth, tb_stats.num_popped_unsat)
    ret_stats.avg_sat_liststay_pops = divide_none_on_zero(tb_stats.total_sat_liststay_pops, tb_stats.num_popped_sat)
    ret_stats.avg_unsat_liststay_pops = divide_none_on_zero(tb_stats.total_unsat_liststay_pops, tb_stats.num_popped_unsat)
    ret_stats.percent_popped_sat = divide_none_on_zero(tb_stats.num_popped_sat, tb_stats.num_popped_sat + tb_stats.num_popped_unsat)

    #
    assert(len(ret_stats.timegroup_sat_unsat_prc_recall) == 0)
    for sat_unsat_count in tb_stats.timegroup_sat_unsat:
        count_prc_recall_stats = []
        for ele in prc_recall(sat_unsat_count):
            if ele is None:
                count_prc_recall_stats.append([ele, 0])
            else:
                count_prc_recall_stats.append([ele, 1])
        assert(len(count_prc_recall_stats) == 4)
        ret_stats.timegroup_sat_unsat_prc_recall.append(count_prc_recall_stats)

    total_sat_unsat_count = [[0, 0], [0, 0]]
    for sat_unsat_count in tb_stats.timegroup_sat_unsat:
        total_sat_unsat_count[0][0] += sat_unsat_count[0][0]
        total_sat_unsat_count[0][1] += sat_unsat_count[0][1]
        total_sat_unsat_count[1][0] += sat_unsat_count[1][0]
        total_sat_unsat_count[1][1] += sat_unsat_count[1][1]

    assert(len(ret_stats.total_sat_unsat_prc_recall) == 0)
    for ele in prc_recall(total_sat_unsat_count):
        if ele is None:
            ret_stats.total_sat_unsat_prc_recall.append([ele, 0])
        else:
            ret_stats.total_sat_unsat_prc_recall.append([ele, 1])

    return ret_stats

# ===========================================================================

def sum_up_percent_entry(sum_percent_entry, percent_count, curr_percent_entry):
    if sum_percent_entry is None:
        assert(percent_count is None)
        sum_percent_entry = PrintStats()
        percent_count = PrintStats()

    for attr in PrintStats.yield_stats_field_attr():
        curr_attr_val = getattr(curr_percent_entry, attr)
        if curr_attr_val is None:
            continue
        else:
            sum_attr_val = getattr(sum_percent_entry, attr)
            setattr(sum_percent_entry, attr, sum_attr_val + curr_attr_val)

            count_attr_val = getattr(percent_count, attr)
            setattr(percent_count, attr, count_attr_val + 1)

    sum_percent_entry.sum_timegroup_sat_unsat_prc_recall(curr_percent_entry)
    return sum_percent_entry, percent_count

def get_avg_percent_entry(sum_percent_entry, percent_count):
    avg_percent_entry = PrintStats()
    for attr in PrintStats.yield_stats_field_attr():
        count_attr_val = getattr(percent_count, attr)
        if count_attr_val > 0:
            sum_attr_val = getattr(sum_percent_entry, attr)
            avg_attr_val = sum_attr_val / count_attr_val
        else:
            avg_attr_val = None
        setattr(avg_percent_entry, attr, avg_attr_val)
    avg_percent_entry.timegroup_sat_unsat_prc_recall = deepcopy(sum_percent_entry.timegroup_sat_unsat_prc_recall)
    avg_percent_entry.total_sat_unsat_prc_recall = deepcopy(sum_percent_entry.total_sat_unsat_prc_recall)
    return avg_percent_entry

def check_valid_progress(progress_time_budget, total_stats, progress):
    curr_progress = None
    if progress in progress_time_budget:
        curr_progress = progress
    else:
        # progress_keys = []
        # for k in progress_time_budget.keys():
        #     if type(k) is str:
        #         continue
        #     progress_keys.append(k)
        # if len(progress_keys) == 0:
        #     return None

        # if progress > max(progress_keys):
        if progress > progress_time_budget["FINAL_PERCENT"]:
            curr_progress = "FINAL"
            assert("FINAL" in progress_time_budget)
    if curr_progress is None:
        return None

    #
    tb_stats = progress_time_budget[curr_progress]
    total_analyzed = tb_stats.num_sat_analyzed + tb_stats.num_unsat_analyzed
    if total_analyzed <= 0:
        return None

    for attr in dir(TotalStats):
        if attr.startswith("__"):
            continue
        ele = getattr(total_stats, attr)
        if ele == 0:
            print("check_valid_progress FAILED with \"{0}\" == 0".format(attr))
            return None
    return curr_progress

def obtain_use_time_budget(progress_stats, is_percent_progress=False, is_path_percent_progress=False):
    progress_percent_timebudget, progress_percent_satpath, total_stats = progress_stats
    if is_path_percent_progress:
        assert(not is_percent_progress)
        return progress_percent_satpath, total_stats
    elif is_percent_progress:
        return progress_percent_timebudget, total_stats
    else: # Not using time progress now
        assert(False)

# ===========================================================================

def avg_percent_entry_cross_apps(app_progress_time_budget, progress, is_percent_progress=False, is_path_percent_progress=False, app_curr_fold=None):
    sum_percent_entry = None
    percent_count = None
    app_num_summed = 0
    for app, progress_stats in app_progress_time_budget.items():
        if app_curr_fold is not None and app not in app_curr_fold:
            continue
        use_time_budget, total_stats = obtain_use_time_budget(progress_stats, is_percent_progress=is_percent_progress, is_path_percent_progress=is_path_percent_progress)

        curr_progress = check_valid_progress(use_time_budget, total_stats, progress)
        if not curr_progress:
            continue
        tb_stats = use_time_budget[curr_progress]

        # For path percent, record stats it takes to reach at least X% of paths
        # So skip stats that are not there yet
        if is_path_percent_progress:
            if progress != "FINAL" and tb_stats.percent_sat_paths < progress / 100.0:
                continue
        elif is_percent_progress: # time percent progress
            if progress != "FINAL" and tb_stats.percent_orig_time < progress / 100.0:
                continue

        curr_percent_entry = get_percent_stats(tb_stats, total_stats)
        sum_percent_entry, percent_count = sum_up_percent_entry(sum_percent_entry, percent_count, curr_percent_entry)

        app_num_summed += 1

    if app_num_summed == 0:
        avg_percent_entry = None
    else:
        avg_percent_entry = get_avg_percent_entry(sum_percent_entry, percent_count)
    return avg_percent_entry, app_num_summed

def print_app_progress_time(app_progress_time_budget, is_percent_progress=False, is_path_percent_progress=False, app_curr_fold=None):
    progress_avg_time_budget = dict()
    if is_percent_progress or is_path_percent_progress:
        progress_fmt="{0}%"
        timebudget_list = PERCENT_TIME_BUDGET
    else:
        progress_fmt="{0}"
        timebudget_list = TIME_BUDGET_LIST

    for progress in timebudget_list:
        exact_avg_analysis_time = 0
        exact_num_apps = 0
        for app, progress_stats in app_progress_time_budget.items():
            if app_curr_fold is not None and app not in app_curr_fold:
                continue

            use_time_budget, total_stats = obtain_use_time_budget(progress_stats, is_percent_progress=is_percent_progress, is_path_percent_progress=is_path_percent_progress)
            curr_progress = check_valid_progress(use_time_budget, total_stats, progress)
            if not curr_progress:
                continue
            tb_stats = use_time_budget[curr_progress]

            exact_avg_analysis_time += calculate_stats_total_time(tb_stats)
            exact_num_apps += 1
        if exact_num_apps == 0:
            continue
        exact_avg_analysis_time /= exact_num_apps

        #
        avg_percent_entry, app_num_summed = avg_percent_entry_cross_apps(app_progress_time_budget, progress, is_percent_progress=is_percent_progress, is_path_percent_progress=is_path_percent_progress, app_curr_fold=app_curr_fold)
        if not avg_percent_entry:
            continue
        progress_avg_time_budget[progress] = (avg_percent_entry, app_num_summed)

    print_progress_avg_percent_entry(progress_avg_time_budget, progress_fmt=progress_fmt)

# ===========================================================================

if __name__ == "__main__":
    args = parse_args(is_print_args=True)
    #
    # print_app_progress_time(args.app_progress_time_budget, is_percent_progress=False, app_curr_fold=args.app_curr_fold)
    # print()
    # print("===================================================")
    # print()
    print("=> PROGRESS TIME PERCENT")
    print_app_progress_time(args.app_progress_time_budget, is_percent_progress=True, app_curr_fold=args.app_curr_fold)
    print("=> PROGRESS PATH PERCENT")
    print_app_progress_time(args.app_progress_time_budget, is_path_percent_progress=True, app_curr_fold=args.app_curr_fold)

