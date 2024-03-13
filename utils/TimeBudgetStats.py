
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from constants.timegroup_constants import *
from utils.common import get_tenth_time_ind, apprx_equal, divide_none_on_zero

# ===========================================================================

class TotalStats():
    total_analysis_time = 0
    total_find_time = 0
    total_sat_analysis_time = 0
    total_sat_find_time = 0
    num_total_paths = 0
    num_total_sat_paths = 0

    def update(self, actual_class, constr_solve_time, execute_time):
        self.total_analysis_time += constr_solve_time
        self.total_find_time += execute_time

        self.num_total_paths += 1

        if actual_class:
            self.total_sat_analysis_time += constr_solve_time
            self.total_sat_find_time += execute_time

            self.num_total_sat_paths += 1

# ===========================================================================

class TimeBudgetStats():
    percent_sat_paths = None
    percent_orig_time = None

    worklist_max_size = 0 #
    worklist_curr_size = 0

    num_sat_omitted = 0
    num_unsat_omitted = 0
    num_sat_analyzed = 0
    num_unsat_analyzed = 0
    total_time_analyze_sat = 0 # Analysis time means constraint solve time
    total_time_analyze_unsat = 0
    total_time_sat_analysis_omitted = 0 #

    analyzed_confidence = 0 #
    path_find_time_sat = 0 # Path find time means path execute time
    path_find_time_unsat = 0
    path_find_time = 0
    strategy_overhead = 0

    num_sat_missed = 0
    num_unsat_saved = 0
    time_missed_sat = 0
    missed_confidence = 0 #
    time_saved_unsat = 0
    saved_confidence = 0 #

    num_sat_wait_finish = 0
    num_unsat_wait_finish = 0
    time_wait_finish_sat = 0
    time_wait_finish_unsat = 0
    wait_finish_confidence = 0 #

    num_filter_pass_sat = 0
    num_filter_pass_unsat = 0
    num_filter_notpass_sat = 0
    num_filter_notpass_unsat = 0
    time_filter_pass_sat = 0
    time_filter_pass_unsat = 0
    time_filter_notpass_sat = 0
    time_filter_notpass_unsat = 0

    num_sat_wait_notfinished = 0
    num_unsat_wait_notfinished = 0
    time_wait_notfinished_sat = 0
    time_wait_notfinished_unsat = 0
    time_used_wait_notfinished_sat = 0
    time_used_wait_notfinished_unsat = 0
    wait_notfinished_confidence = 0 #

    num_sat_long_analyze = 0
    num_unsat_long_analyze = 0
    long_analyze_confidence = 0 #
    time_long_analyze_sat = 0
    time_long_analyze_unsat = 0

    wait_time = 0
    random_filter = None
    pred_filter = None
    depth_filter = None

    #
    ranklist_path_depth_sum = None
    ranklist_num_paths = None

    # Sat / unsat paths that get popped out of the ranklist
    total_popped_sat_depth = 0
    total_popped_unsat_depth = 0
    total_sat_liststay_pops = 0 # num of pops a sat path encounters while it is in list
    total_unsat_liststay_pops = 0 # num of pops an unsat path encounters while it is in list
    total_sat_liststsay_iters = 0 # num of loop iters a sat path encounters while it is in list
    total_unsat_liststsay_iters = 0
    num_popped_sat = 0
    num_popped_unsat = 0

    # Updated by count_timegroup_sat_unsat_pred
    timegroup_sat_unsat = None

    def __init__(self, wait_filter):
        if wait_filter is not None:
            random_filter, pred_filter, depth_filter, wait_time = wait_filter
            self.wait_time = wait_time
            self.random_filter = random_filter
            self.pred_filter = pred_filter
            self.depth_filter = depth_filter
            if self.wait_time > 0:
                assert(self.random_filter is None)
                assert(self.pred_filter is None)
            else:
                if self.random_filter is not None:
                    assert(self.pred_filter is None)
                    assert(self.depth_filter is None)
                if self.pred_filter is not None:
                    assert(self.random_filter is None)
                    assert(self.depth_filter is None)
                if self.depth_filter is not None:
                    assert(self.pred_filter is None)
                    assert(self.random_filter is None)
                    self.ranklist_path_depth_sum = 0
                    self.ranklist_num_paths = 0

        self.timegroup_sat_unsat = []
        for time_str in TIME_LIST:
            self.timegroup_sat_unsat.append([[0, 0], [0, 0]])

    def count_timegroup_sat_unsat_pred(self, constr_solve_time,
                                       actual_class, predicted_class):
        time_ind = get_tenth_time_ind(constr_solve_time, len(TIME_LIST))
        actual_selection = 1 if actual_class else 0
        predict_selection = 1 if predicted_class else 0
        self.timegroup_sat_unsat[time_ind][actual_selection][predict_selection] += 1

    def check_final_analysis_path(self, num_total_paths):
        stats_total_paths = self.num_sat_analyzed + self.num_unsat_analyzed + self.num_sat_missed + self.num_unsat_saved
        assert(num_total_paths == stats_total_paths)

    def check_final_analysis_time(self, total_analysis_time, total_find_time,
                                  has_skipped_paths):
        if not has_skipped_paths:
            stats_total_analysis_time = self.total_time_analyze_sat + self.total_time_analyze_unsat
            assert(apprx_equal(stats_total_analysis_time, total_analysis_time))
            assert(apprx_equal(self.path_find_time, total_find_time))

    def check_final_finish_notfinish(self):
        if self.wait_time == 0:
            return
        total_sat_paths = self.num_sat_analyzed + self.num_sat_missed
        sat_paths_with_wait = self.num_sat_wait_finish + self.num_sat_wait_notfinished
        assert(total_sat_paths == sat_paths_with_wait)

        total_unsat_paths = self.num_unsat_analyzed + self.num_unsat_saved
        unsat_paths_with_wait = self.num_unsat_wait_finish + self.num_unsat_wait_notfinished
        assert(total_unsat_paths == unsat_paths_with_wait)

    def check_all_notfinished_is_used(self):
        assert(apprx_equal(self.time_wait_notfinished_sat, self.time_used_wait_notfinished_sat))
        assert(apprx_equal(self.time_wait_notfinished_unsat, self.time_used_wait_notfinished_unsat))

    def check_wait_consistent(self):
        assert(self.wait_time >= 0)

        sat_wait_long_time = self.time_wait_finish_sat + self.time_long_analyze_sat
        unsat_wait_long_time = self.time_wait_finish_unsat + self.time_long_analyze_unsat
        assert(apprx_equal(self.total_time_analyze_sat, sat_wait_long_time))
        assert(apprx_equal(self.total_time_analyze_unsat, unsat_wait_long_time))

        num_sat_wait_long = self.num_sat_wait_finish + self.num_sat_long_analyze
        num_unsat_wait_long = self.num_unsat_wait_finish + self.num_unsat_long_analyze
        assert(self.num_sat_analyzed == num_sat_wait_long)
        assert(self.num_unsat_analyzed == num_unsat_wait_long)

    def add_find_time(self, path_find_time, actual_class):
        self.path_find_time += path_find_time
        if actual_class:
            self.path_find_time_sat += path_find_time
        else:
            self.path_find_time_unsat += path_find_time

    def add_strategy_overhead(self, overhead):
        self.strategy_overhead += overhead

    def remove_parent_child_listsize_predrank(self, num_states_removed, remove_parent_child_ranklist_cost):
        self.add_strategy_overhead(num_states_removed * remove_parent_child_ranklist_cost)

    def add_analysis_time(self, actual_class, path_analysis_time, confidence):
        self.analyzed_confidence += confidence
        if actual_class:
            self.num_sat_analyzed += 1
            self.total_time_analyze_sat += path_analysis_time
        else:
            self.num_unsat_analyzed += 1
            self.total_time_analyze_unsat += path_analysis_time

    def check_wait_finished(self, path_analysis_time):
        self.check_wait_consistent()
        assert(self.wait_time > 0)
        return (path_analysis_time <= self.wait_time)

    def update_worklist_size(self, new_size):
        self.worklist_max_size = max(self.worklist_max_size, new_size)
        self.worklist_curr_size = new_size

    def update_wait_finish_time(self, wait_finish, actual_class, path_analysis_time,
                                confidence):
        self.check_wait_consistent()
        assert(self.wait_time > 0)
        if wait_finish:
            self.wait_finish_confidence += confidence
            if actual_class:
                self.num_sat_wait_finish += 1
                self.time_wait_finish_sat += path_analysis_time
            else:
                self.num_unsat_wait_finish += 1
                self.time_wait_finish_unsat += path_analysis_time
            
            self.add_analysis_time(actual_class, path_analysis_time, confidence)
        else:
            self.wait_notfinished_confidence += confidence
            if actual_class:
                self.num_sat_wait_notfinished += 1
                self.time_wait_notfinished_sat += self.wait_time
            else:
                self.num_unsat_wait_notfinished += 1
                self.time_wait_notfinished_unsat += self.wait_time
        self.check_wait_consistent()

    def perform_wait(self, actual_class, path_analysis_time, confidence):
        self.check_wait_consistent()
        assert(self.wait_time > 0)

        wait_finish = self.check_wait_finished(path_analysis_time)
        self.update_wait_finish_time(wait_finish, actual_class, path_analysis_time,
                                     confidence)

        self.check_wait_consistent()
        return wait_finish

    def perform_filter_analyze(self, sat_rank, actual_class, path_analysis_time,
                               state_depth=None):
        passed = False
        if self.random_filter is not None:
            assert(self.pred_filter is None)
            passed = (random.random() <= self.random_filter)
        elif self.pred_filter is not None:
            assert(self.random_filter is None)
            is_less_than, pred_filter_val = self.pred_filter
            if not is_less_than:
                passed = (sat_rank >= pred_filter_val)
            else:
                passed = (sat_rank <= pred_filter_val)
        elif self.depth_filter is not None:
            assert(state_depth is not None)
            assert(self.ranklist_path_depth_sum is not None)
            assert(self.ranklist_num_paths is not None)
            if self.ranklist_num_paths == 0:
                passed = False
            else:
                avg_ranklist_path_depth = self.ranklist_path_depth_sum / float(self.ranklist_num_paths)
                passed = (state_depth >= avg_ranklist_path_depth + self.depth_filter)
        else: # No filter, no need to collect stats
            return passed

        if passed:
            self.long_analyze(actual_class, path_analysis_time, sat_rank) #

            if actual_class:
                self.num_filter_pass_sat += 1
                self.time_filter_pass_sat += path_analysis_time
            else:
                self.num_filter_pass_unsat += 1
                self.time_filter_pass_unsat += path_analysis_time
        else:
            if actual_class:
                self.num_filter_notpass_sat += 1
                self.time_filter_notpass_sat += path_analysis_time
            else:
                self.num_filter_notpass_unsat += 1
                self.time_filter_notpass_unsat += path_analysis_time
        return passed

    def use_wait_not_finished(self, actual_class):
        if self.wait_time == 0:
            return
        if actual_class:
            self.time_used_wait_notfinished_sat += self.wait_time
            assert(self.time_wait_notfinished_sat > 0)
        else:
            self.time_used_wait_notfinished_unsat += self.wait_time
            assert(self.time_wait_notfinished_unsat > 0)

    def long_analyze(self, actual_class, path_analysis_time, confidence):
        self.check_wait_consistent()
        self.long_analyze_confidence += confidence
        if actual_class:
            self.num_sat_long_analyze += 1
            self.time_long_analyze_sat += path_analysis_time
        else:
            self.num_unsat_long_analyze += 1
            self.time_long_analyze_unsat += path_analysis_time
        self.add_analysis_time(actual_class, path_analysis_time, confidence)
        self.check_wait_consistent()

    def skip(self, actual_class, path_analysis_time, confidence):
        self.check_wait_consistent()

        if actual_class:
            self.num_sat_missed += 1
            self.time_missed_sat += path_analysis_time
            self.missed_confidence += confidence
        else:
            self.num_unsat_saved += 1
            self.time_saved_unsat += path_analysis_time
            self.saved_confidence += confidence

    def avg_confidence(self):
        num_analyzed = self.num_sat_analyzed + self.num_unsat_analyzed
        self.analyzed_confidence = divide_none_on_zero(self.analyzed_confidence, num_analyzed)

        self.missed_confidence = divide_none_on_zero(self.missed_confidence, self.num_sat_missed)
        self.saved_confidence = divide_none_on_zero(self.saved_confidence, self.num_unsat_saved)

        num_wait_finish = self.num_sat_wait_finish + self.num_unsat_wait_finish
        self.wait_finish_confidence = divide_none_on_zero(self.wait_finish_confidence,num_wait_finish)

        num_wait_notfinished = self.num_sat_wait_notfinished + self.num_unsat_wait_notfinished
        self.wait_notfinished_confidence = divide_none_on_zero(self.wait_notfinished_confidence, num_wait_notfinished)

        num_long_analyze = self.num_sat_long_analyze + self.num_unsat_long_analyze
        self.long_analyze_confidence = divide_none_on_zero(self.long_analyze_confidence, num_long_analyze)

    def pop_state(self, state, actual_class):
        if actual_class:
            self.total_popped_sat_depth += state.state_depth
            self.total_sat_liststay_pops += state.list_stay_pop
            self.total_sat_liststsay_iters += state.list_stay_iter
            self.num_popped_sat += 1
        else:
            self.total_popped_unsat_depth += state.state_depth
            self.total_unsat_liststay_pops += state.list_stay_pop
            self.total_unsat_liststsay_iters += state.list_stay_iter 
            self.num_popped_unsat += 1

