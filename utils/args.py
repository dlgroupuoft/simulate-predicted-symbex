
import os
import sys

import argparse
import re

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.common import load_pickle_res

# ==============================================================================================

def parse_wait_filter(wait_filter_str, has_dep):
    random_filter, pred_filter, depth_filter = None, None, None
    wait_time = 0
    if wait_filter_str is not None:
        if wait_filter_str.startswith("randomfilter"):
            filter_match = re.match(r"randomfilter([0-9\.]+)", wait_filter_str)
            random_filter = float(filter_match.group(1))
            assert(random_filter >= 0 and random_filter <= 1)
        elif wait_filter_str.startswith("predfilter"):
            filter_match = re.match(r"predfilter(\-?)([0-9\.]+)", wait_filter_str)
            pred_filter = float(filter_match.group(2))
            assert(pred_filter >= 0 and pred_filter <= 1)
            is_less_than = (len(filter_match.group(1)) > 0)
            pred_filter = (is_less_than, pred_filter)
        elif wait_filter_str.startswith("waittime"):
            filter_match = re.match(r"waittime([0-9]+)", wait_filter_str)
            wait_time = int(filter_match.group(1))
        elif has_dep and wait_filter_str.startswith("depthfilter"):
            filter_match = re.match(r"depthfilter([0-9]+)", wait_filter_str)
            depth_filter = int(filter_match.group(1))
        else:
            assert(False)

    return random_filter, pred_filter, depth_filter, wait_time

# ==============================================================================================

def check_general_args(args):
    if args.list_size is not None:
        assert(args.list_size > 0)

def check_nodep_input_args(args):
    if args.mode != 'NOPRED':
        assert(not args.hard_pred_skip_unknown)

def check_dep_input_args(args):
    if args.rank_objective == 'sat' and args.mode == 'DEP_RANK':
        print("!!! WARNING: sat_rank would not achieve speedup with DEP_RANK !!!")
    if args.rankDepthMinuend is not None:
        assert(args.rankUnsatRatio is not None)
    assert(args.rankPrevBack is None or args.rankPrevBack > 0)

# ==============================================================================================

def add_print_args(p):
    p.add_argument('--app_progress_time_budget_filename', type=str)
    p.add_argument('--app_fold_dir', type=str)
    p.add_argument('--curr_fold_num', type=int)
    p.add_argument('--app_total_fold_num', default=5, type=int)

    p.add_argument('--app_progress_time_budget', default=None, type=dict)
    p.add_argument('--app_curr_fold', default=None, type=set)

def add_general_args(p):
    p.add_argument('--list_size', default=None, type=int)
    p.add_argument('--wait_filter_str', default=None, type=str)

    p.add_argument('--sim_recall_dict_filename', default=None, type=str)
    p.add_argument('--sim_recall_dict', default=None)
    p.add_argument('--shift_model_sat_unsat_threshold', default=None, type=float)
    p.add_argument('--hard_pred_confidence_threshold', default=0.5, type=float)

def add_nodep_input_args(p):
    p.add_argument('--app_info_filename', default='app_info_list.p', type=str)
    p.add_argument('--app_path_time_filename', default='app_path_time.p', type=str)
    p.add_argument('--app_pred_filename', default='randForest_mix_app_pred_list.p', type=str)
    p.add_argument('--out_basename', default='simulate_out', type=str)

    p.add_argument('--mode', choices=['NO_PRED', 'HARD_PRED', 'SAT_RANK', 'RANDOM_RANK'], type=str)
    p.add_argument('--hard_pred_skip_unknown', action='store_true')

def add_dep_input_args(p):
    p.add_argument('--preds_dir', default='preds_keepunsat', type=str)
    p.add_argument('--state_cfg_time_dir', default='state_cfg_time', type=str)
    p.add_argument('--out_filename', default='simulate_out.p', type=str)

    p.add_argument('--mode', choices=['NOPRED', 'HARDPRED', 'DEP_RANK', 'INDEP_RANK', 'REMOVEDEP_LISTSIZE'] , type=str)
    p.add_argument('--no_path_prune', action='store_true')
    p.add_argument('--rank_objective', choices=['sat', 'unsat', 'random', 'hybrid'], type=str)
    p.add_argument('--broad_start', action='store_true')

    p.add_argument('--rankEarlyStateFirst', action='store_true', help="Rank early state first if this argument is set. Otherwise, rank states randomly.")
    p.add_argument('--rankUnsatRatio', default=None, type=float)
    p.add_argument('--rankDepthMinuend', default=None, type=float)
    p.add_argument('--rankDepthReset', action='store_true')
    p.add_argument('--rankPrevBack', default=None, type=int)
    p.add_argument('--rankPrevSatDiffRatio', default=None, type=float)
    p.add_argument('--rankNoAnalyzePrev', action='store_true')
    p.add_argument('--rankClearWindowUnsatParent', action='store_true')

# ==============================================================================================

def parse_args(has_path_dep=False, is_print_args=False):
    p = argparse.ArgumentParser()
    if is_print_args:
        add_print_args(p)
    else:
        add_general_args(p)
        if has_path_dep:
            add_dep_input_args(p)
        else:
            add_nodep_input_args(p)
    args = p.parse_args()

    #
    if is_print_args:
        args.app_progress_time_budget = load_pickle_res(args.app_progress_time_budget_filename)
        if args.app_fold_dir is not None:
            app_fold_filename = os.path.join(args.app_fold_dir, "app_split_{0}folds.p".format(args.app_total_fold_num))
            app_fold_split = load_pickle_res(app_fold_filename)
            assert(args.curr_fold_num >= 0 and args.curr_fold_num < len(app_fold_split))
            args.app_curr_fold = set(app_fold_split[args.curr_fold_num])
            print("len(args.app_curr_fold): {0}".format(len(args.app_curr_fold)))
    else:
        check_general_args(args)
        if has_path_dep:
            check_dep_input_args(args)
        else:
            check_nodep_input_args(args)

        #
        out_basefile = args.out_basename if not has_path_dep else args.out_filename
        out_dir = os.path.dirname(out_basefile)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        if args.sim_recall_dict_filename is not None:
            args.sim_recall_dict = load_pickle_res(args.sim_recall_dict_filename)
        assert((args.sim_recall_dict is None) or (args.shift_model_sat_unsat_threshold is None))

    return args

