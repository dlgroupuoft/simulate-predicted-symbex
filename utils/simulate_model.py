
import os
import sys
import random

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.common import get_tenth_time_ind

def simulate_hard_pred(args, actual_class, constr_solve_time):
    assert(args.sim_recall_dict is not None)

    time_ind = get_tenth_time_ind(constr_solve_time, len(TIME_LIST))
    if time_ind in args.sim_recall_dict:
        assert("all" not in args.sim_recall_dict)
        sim_sat_recall, sim_unsat_recall = args.sim_recall_dict[time_ind]
    else:
        assert("all" in args.sim_recall_dict)
        sim_sat_recall, sim_unsat_recall = args.sim_recall_dict["all"]
    assert(sim_sat_recall >= 0 and sim_sat_recall <= 1)
    assert(sim_unsat_recall >=0 and sim_unsat_recall <= 1)

    if actual_class:
        sim_recall = sim_sat_recall
        correct_pred = 1
        incorrect_pred = 0
    else:
        sim_recall = sim_unsat_recall
        correct_pred = 0
        incorrect_pred = 1

    if random.random() < sim_recall:
        return correct_pred
    else:
        return incorrect_pred

def simulate_confidence_shift(args, actual_class, sat_confidence):
    if args.shift_model_sat_unsat_threshold is not None:
        if actual_class:
            sat_confidence += args.shift_model_sat_unsat_threshold
        else:
            sat_confidence -= args.shift_model_sat_unsat_threshold
        sat_confidence = max(0, sat_confidence)
        sat_confidence = min(1, sat_confidence)
    return sat_confidence

def simulate_pred_sat_rank(args, actual_class, sat_confidence, constr_solve_time):
    if args.sim_recall_dict is not None:
        pred_sat_val = simulate_hard_pred(args, actual_class, constr_solve_time)
    elif args.shift_model_sat_unsat_threshold is not None:
        pred_sat_val = simulate_confidence_shift(args, actual_class, sat_confidence)
    else:
        pred_sat_val = sat_confidence
    return pred_sat_val


