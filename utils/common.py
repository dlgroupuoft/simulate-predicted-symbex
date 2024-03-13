
import os
import pickle
import sys

EPSILON = 0.001

# ======================================================================

def safe_pickle(data, file):
    """ 
    Pickle big files safely, processing them in chunks
    :param data: data to be pickled
    :param file: file to pickle it into
    """
    max_bytes = 2**31 - 1 
    pickle_out = pickle.dumps(data)
    n_bytes = sys.getsizeof(pickle_out)
    with open(file, 'wb') as f:
        count = 0 
        for i in range(0, n_bytes, max_bytes):
            f.write(pickle_out[i:min(n_bytes, i + max_bytes)])
            count += 1
    print("Saved pickle file: {0}".format(file))

def load_pickle_res(pickle_filename):
    print("Loading {0}...".format(pickle_filename))
    with open(pickle_filename, "rb") as in_file:
        pickle_res = pickle.load(in_file)
    return pickle_res

def get_tenth_time_ind(time_left, tenth_list_len):
    time_ind = 0
    while time_left >= 10:
        if time_ind == tenth_list_len - 1:
            break
        time_left = time_left // 10
        time_ind += 1
    return time_ind

def apprx_equal(num1, num2):
    return (num1 <= num2 * (1 + EPSILON)) and (num1 >= num2 * (1 - EPSILON))


def divide_none_on_zero(numerator, denominator):
    if denominator == 0:
        return None
    else:
        return numerator / denominator

