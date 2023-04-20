import os
import numpy as np
import torch
import pickle


def tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x
    x = torch.tensor(x, dtype=torch.float32).to(device)
    return x


def to_np(t):
    return t.cpu().detach().numpy()


def random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_thread_count(count=1):
    os.environ['OMP_NUM_THREADS'] = "{}".format(count)
    os.environ['MKL_NUM_THREADS'] = "{}".format(count)
    torch.set_num_threads(count)


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def get_unpicklable(instance, exception=None, string='', first_only=True):
    """
    Recursively go through all attributes of instance and return a list of whatever
    can't be pickled.

    Set first_only to only print the first problematic element in a list, tuple or
    dict (otherwise there could be lots of duplication).
    """
    problems = []
    if isinstance(instance, tuple) or isinstance(instance, list):
        for k, v in enumerate(instance):
            try:
                pickle.dumps(v)
            except BaseException as e:
                problems.extend(get_unpicklable(v, e, string + f'[{k}]'))
                if first_only:
                    break
    elif isinstance(instance, dict):
        for k in instance:
            try:
                pickle.dumps(k)
            except BaseException as e:
                problems.extend(get_unpicklable(
                    k, e, string + f'[key type={type(k).__name__}]'
                ))
                if first_only:
                    break
        for v in instance.values():
            try:
                pickle.dumps(v)
            except BaseException as e:
                problems.extend(get_unpicklable(
                    v, e, string + f'[val type={type(v).__name__}]'
                ))
                if first_only:
                    break
    else:
        for k, v in instance.__dict__.items():
            try:
                pickle.dumps(v)
            except BaseException as e:
                problems.extend(get_unpicklable(v, e, string + '.' + k))

    # if we get here, it means pickling instance caused an exception (string is not
    # empty), yet no member was a problem (problems is empty), thus instance itself
    # is the problem.
    if string != '' and not problems:
        problems.append(
            string + f" (Type '{type(instance).__name__}' caused: {exception})"
        )

    return problems
