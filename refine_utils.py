import sys
import numpy as np

sys.path.append('approximate_archetypal_analysis/src')
from common import pose_utils as pu
from common import general
from core import op

ACTION_CLS_LIST = [
    "walking", "eating", "smoking", "discussion", "directions", "greeting",
    "phoning", "posing", "purchases", "sitting", "sittingdown", "photo",
    "waiting", "walkdog", "walktogether"
]


def get_idx_action(action):
    return ACTION_CLS_LIST.index(action.lower())


def normalize(arr):
    return np.asarray(
        op.to_unit_sphere(arr, method='l2'), dtype=np.float64, order='F')


def refine(output, dic, coeff_fun, verbose_info=False):
    """Refine poses by projecting them into simplices.

    Args:
        output: [batch_size, dim] poses array.
        dic: [dim, n_basis] dictionary.
        coeff_fun: A function accept parameters `(data, dic)` and compute the
            decomp coefficient.
        verbose_info: Whether or not compute extra information.

    Returns:
        A refined [batch_size, dim] poses array.
        A dictionary contains verbose information.
    """
    batch_size, _ = output.shape

    seq = output.reshape([batch_size, -1, 3])
    dist_matrix = np.stack(
        [pu.pose_to_matrix(pose, triu=True) for pose in seq], axis=1)
    dist_matrix = normalize(dist_matrix)
    coeff = coeff_fun(dist_matrix, dic)
    dist_matrix_recon = np.dot(dic, coeff)

    pose_recon = np.stack([
        pu.matrix_to_pose(dmat, triu=True) for dmat in dist_matrix_recon.T
    ]).reshape([batch_size, -1])

    extra_info = {}
    if verbose_info:
        extra_info['l2_err_dist'] = np.mean(
            np.sum(np.power(dist_matrix_recon - dist_matrix, 2), axis=0))
        extra_info['pose_err'] = np.mean(
            pu.align_and_err(pose_recon.T, output.T))
    return pose_recon, extra_info


get_refine_config = general.get_config
get_coeff_fun = general.get_coeff_fun
