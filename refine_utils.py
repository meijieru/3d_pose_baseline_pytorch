import sys
import numpy as np

sys.path.append('approximate_archetypal_analysis/src')
from common import pose_utils as pu
from common import general
from core import op


def normalize(arr):
    return np.asarray(
        op.to_unit_sphere(arr, method='l2'), dtype=np.float64, order='F')


def refine(output, dic, method='optimized'):
    """Refine poses by projecting them into simplices.

    Args:
        output: [batch_size, dim] poses array.
        dic: [dim, n_basis] dictionary.
        method: Method for computing the coefficient.

    Returns:
        A refined [batch_size, dim] poses array.
    """
    coeff_fun = general.get_coeff_fun(method)
    batch_size, _ = output.shape

    seq = output.reshape([batch_size, -1, 3])
    dist_matrix = np.stack(
        [pu.pose_to_matrix(pose, triu=True) for pose in seq], axis=1)
    dist_matrix = normalize(dist_matrix)
    coeff = coeff_fun(dist_matrix, dic)
    dist_matrix_recon = np.dot(dic, coeff)

    pose_recon = np.stack(
        [pu.matrix_to_pose(dmat, triu=True) for dmat in dist_matrix_recon.T])
    return pose_recon.reshape([batch_size, -1])
