import os
import sys
from collections import defaultdict
import numpy as np

sys.path.append('approximate_archetypal_analysis/src')
from common import pose_utils as pu
from common import human36m_utils as hu
from common import general
from core import cpp
from core import op

ACTION_CLS_LIST = [
    "walking", "eating", "smoking", "discussion", "directions", "greeting",
    "phoning", "posing", "purchases", "sitting", "sittingdown", "photo",
    "waiting", "walkdog", "walktogether"
]


def get_idx_action(action):
    return ACTION_CLS_LIST.index(action.lower())


def get_refine_model_config(opt):

    def recon_by_pca(data, model_data):
        model, data_mean = model_data['pca'], model_data['mean']
        recon = model.inverse_transform(
            model.transform(data - data_mean.T)) + data_mean.T
        return recon.T

    num_actions = 15
    recon_fun_map = {
        'pca': recon_by_pca,
    }
    if opt.refine_dir:
        train_dir = os.path.join(opt.refine_dir, 'train')
        model_data = general.get_pickle(os.path.join(train_dir, 'model.pkl'))
        recon_fun = recon_fun_map[opt.refine_method]
        return model_data, False, {k: recon_fun for k in range(num_actions)}, {}
    else:
        return None, False, defaultdict(lambda: None), {}


def refine_by_model(output, model_data, recon_fun, verbose_info=True):
    batch_size, _ = output.shape
    seq = hu.pose_preprocess(
        output.reshape([batch_size, -1, 3]),
        orthogonal=True).reshape([batch_size, -1])
    recons = recon_fun(seq, model_data).T

    align_list = []
    for recon, pose_gt in zip(recons, output):
        align = pu.align_to_gt(recon.reshape([-1, 3]), pose_gt.reshape([-1, 3]))
        align_list.append(align)
    pose_recon = np.stack(align_list).reshape([batch_size, -1])

    extra_info = {}
    if verbose_info:
        extra_info['l2_err_dist'] = np.mean(
            np.sum(np.power(recons - seq, 2), axis=1))
        extra_info['pose_err'] = np.mean(
            pu.align_and_err(pose_recon.T, output.T))
    return pose_recon, extra_info


def get_refine_config(opt):
    if opt.refine_dir:
        train_dir = os.path.join(opt.refine_dir, 'train')
        dic = general.get_pickle(os.path.join(train_dir, 'D.pkl'))['D']

        config = general.get_config(train_dir)
        per_action = config['per_action']
        to_dist_matrix = config.get('dist_matrix', False)
        data_params = config['data_params']
        actions = data_params[
            'actions'] if data_params['actions'] is not None else list(
                range(len(hu.ACTION_CLS_LIST)))

        if opt.refine_use_simplices:
            if opt.refine_method:
                raise ValueError()
            simp_path = os.path.join(opt.refine_dir, 'merged/merged.pkl')
            simp_data = general.get_pickle(simp_path)['merged']['simplices']
            method = 'simplices'

            if not per_action:
                basis_map = cpp.cal_basis_map(simp_data, dic.shape[1])
                simp_data = {k: simp_data for k in actions}
                basis_map = {k: basis_map for k in actions}
            else:
                basis_map = {
                    k: cpp.cal_basis_map(simp_data[k], dic[k].shape[1])
                    for k in actions
                }

            if opt.refine_penalty_fun == 'none':
                coeff_funs = {
                    # FIXME(meijieru): tune the parameters
                    k: (lambda data, dic: cpp.decomp_simplices_naive(data, dic, simp_data[k], basis_map[k], top_k=20))
                    for k in actions
                }
            else:
                penalty_fun = general.get_penalty_fun(opt.refine_penalty_fun)
                coeff_funs = {
                    # FIXME(meijieru): tune the parameters
                    k: (lambda data, dic: cpp.decomp_simplices(data, dic, simp_data[k], basis_map[k], top_k=20, penalty_fun=penalty_fun))
                    for k in actions
                }
        else:
            if opt.refine_method:
                method = opt.refine_method
                method_params = None
            else:  # fallback to training params
                method = config['method']
                method_params = config['method_params']
            coeff_fun = general.get_coeff_fun(method, method_params)
            coeff_funs = {k: coeff_fun for k in actions}
    else:
        dic = None
        per_action = False
        coeff_funs = defaultdict(lambda: None)
        to_dist_matrix = False

    return dic, per_action, coeff_funs, {
        'to_dist_matrix': to_dist_matrix,
        'use_simplices': opt.refine_use_simplices
    }


def refine(output,
           dic,
           coeff_fun,
           to_dist_matrix=False,
           use_simplices=False,
           verbose_info=False):
    """Refine poses by projecting them into simplices.

    Args:
        output: [batch_size, dim] poses array.
        dic: [dim, n_basis] dictionary.
        coeff_fun: A function accept parameters `(data, dic)` and compute the
            decomp coefficient.
        to_dist_matrix: Whether reconstruct the distance matrix or the pose.
        use_simplices: Whether use simplices reconstruction.
        verbose_info: Whether or not compute extra information.

    Returns:
        A refined [batch_size, dim] poses array.
        A dictionary contains verbose information.
    """
    batch_size, _ = output.shape
    seq = hu.pose_preprocess(
        output.reshape([batch_size, -1, 3]),
        orthogonal=True).reshape([batch_size, -1]).T
    seq = pu.data_preprocess(seq, to_dist_matrix=to_dist_matrix)
    coeff = coeff_fun(seq, dic)
    if use_simplices:
        recons = coeff['recon']
    else:
        recons = np.dot(dic, coeff)

    if to_dist_matrix:
        recons = [pu.matrix_to_pose(dmat, triu=True) for dmat in recons]

    align_list = []
    for recon, pose_gt in zip(recons.T, output):
        align = pu.align_to_gt(recon.reshape([-1, 3]), pose_gt.reshape([-1, 3]))
        align_list.append(align)
    pose_recon = np.stack(align_list).reshape([batch_size, -1])

    extra_info = {}
    if verbose_info:
        extra_info['l2_err_dist'] = np.mean(
            np.sum(np.power(recons - seq, 2), axis=0))
        extra_info['pose_err'] = np.mean(
            pu.align_and_err(pose_recon.T, output.T))
    return pose_recon, extra_info


plot_pose_seq = hu.plot_pose_seq
convert_to_pose_16 = hu.convert_to_pose_16
