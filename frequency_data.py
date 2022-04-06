#!/usr/bin/env python
# coding: utf-8
import json
import os.path
import numpy as np
import pandas as pd
from read_madness_json import read_molrespone_json, tensor_to_numpy

response_dir = os.path.curdir
moldft_dir = response_dir + "/.."
mol_name = os.path.realpath(moldft_dir).split("/")[-1]
moldft_json = moldft_dir + "/calc_info.json"
with open(moldft_json, "r") as json_file:
    moldft_j = json.load(json_file)

response_json = "response_base.json"

with open(response_json, "r") as json_file:
    response_j = json.load(json_file)
    r_params, proto_data = read_molrespone_json(response_j)

num_states = r_params["states"]
num_orbitals = r_params["num_orbitals"]
num_orders = 2

orb_energies = tensor_to_numpy(moldft_j["scf_eigenvalues_a"])


def read_response_protocol_data(protocol_data: json, num_states):
    num_protocols = protocol_data.__len__()
    dcol = []
    xcol = []
    ycol = []
    for i in range(num_states):
        dcol.append('d' + str(i))
        xcol.append('x' + str(i))
        ycol.append('y' + str(i))
    polar_dfs = []
    residual_dfs = []
    protos = []
    iters = []
    iter_p = 0
    for proto in response_j["protocol_data"]:
        protos.append(proto['proto'])
        num_iters = proto['iter_data'].__len__()
        proto_array = np.ones((num_iters, 1)) * proto['proto']
        polar_data = np.empty((num_iters, 9))
        dres = np.empty((num_iters, num_states))
        xres = np.empty((num_iters, num_states))
        yres = np.empty((num_iters, num_states))
        i = 0
        for iter in proto['iter_data']:
            polar_data[i, :] = tensor_to_numpy(iter['polar']).flatten()
            dres[i, :] = tensor_to_numpy(iter['density_residuals']).flatten()
            xres[i, :] = tensor_to_numpy(iter['res_X']).flatten()
            yres[i, :] = tensor_to_numpy(iter['res_Y']).flatten()
            i += 1
            iters.append(iter_p)
            iter_p += 1
        proto_df = pd.DataFrame(proto_array, columns=['protocol'])
        polar_df = pd.DataFrame(polar_data,
                                columns=['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz'])
        polar_df = pd.concat([proto_df, polar_df], axis=1)
        dres_df = pd.DataFrame(dres, columns=dcol)
        xres_df = pd.DataFrame(xres, columns=xcol)
        yres_df = pd.DataFrame(yres, columns=ycol)
        residuals_df = pd.concat([proto_df, dres_df, xres_df, yres_df], axis=1)
        polar_dfs.append(polar_df)
        residual_dfs.append(residuals_df)

    iters_df = pd.DataFrame(iters, columns=['iterations'])
    final_polar = pd.concat(polar_dfs, ignore_index=True)
    final_res = pd.concat(residual_dfs, ignore_index=True)
    final_polar = pd.concat([iters_df, final_polar], axis=1)
    final_res = pd.concat([iters_df, final_res], axis=1)
    return final_polar, final_res


polar_df, residuals = read_response_protocol_data(response_j, num_states)

polar_df.to_csv('polar.csv')
residuals.to_csv('res.csv')
polar_df.to_excel('polar.xlsx')
residuals.to_excel('residuals.xlsx')
# data.to_csv('data.csv')
