
import numpy as np
from numpy import linalg as LA
from madnessToDaltony import *
import os
import pandas as pd
import json


class MadnessReader:

    def __init__(self):

        PROOT = os.getcwd()
        if not os.path.exists("dalton"):
            os.mkdir("dalton")
        with open(PROOT+'/molecules/frequency.json') as json_file:
            self.freq_json = json.loads(json_file.read())

    def __tensor_to_numpy(self, j):
        array = np.empty(j["size"])
        array[:] = j["vals"]
        return np.reshape(array, tuple(j["dims"]))

    def get_molresponse_json(self, response_info):
        """Takes in a response base json and returns parameters dict and protocol information dict"""
        protocol_data = response_info['protocol_data']
        response_parameters = response_info['response_parameters']
        n_states = response_parameters["states"]
        n_orbitals = response_parameters["num_orbitals"]
        num_protos = len(protocol_data)
        protos = []
        proto_data = []
        for p in range(num_protos):
            protos.append(protocol_data[p]["proto"])
            iter_data = protocol_data[p]["iter_data"]
            if response_parameters["excited_state"]:
                proto_data.append(self.__read_excited_proto_iter_data(
                    iter_data, n_states, n_orbitals))
            else:
                proto_data.append(self.__read_frequency_proto_iter_data(
                    iter_data, n_states, n_orbitals))
        return response_parameters, proto_data

    def __read_excited_proto_iter_data(self, protocol_data, num_states):
        num_protocols = protocol_data.__len__()
        dcol = []
        xcol = []
        ycol = []
        for i in range(num_states):
            dcol.append('d' + str(i))
            xcol.append('x' + str(i))
            ycol.append('y' + str(i))
        omega_dfs = []
        residual_dfs = []
        protos = []
        kprotos = []
        iters = []
        iter_p = 0
        for proto in protocol_data:
            protos.append(proto['proto'])
            kprotos.append(proto['k'])
            num_iters = proto['iter_data'].__len__()
            proto_array = np.ones((num_iters, 1)) * proto['proto']
            kproto_array = np.ones((num_iters, 1)) * proto['k']
            omega_array = np.empty((num_iters, num_states))
            dres = np.empty((num_iters, num_states))
            xres = np.empty((num_iters, num_states))
            yres = np.empty((num_iters, num_states))
            i = 0
            for iter in proto['iter_data']:
                # diagonalize the polarizability
                omega = self.__tensor_to_numpy(iter['omega']).flatten()
                # alpha=.5*(alpha+alpha.transpose())
                #w, v = LA.eig(alpha)
                #print("alpha : ",alpha)
                omega_array[i, :] = omega
                dres[i, :] = self.__tensor_to_numpy(
                    iter['density_residuals']).flatten()
                xres[i, :] = self.__tensor_to_numpy(iter['res_X']).flatten()
                yres[i, :] = self.__tensor_to_numpy(iter['res_Y']).flatten()
                i += 1
                iters.append(iter_p)
                iter_p += 1
            kproto_df = pd.DataFrame(kproto_array, columns=['k'])
            proto_df = pd.DataFrame(proto_array, columns=['thresh'])
            omega_df = pd.DataFrame(omega_array)
            omega_df = pd.concat([kproto_df, proto_df, omega_df], axis=1)
            dres_df = pd.DataFrame(dres, columns=dcol)
            xres_df = pd.DataFrame(xres, columns=xcol)
            yres_df = pd.DataFrame(yres, columns=ycol)
            residuals_df = pd.concat(
                [kproto_df, proto_df, dres_df, xres_df, yres_df], axis=1)
            omega_dfs.append(omega_df)
            residual_dfs.append(residuals_df)

        iters_df = pd.DataFrame(iters, columns=['iterations'])
        final_omega = pd.concat(omega_dfs, ignore_index=True)
        final_res = pd.concat(residual_dfs, ignore_index=True)
        final_omega = pd.concat([iters_df, final_omega], axis=1)
        final_res = pd.concat([iters_df, final_res], axis=1)
        return final_omega, final_res


    def __open_frequency_rbj(self, mol, xc, operator, freq):

        sfreq = "%f" % freq
        # first number before decimal
        f1 = sfreq.split('.')[0]
        # second number after decimal
        f2 = sfreq.split('.')[1]

        moldir = PROOT+'/'+xc+'/'+mol
        dfile = operator+'_'+xc+'_'+f1+'-'+f2
        jsonf = 'response_base.json'

        path = '/'.join([moldir, dfile, jsonf])

        with open(path) as json_file:
            response_j = json.loads(json_file.read())

        return response_j

    def __open_ground_json(self, mol, xc):

        moldir = PROOT+'/'+xc+'/'+mol
        jsonf = 'calc_info.json'

        path = '/'.join([moldir, jsonf])

        with open(path) as json_file:
            response_j = json.loads(json_file.read())

        return response_j

    def get_ground_scf_data(self, mol, xc):

        j = self.__open_ground_json(mol, xc)

        params = j['parameters']
        scf_e_data = j['scf_e_data']

        return params, scf_e_data

    def __open_excited_rbj(self, mol, xc, num_states):

        print(PROOT)
        moldir = PROOT+'/'+xc+'/'+mol
        dfile = "excited-"+str(num_states)
        jsonf = 'response_base.json'

        path = '/'.join([moldir, dfile, jsonf])

        with open(path) as json_file:
            response_j = json.loads(json_file.read())

        return response_j

    def __read_response_protocol_data(self, protocol_data: json, num_states):
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
        kprotos = []
        iters = []
        iter_p = 0
        for proto in protocol_data:
            protos.append(proto['proto'])
            kprotos.append(proto['k'])
            num_iters = proto['iter_data'].__len__()
            proto_array = np.ones((num_iters, 1)) * proto['proto']
            kproto_array = np.ones((num_iters, 1)) * proto['k']
            polar_data = np.empty((num_iters, 9))
            dres = np.empty((num_iters, num_states))
            xres = np.empty((num_iters, num_states))
            yres = np.empty((num_iters, num_states))
            i = 0
            for iter in proto['iter_data']:
                # diagonalize the polarizability
                alpha = self.__tensor_to_numpy(iter['polar']).flatten()
                # alpha=.5*(alpha+alpha.transpose())
                #w, v = LA.eig(alpha)
                #print("alpha : ",alpha)
                polar_data[i, :] = alpha
                dres[i, :] = self.__tensor_to_numpy(
                    iter['density_residuals']).flatten()
                xres[i, :] = self.__tensor_to_numpy(iter['res_X']).flatten()
                yres[i, :] = self.__tensor_to_numpy(iter['res_Y']).flatten()
                i += 1
                iters.append(iter_p)
                iter_p += 1
            kproto_df = pd.DataFrame(kproto_array, columns=['k'])
            proto_df = pd.DataFrame(proto_array, columns=['thresh'])
            polar_df = pd.DataFrame(polar_data,
                                    columns=['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz'])
            polar_df = pd.concat([kproto_df, proto_df, polar_df], axis=1)
            dres_df = pd.DataFrame(dres, columns=dcol)
            xres_df = pd.DataFrame(xres, columns=xcol)
            yres_df = pd.DataFrame(yres, columns=ycol)
            residuals_df = pd.concat(
                [kproto_df, proto_df, dres_df, xres_df, yres_df], axis=1)
            polar_dfs.append(polar_df)
            residual_dfs.append(residuals_df)

        iters_df = pd.DataFrame(iters, columns=['iterations'])
        final_polar = pd.concat(polar_dfs, ignore_index=True)
        final_res = pd.concat(residual_dfs, ignore_index=True)
        final_polar = pd.concat([iters_df, final_polar], axis=1)
        final_res = pd.concat([iters_df, final_res], axis=1)
        return final_polar, final_res

    def __get_polar_data(self, rbase_j):
        num_states = rbase_j['response_parameters']['states']
        fp, fres = self.__read_response_protocol_data(
            rbase_j['protocol_data'], num_states)
        return fp.iloc[-1, 1:].append(fres.iloc[-1, 3:])

    # TODO get the ground data
    def get_polar_result(self, mol, xc, operator):

        freq = freq_json[mol][xc][operator]
        fdata = {}
        for f in freq:
            rbasej = self.__open_frequency_rbj(mol, xc, operator, f)
            fdata[str(f)] = self.__get_polar_data(rbasej)

        rdf = pd.DataFrame(fdata).T
        return rdf

    def get_excited_data(self, mol, xc):

        num_states = freq_json[mol][xc]['excited-state']
        fdata = {}
        rbasej = self.__open_excited_rbj(mol, xc, num_states)
        omega,residuals = self.__read_excited_proto_iter_data(
            rbasej['protocol_data'], num_states)
        params = rbasej['parameters']
        return params,omega,residuals
    def get_excited_result(self, mol, xc):

        num_states = freq_json[mol][xc]['excited-state']
        params,omega,residuals=self.get_excited_data(mol,xc)











# input response_info json and returns a dict of response paramters
 # and a list of dicts of numpy arrays holding response data
class MadRunner:

    def run_response(self, mol, xc, operator):

        if operator == 'dipole':
            mad_command = 'mad-freq'
        elif operator == 'excited-state':
            mad_command = 'mad-excited'
            madnessCommand = ' '.join([mad_command, mol, xc])
        else:
            print('not implemented yet')
            return 1
        process = subprocess.Popen(
            madnessCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        return output, error

    def run_madness_ground(self, mol, xc):

        mad_command = 'database-moldft'
        madnessCommand = ' '.join([mad_command, mol, xc])
        process = subprocess.Popen(
            madnessCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        return output, error
