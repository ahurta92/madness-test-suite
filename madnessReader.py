import json
import seaborn as sns
import os

import matplotlib.pyplot as plt

from daltonRunner import DaltonRunner

import numpy as np
import pandas as pd

from madnessToDaltony import *
from scipy.stats import norm

import statistics


class MadnessReader:
    def __init__(self):

        PROOT = os.getcwd()
        if not os.path.exists("dalton"):
            os.mkdir("dalton")
        with open(PROOT + "/molecules/frequency.json") as json_file:
            self.freq_json = json.loads(json_file.read())

    def __tensor_to_numpy(self, j):
        array = np.empty(j["size"])
        array[:] = j["vals"]
        return np.reshape(array, tuple(j["dims"]))

    def get_molresponse_json(self, response_info):
        """Takes in a response base json and returns parameters dict and protocol information dict"""
        protocol_data = response_info["protocol_data"]
        response_parameters = response_info["response_parameters"]
        n_states = response_parameters["states"]
        n_orbitals = response_parameters["num_orbitals"]
        num_protos = len(protocol_data)
        protos = []
        proto_data = []
        for p in range(num_protos):
            protos.append(protocol_data[p]["proto"])
            iter_data = protocol_data[p]["iter_data"]
            if response_parameters["excited_state"]:
                proto_data.append(
                    self.__read_excited_proto_iter_data(iter_data, n_states, n_orbitals)
                )
            else:
                proto_data.append(
                    self.__read_frequency_proto_iter_data(
                        iter_data, n_states, n_orbitals
                    )
                )
        return response_parameters, proto_data

    def __read_excited_proto_iter_data(self, protocol_data, num_states):
        num_protocols = protocol_data.__len__()
        dcol = []
        xcol = []
        ycol = []
        for i in range(num_states):
            dcol.append("d" + str(i))
            xcol.append("x" + str(i))
            ycol.append("y" + str(i))

        omega_dfs = []
        residual_dfs = []

        bsh_norms_dfs = []
        d_norms_dfs = []

        protos = []
        kprotos = []
        iters = []
        iter_p = 0
        num_iters_per_protocol = []
        for proto in protocol_data:
            protos.append(proto["proto"])
            kprotos.append(proto["k"])
            num_iters = proto["iter_data"].__len__()
            num_iters_per_protocol.append(num_iters)
            proto_array = np.ones((num_iters, 1)) * proto["proto"]
            kproto_array = np.ones((num_iters, 1)) * proto["k"]
            omega_array = np.empty((num_iters, num_states))
            dres = np.empty((num_iters, num_states))
            xres = np.empty((num_iters, num_states))
            yres = np.empty((num_iters, num_states))

            chi_x_norms = np.empty((num_iters, num_states))
            chi_y_norms = np.empty((num_iters, num_states))
            rho_norms = np.empty((num_iters, num_states))

            i = 0
            for iter in proto["iter_data"]:
                # diagonalize the polarizability
                omega = self.__tensor_to_numpy(iter["omega"]).flatten()
                # alpha=.5*(alpha+alpha.transpose())
                # w, v = LA.eig(alpha)
                # print("alpha : ",alpha)
                omega_array[i, :] = omega
                dres[i, :] = self.__tensor_to_numpy(iter["density_residuals"]).flatten()
                xres[i, :] = self.__tensor_to_numpy(iter["res_X"]).flatten()
                yres[i, :] = self.__tensor_to_numpy(iter["res_Y"]).flatten()

                try:

                    chi_x_norms[i, :] = self.__tensor_to_numpy(iter["chi_norms_x"]).flatten()
                    chi_y_norms[i, :] = self.__tensor_to_numpy(iter["chi_norms_y"]).flatten()
                except:
                    chi_x_norms[i, :] = self.__tensor_to_numpy(iter["chi_norms"]).flatten()
                    chi_y_norms[i, :] = self.__tensor_to_numpy(iter["rel_chi_norms"]).flatten()
                rho_norms[i, :] = self.__tensor_to_numpy(iter["rho_norms"]).flatten()

                i += 1
                iters.append(iter_p)
                iter_p += 1
            for j in range(1, num_iters_per_protocol.__len__()):
                num_iters_per_protocol[j] = (
                        num_iters_per_protocol[j] + num_iters_per_protocol[j - 1]
                )
            # num_iters_per_protocol[-1] -= 1
            kproto_df = pd.DataFrame(kproto_array, columns=["k"])
            proto_df = pd.DataFrame(proto_array, columns=["thresh"])
            omega_df = pd.DataFrame(omega_array)
            omega_df = pd.concat([kproto_df, proto_df, omega_df], axis=1)
            dres_df = pd.DataFrame(dres, columns=dcol)
            xres_df = pd.DataFrame(xres, columns=xcol)
            yres_df = pd.DataFrame(yres, columns=ycol)

            rho_norms_df = pd.DataFrame(rho_norms, columns=dcol)

            chi_x_norms_df = pd.DataFrame(chi_x_norms, columns=xcol)
            chi_y_norms_df = pd.DataFrame(chi_y_norms, columns=ycol)

            bsh_norms_df = pd.concat([chi_x_norms_df, chi_y_norms_df], axis=1)
            residuals_df = pd.concat(
                [kproto_df, proto_df, dres_df, xres_df, yres_df], axis=1
            )
            omega_dfs.append(omega_df)
            residual_dfs.append(residuals_df)

        iters_df = pd.DataFrame(iters, columns=["iterations"])
        final_omega = pd.concat(omega_dfs, ignore_index=True)
        final_res = pd.concat(residual_dfs, ignore_index=True)
        final_omega = pd.concat([iters_df, final_omega], axis=1)
        final_res = pd.concat([iters_df, final_res], axis=1)
        return final_omega, final_res, num_iters_per_protocol, rho_norms_df, bsh_norms_df

    def __open_frequency_rbj(self, mol, xc, operator, freq):

        sfreq = "%f" % freq
        # first number before decimal
        f1 = sfreq.split(".")[0]
        # second number after decimal
        f2 = sfreq.split(".")[1]

        moldir = PROOT + "/" + xc + "/" + mol
        dfile = operator + "_" + xc + "_" + f1 + "-" + f2
        jsonf = "response_base.json"

        path = "/".join([moldir, dfile, jsonf])

        with open(path) as json_file:
            response_j = json.loads(json_file.read())

        return response_j

    def __open_ground_json(self, mol, xc):

        moldir = PROOT + "/" + xc + "/" + mol
        jsonf = "calc_info.json"

        path = "/".join([moldir, jsonf])
        # print("mad_path",path)

        with open(path) as json_file:
            response_j = json.loads(json_file.read())

        return response_j

    def get_ground_scf_data(self, mol, xc):

        j = self.__open_ground_json(mol, xc)

        params = j["parameters"]
        scf_e_data = j["scf_e_data"]
        timing = j["wall_time"]

        return params, scf_e_data, timing, j

    def __open_excited_rbj(self, mol, xc, num_states):

        # print(PROOT)
        moldir = PROOT + "/" + xc + "/" + mol
        dfile = "excited-" + str(num_states)
        jsonf = "response_base.json"

        path = "/".join([moldir, dfile, jsonf])

        with open(path) as json_file:
            response_j = json.loads(json_file.read())

        return response_j

    def __read_response_protocol_data(self, protocol_data: json, num_states):
        num_protocols = protocol_data.__len__()
        dcol = []
        xcol = []
        ycol = []
        for i in range(num_states):
            dcol.append("d" + str(i))
            xcol.append("x" + str(i))
            ycol.append("y" + str(i))
        polar_dfs = []
        residual_dfs = []

        bsh_norms_dfs = []
        d_norms_dfs = []

        protos = []
        kprotos = []
        iters = []
        iter_p = 0
        num_iters_per_protocol = []
        for proto in protocol_data:
            protos.append(proto["proto"])
            kprotos.append(proto["k"])
            num_iters = proto["iter_data"].__len__()
            num_iters_per_protocol.append(num_iters)
            proto_array = np.ones((num_iters, 1)) * proto["proto"]
            kproto_array = np.ones((num_iters, 1)) * proto["k"]
            polar_data = np.empty((num_iters, 9))
            dres = np.empty((num_iters, num_states))
            xres = np.empty((num_iters, num_states))
            yres = np.empty((num_iters, num_states))

            chi_x_norms = np.empty((num_iters, 3))
            chi_y_norms = np.empty((num_iters, 3))
            rho_norms = np.empty((num_iters, 3))

            i = 0
            for iter in proto["iter_data"]:
                # diagonalize the polarizability
                alpha = self.__tensor_to_numpy(iter["polar"]).flatten()
                # alpha=.5*(alpha+alpha.transpose())
                # w, v = LA.eig(alpha)
                # print("alpha : ",alpha)
                polar_data[i, :] = alpha

                dres[i, :] = self.__tensor_to_numpy(iter["density_residuals"]).flatten()
                xres[i, :] = self.__tensor_to_numpy(iter["res_X"]).flatten()
                yres[i, :] = self.__tensor_to_numpy(iter["res_Y"]).flatten()

                chi_x_norms[i, :] = self.__tensor_to_numpy(iter["chi_norms_x"]).flatten()
                chi_y_norms[i, :] = self.__tensor_to_numpy(iter["chi_norms_y"]).flatten()
                rho_norms[i, :] = self.__tensor_to_numpy(iter["rho_norms"]).flatten()

                i += 1
                iters.append(iter_p)
                iter_p += 1
            kproto_df = pd.DataFrame(kproto_array, columns=["k"])
            proto_df = pd.DataFrame(proto_array, columns=["thresh"])
            polar_df = pd.DataFrame(
                polar_data,
                columns=["xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"],
            )
            polar_df = pd.concat([kproto_df, proto_df, polar_df], axis=1)
            dres_df = pd.DataFrame(dres, columns=dcol)
            xres_df = pd.DataFrame(xres, columns=xcol)
            yres_df = pd.DataFrame(yres, columns=ycol)

            rho_norms_df = pd.DataFrame(rho_norms, columns=dcol)

            chi_x_norms_df = pd.DataFrame(chi_x_norms, columns=xcol)
            chi_y_norms_df = pd.DataFrame(chi_y_norms, columns=ycol)

            bsh_norms_df = pd.concat([chi_x_norms_df, chi_y_norms_df], axis=1)

            residuals_df = pd.concat(
                [kproto_df, proto_df, dres_df, xres_df, yres_df], axis=1
            )
            polar_dfs.append(polar_df)
            residual_dfs.append(residuals_df)

            d_norms_dfs.append(rho_norms_df)
            bsh_norms_dfs.append(bsh_norms_df)

        iters_df = pd.DataFrame(iters, columns=["iterations"])
        final_polar = pd.concat(polar_dfs, ignore_index=True)
        final_res = pd.concat(residual_dfs, ignore_index=True)
        final_polar = pd.concat([iters_df, final_polar], axis=1)
        final_res = pd.concat([iters_df, final_res], axis=1)
        final_d_norms = pd.concat(d_norms_dfs, ignore_index=True)
        final_bsh_norms = pd.concat(bsh_norms_dfs, ignore_index=True)

        return final_polar, final_res, num_iters_per_protocol, final_d_norms, final_bsh_norms

    def __get_polar_data(self, rbase_j):
        num_states = rbase_j["parameters"]["states"]
        (
            freq_data,
            residuals,
            num_iters_per_protocol,
            d_norms_df,
            bsh_norms_df

        ) = self.__read_response_protocol_data(rbase_j["protocol_data"], num_states)
        params = rbase_j["parameters"]
        return params, freq_data, residuals, num_iters_per_protocol, d_norms_df, bsh_norms_df

    # TODO get the ground data
    def get_polar_result(self, mol, xc, operator):

        freq = freq_json[mol][xc][operator]
        full_freq_data = {}
        fdata = {}

        thresh_data = {}
        k_data = {}
        iter_data = {}
        d_res_data = {}
        bsh_res_data = {}

        d_norms_data = {}
        bsh_norms_data = {}

        wall_time = {}
        cpu_time = {}
        converged = {}
        num_iter_proto = {}
        params = None
        full_response_base = {}

        for f in freq:
            try:
                rbasej = self.__open_frequency_rbj(mol, xc, operator, f)
                full_response_base[f] = rbasej

                converged_f = rbasej["converged"]
                time_data_f = rbasej["time_data"]
                num_states = rbasej["parameters"]["states"]

                (
                    params,
                    freq_data,
                    residuals,
                    num_iters_per_protocol,
                    d_norms_df,
                    bsh_norms_df
                ) = self.__get_polar_data(rbasej)
                # for j in range(0, num_iters_per_protocol.__len__()):
                #    num_iters_per_protocol[j] += 1

                # for j in range(1, num_iters_per_protocol.__len__()):
                #    num_iters_per_protocol[j] = num_iters_per_protocol[j] + num_iters_per_protocol[j - 1]

                num_iter_proto[str(f)] = num_iters_per_protocol

                fdata[str(f)] = pd.DataFrame(freq_data)
                fdata[str(f)] = freq_data.iloc[-1, :]

                iterations = residuals.iloc[:, 0:1]
                k = residuals.iloc[:, 1:2]
                thresh = residuals.iloc[:, 2:3]
                d_residuals = residuals.iloc[:, 3: (3 + num_states)]
                bsh_residuals = residuals.iloc[:, (3 + num_states):]

                k_data[str(f)] = pd.DataFrame(k)
                converged[str(f)] = converged_f
                iter_data[str(f)] = pd.DataFrame(iterations)
                thresh_data[str(f)] = pd.DataFrame(thresh)

                d_norms_data[str(f)] = pd.DataFrame(d_norms_df)
                bsh_norms_data[str(f)] = pd.DataFrame(bsh_norms_df)

                d_res_data[str(f)] = pd.DataFrame(d_residuals)
                total_iters = sum(num_iters_per_protocol)
                d_res_data[str(f)].index = [str(i) for i in range(1, total_iters + 1)]

                bsh_res_data[str(f)] = pd.DataFrame(bsh_residuals)
                bsh_res_data[str(f)].index = [str(i) for i in range(1, total_iters + 1)]

                cpu_time_data_f = time_data_f["cpu_time"]
                wall_time_data_f = time_data_f["wall_time"]
                cpu_dict = {}
                for k, v in cpu_time_data_f.items():
                    cpu_dict[k] = v[0]

                cpu_time[str(f)] = pd.DataFrame(cpu_dict)

                wall_time_dict = {}
                for k, v in wall_time_data_f.items():
                    wall_time_dict[k] = v[0]

                wall_time[str(f)] = pd.DataFrame(wall_time_dict)
            except FileNotFoundError as not_found:
                print(f, " not found:", not_found)
                rdf = pd.DataFrame(fdata).T
                return (
                    params,
                    iter_data,
                    k_data,
                    thresh_data,
                    d_res_data,
                    bsh_res_data,
                    full_freq_data,
                    rdf,
                    wall_time,
                    cpu_time,
                    pd.Series(converged),
                    num_iter_proto,
                    d_norms_data,
                    bsh_norms_data,
                    full_response_base

                )

        rdf = pd.DataFrame(fdata).T
        return (
            params,
            iter_data,
            k_data,
            thresh_data,
            d_res_data,
            bsh_res_data,
            full_freq_data,
            rdf,
            wall_time,
            cpu_time,
            pd.Series(converged),
            num_iter_proto,
            d_norms_data,
            bsh_norms_data,
            full_response_base
        )

    def get_excited_data(self, mol, xc):

        num_states = freq_json[mol][xc]["excited-state"]
        fdata = {}
        rbasej = self.__open_excited_rbj(mol, xc, num_states)
        converged = rbasej["converged"]
        omega, residuals, num_iter_proto, d_norms_df, bsh_norms_df = self.__read_excited_proto_iter_data(
            rbasej["protocol_data"], num_states
        )
        params = rbasej["parameters"]
        time_data_f = rbasej["time_data"]
        cpu_time_data_f = time_data_f["cpu_time"]
        wall_time_data_f = time_data_f["wall_time"]

        cpu_dict = {}
        for k, v in cpu_time_data_f.items():
            cpu_dict[k] = v[0]

        cpu_time = pd.DataFrame(cpu_dict)

        wall_time_dict = {}
        for k, v in wall_time_data_f.items():
            wall_time_dict[k] = v[0]

        wall_time = pd.DataFrame(wall_time_dict)
        return params, omega, residuals, wall_time, cpu_time, converged, num_iter_proto, rbasej, d_norms_df, bsh_norms_df

    def get_excited_result(self, mol, xc):

        num_states = freq_json[mol][xc]["excited-state"]
        (
            params,
            full_omega,
            residuals,
            cpu_time,
            wall_time,
            converged,
            num_iter_proto,
            response_base,
            d_norms_df,
            bsh_norms_df
        ) = self.get_excited_data(mol, xc)
        iterations = residuals.iloc[:, 0:1]
        k = residuals.iloc[:, 1:2]
        thresh = residuals.iloc[:, 2:3]
        d_residuals = residuals.iloc[:, 3: (3 + num_states)]
        bsh_residuals = residuals.iloc[:, (3 + num_states):]

        return (
            params,
            iterations,
            k,
            thresh,
            d_residuals,
            bsh_residuals,
            full_omega.iloc[-1, 3:],
            full_omega,
            wall_time,
            cpu_time,
            converged,
            num_iter_proto,
            response_base,
            d_norms_df,
            bsh_norms_df
        )


class FrequencyData:
    def __init__(self, mol, xc, operator):
        self.response_base = {}
        self.calc_info = None
        self.dalton_data = {}
        self.mol = mol
        self.xc = xc
        self.operator = operator
        mad_reader = MadnessReader()

        (
            self.ground_params,
            self.ground_scf_data,
            self.ground_timing,
            self.calc_info,
        ) = mad_reader.get_ground_scf_data(mol, xc)
        e_name_list = ["e_coulomb", "e_kinetic", "e_local", "e_nrep", "e_tot"]
        self.ground_e = {}
        for e_name in e_name_list:
            self.ground_e[e_name] = self.ground_scf_data[e_name][-1]
        (
            self.params,
            self.iter_data,
            self.k_data,
            self.thresh_data,
            self.d_residuals,
            self.bsh_residuals,
            self.full_polar,
            self.polar_df,
            self.wall_time,
            self.cpu_time,
            self.converged,
            self.num_iter_proto,
            self.d_norms,
            self.bsh_norms,
            self.response_base

        ) = mad_reader.get_polar_result(mol, xc, operator)
        self.num_states = self.params["states"]

    def plot_density_residuals(self):

        for f, r_df in self.d_residuals.items():
            r_df.plot(
                title=str(self.mol) + " frequency density residual plot: " + str(f),
                logy=True,
            )

    def plot_bsh_residuals(self):

        for f, r_df in self.bsh_residuals.items():
            r_df.plot(
                title=str(self.mol) + "Frequency BSH Residual plot: " + str(f),
                logy=True,
            )

    def final_bsh_residuals(self):
        val = {}
        for f, d in self.bsh_residuals.items():
            val[f] = d.iloc[-1, :]
        val = pd.DataFrame(val).T
        return val

    def final_density_residuals(self):
        val = {}
        for f, d in self.d_residuals.items():
            val[f] = d.iloc[-1, :]
        val = pd.DataFrame(val).T

        newKeys = {
            "d0": "density_residualX",
            "d1": "density_residualY",
            "d2": "density_residualZ",
        }

        val.rename(columns=newKeys, inplace=True)
        return val

    def get_thresh_data(self):
        return self.thresh_data

    def compare_dalton(self, basis):
        dalton_reader = DaltonRunner()
        ground_dalton, response_dalton = dalton_reader.get_frequency_result(
            self.mol, self.xc, self.operator, basis
        )

        ground_compare = pd.concat(
            [
                ground_dalton,
                pd.Series(self.ground_timing, index=["wall-time"]),
                pd.Series(self.ground_e),
            ]
        )

        self.dalton_data[basis] = response_dalton
        freq = response_dalton.iloc[:, 0]
        polar_df = self.polar_df.iloc[:, 3:].reset_index(drop=True)
        polar_df = pd.concat([freq, polar_df], axis=1)

        diff_df = pd.concat(
            [polar_df.iloc[:, 0], polar_df.iloc[:, 1:] - response_dalton.iloc[:, 1:]],
            axis=1,
        )

        return ground_compare, response_dalton, polar_df, diff_df

    def plot_polar_data(self, basis, ij_list):
        dalton_reader = DaltonRunner()
        if basis in self.dalton_data:
            response_dalton = self.dalton_data[basis]
        else:
            ground_dalton, response_dalton = dalton_reader.get_frequency_result(
                self.mol, self.xc, self.operator, basis
            )
            self.dalton_data[basis] = response_dalton

        dal_list = []
        mad_list = []

        for e in ij_list:
            dal_list.append("dal-" + e)
            mad_list.append("mad-" + e)

        dal_df = response_dalton[ij_list].reset_index(drop=True)
        dal_df.columns = dal_list

        freq = response_dalton.iloc[:, 0]

        mad_df = self.polar_df[ij_list].reset_index(drop=True)
        mad_df.columns = mad_list

        comp_df = pd.concat([freq, dal_df, mad_df], axis=1).set_index("frequencies")
        comp_df.plot(title=str(self.mol))

    def compare_polar_basis_list(self, ij_j_list, basis_list):
        dalton_reader = DaltonRunner()

        freq = pd.Series(self.polar_df.index.values)
        freq.name = "Frequency"

        compare_dict = [freq]

        for basis in basis_list:
            ground_dalton, response_dalton = dalton_reader.get_frequency_result(
                self.mol, self.xc, self.operator, basis
            )
            col = response_dalton[ij_j_list]
            col.name = basis
            compare_dict.append(col)
        polar_df = self.polar_df.iloc[:, 3:].reset_index(drop=True)
        mad_col = polar_df[ij_j_list].iloc[:]
        mad_col.name = "MRA"
        compare_dict.append(mad_col)
        polar_df = pd.concat(compare_dict, axis=1)
        polar_df.set_index("Frequency", inplace=True)

        polar_df.plot(title=self.mol + " Polarizability " + ij_j_list)

        # Epolar_df = pd.concat([freq, polar_df], axis=1)

        return polar_df

    def compare_diff_basis_list(self, ij_j_list, basis_list):
        dal_runner = DaltonRunner()

        freq = pd.Series(self.polar_df.index.values)
        freq.name = "Frequency"

        compare_dict = [freq]

        polar_df = self.polar_df.iloc[:, 3:].reset_index(drop=True)
        mad_col = polar_df[ij_j_list].iloc[:]
        mad_col.name = "MRA"

        for basis in basis_list:
            ground_dalton, response_dalton = dal_runner.get_frequency_result(
                self.mol, self.xc, self.operator, basis
            )
            col = response_dalton[ij_j_list]
            col = col - mad_col
            col.name = basis
            compare_dict.append(col)
        # compare_dict.append(mad_col)
        polar_df = pd.concat(compare_dict, axis=1)
        polar_df.set_index("Frequency", inplace=True)

        polar_df.plot(title=self.mol + " Polarizability " + ij_j_list)
        plotname = "diff_plots/" + self.mol + "_" + basis_list[0] + ".svg"

        plt.savefig(plotname)

        # Epolar_df = pd.concat([freq, polar_df], axis=1)

        return polar_df


class ExcitedData:
    def __init__(self, mol, xc):
        self.mol = mol
        self.xc = xc
        mad_reader = MadnessReader()

        (
            self.ground_params,
            self.ground_scf_data,
            self.ground_timing,
            self.calc_info
        ) = mad_reader.get_ground_scf_data(mol, xc)
        e_name_list = ["e_coulomb", "e_kinetic", "e_local", "e_nrep", "e_tot"]
        self.ground_e = {}
        for e_name in e_name_list:
            self.ground_e[e_name] = self.ground_scf_data[e_name][-1]
        (
            self.params,
            self.iterations,
            self.k,
            self.thresh_data,
            self.d_residuals,
            self.bsh_residuals,
            self.omega,
            self.full_omega,
            self.wall_time,
            self.cpu_time,
            self.converged,
            self.num_iter_proto,
            self.response_base,
            self.d_norms,
            self.bsh_norms
        ) = mad_reader.get_excited_result(mol, xc)
        self.num_states = self.params["states"]

    def get_thresh_data(self):
        return self.thresh_data

    def plot_density_residuals(self):
        self.d_residuals.plot(
            title=str(self.mol) + " Excited Density Residual plot: ", logy=True
        )

    def plot_bsh_residuals(self):
        self.bsh_residuals.plot(
            title=str(self.mol) + "Excited  BSH Residual plot: ", logy=True
        )

    def compare_dalton(self, basis):
        dalton_reader = DaltonRunner()
        ground_dalton, response_dalton = dalton_reader.get_excited_result(
            self.mol, self.xc, basis
        )

        ground_compare = pd.concat(
            [
                ground_dalton,
                pd.Series(self.ground_timing, index=["wall_time"]),
                pd.Series(self.ground_e),
            ]
        )
        omega_df = response_dalton.iloc[0: self.num_states]
        omega_df.loc[:, "mad-omega"] = self.omega
        omega_df.loc[:, "delta-omega"] = (
                omega_df.loc[:, "freq"] - omega_df.loc[:, "mad-omega"]
        )
        omega_df.loc[:, "d-residual"] = self.d_residuals.iloc[-1, :].reset_index(
            drop=True
        )
        omega_df.loc[:, "bshx-residual"] = self.bsh_residuals.iloc[
                                           -1, 0: self.num_states
                                           ].reset_index(drop=True)
        omega_df.loc[:, "bshy-residual"] = self.bsh_residuals.iloc[
                                           -1, self.num_states::
                                           ].reset_index(drop=True)

        return ground_compare, omega_df


# input response_info json and returns a dict of response paramters
# and a list of dicts of numpy arrays holding response data


# Plotting definitions

dalton_reader = DaltonRunner()


def create_polar_table(mol, xc, basis_list, xx):
    ground_dalton, response_dalton = dalton_reader.get_frequency_result(
        mol, "hf", "dipole", basis_list[0]
    )
    freq = response_dalton["frequencies"]
    g_data = {}
    xx_data = []
    for i in range(len(freq)):
        xx_data.append({})
    for basis in basis_list:
        ground_dalton, response_dalton = dalton_reader.get_frequency_result(
            mol, "hf", "dipole", basis
        )
        for i in range(len(freq)):
            xx_data[i][basis] = response_dalton[xx][i]
        g_data[basis] = ground_dalton["totalEnergy"]
    g_df = pd.Series(g_data)
    g_df.name = "Total HF Energy"
    names = []
    for f in freq:
        raw_f = r"{}".format(str(f))
        # names.append(r'$$\alpha_{xx}('+raw_f+r')$$')
        names.append("a(" + "{:.3f}".format(f) + ")")
    r_dfs = []
    for i in range(len(freq)):
        r_dfs.append(pd.Series(xx_data[i]))
        r_dfs[i].name = names[i]
    dalton_df = pd.concat([g_df] + r_dfs, axis=1)

    moldata = FrequencyData(mol, "hf", "dipole")
    moldata.polar_df[xx]
    mad_data_e = {}
    mad_data_r = {}
    mad_data_e["Total HF Energy"] = moldata.ground_e["e_tot"]

    for i in range(len(names)):
        mad_data_r[names[i]] = moldata.polar_df[xx][i]

    mad_data_e = pd.Series(mad_data_e)
    mad_data_r = pd.Series(mad_data_r)

    mad_data = pd.concat([mad_data_e, mad_data_r], axis=0)
    mad_data.name = "MRA"
    mad_data.key = ["MRA"]
    data = pd.concat([dalton_df.T, mad_data.T], axis=1)
    return data.T


def create_data(mol, basis_list):
    res_dict = {
        "xx": "density_residualX",
        "yy": "density_residualY",
        "zz": "density_residualZ",
    }

    moldata = FrequencyData(mol, "hf", "dipole")
    xx = ["xx", "yy", "zz"]
    data = []
    for x in xx:
        data.append(create_polar_table(mol, "hf", basis_list, x))
    average = (data[0] + data[1] + data[2]) / 3

    diff_data = average - average.loc["MRA"]
    diff_data = diff_data.drop(index="MRA")

    polar_diff = diff_data.drop("Total HF Energy", axis=1)

    residuals = moldata.final_density_residuals()

    cleanX = residuals["density_residualX"].reset_index().drop("index", axis=1)
    cleanY = residuals["density_residualY"].reset_index().drop("index", axis=1)
    cleanZ = residuals["density_residualZ"].reset_index().drop("index", axis=1)

    cleanX.rename(columns={"density_residualX": "MRA density residual"}, inplace=True)
    cleanY.rename(columns={"density_residualY": "MRA density residual"}, inplace=True)
    cleanZ.rename(columns={"density_residualZ": "MRA density residual"}, inplace=True)
    clean = (cleanX + cleanY + cleanZ) / 3
    clean.index = polar_diff.T.index

    average = pd.concat([average, clean.T])

    average.name = "Average Polarizability"

    energy_diff = diff_data["Total HF Energy"]
    return average, diff_data, energy_diff, polar_diff


def create_polar_diff_plot(mol, basis_list):
    title = 'Average Polarizability Basis Set Convergence: ' + mol
    yl = r" $\Delta\alpha_{avg}$" + r" (MRA - BASIS)"

    data, diff_data, energy_diff, polar_diff = create_data(mol, basis_list)
    num_freq = len(list(polar_diff.keys()))
    polar_diff.iloc[:, :].plot(marker="v", linestyle="solid", markersize=12, linewidth=2, colormap='plasma')

    legend = []
    for i in range(num_freq):
        legend.append(r'$\alpha(\omega_{})$'.format(i))

    plt.axhline(linewidth=2, ls="--", color="k")
    plt.legend(legend, fontsize=12)
    plt.xticks(fontsize=14, rotation=20)
    plt.title(title, fontsize=20)
    plt.ylabel(yl, fontsize=14)
    btitle = basis_list[0].replace('D', 'X')
    save = mol + "-" + btitle
    if not os.path.exists("figures"):
        os.mkdir("figures")
    if not os.path.exists("tables"):
        os.mkdir("tables")
    save = "figures/" + save + ".svg"
    plt.savefig(save)

    latex_save = mol + "-" + btitle
    latex_save = "tables/" + latex_save + ".tex"
    data = data.round(decimals=4)
    data.style.to_latex(latex_save)

    return data


def get_excited_mol_series(mol, basis):
    d = ExcitedData(mol, "hf")
    g, e = d.compare_dalton(basis)

    mad_series = e["mad-omega"]
    dal_series = e["freq"]
    md = []
    dd = []
    for m in range(mad_series.size):
        md.append("mra-{v}".format(v=m))
        dd.append("{b}-{v}".format(b=basis, v=m))
    mad_series.index = md
    dal_series.index = dd
    conv_series = pd.Series(d.converged)
    conv_series.index = ["Converged"]

    molseries = pd.concat([conv_series, mad_series, dal_series])
    return molseries


def create_excited_comparison_data(basis, excluded):
    data = {}
    for g in glob.glob("molecules/*.mol"):
        m = g.split("/")
        mol = m[1].split(".")[0]
        if mol not in excluded:
            data[mol] = get_excited_mol_series(mol, basis)
    excited_data = pd.DataFrame(data).round(4)
    return excited_data


def plot_norm_and_residual_i(d, num_i, ax):
    norm_keys = list(d.bsh_norms[list(d.bsh_norms.keys())[0]].keys())
    abs_keys = norm_keys[0:3]
    rel_keys = norm_keys[3::]
    chi_norms_key = norm_keys[3::]
    f_key = list(d.bsh_norms.keys())[num_i]
    dconv = d.params["dconv"]

    chi_norms_i = d.bsh_norms[f_key][chi_norms_key]
    bsh_residuals_i = d.bsh_residuals[f_key][abs_keys]
    rel_residuals_i = d.bsh_residuals[f_key][rel_keys]

    chi_norms_i.plot(ax=ax[0], logy=False, legend=False, color='rgb', title='Chi Norms', marker='*', grid=True)
    bsh_residuals_i.plot(ax=ax[1], logy=True, legend=False, color='rgb', title='Absolute Residuals', marker='*',
                         grid=True)
    rel_residuals_i.plot(ax=ax[2], logy=True, legend=False, color='rgb', title='Relative Residuals', marker='*',
                         grid=True)

    iters = d.num_iter_proto[f_key]
    # print(iters)
    for j in range(1, iters.__len__()):
        iters[j] = iters[j] + iters[j - 1]
    for j in range(0, iters.__len__()):
        if iters[j] != 1:
            iters[j] -= 1
    # print(iters)

    for num_iter_pf in iters:
        for i in range(3):
            ax[i].axvline(x=num_iter_pf, ymin=0, ymax=1, c="black", linestyle="dashed")
    for i in range(3):
        if i != 0:
            ax[i].axhline(y=dconv, xmin=0, xmax=d.num_iter_proto[f_key][-1], c="black", linestyle="dashed", )
        ax[i].grid(which="both")
        ax[i].minorticks_on()
        ax[i].tick_params(which="both", top="on", left="on", right="on", bottom="on", )


def freq_norm_and_residual(mol, xc, op, save):
    d = FrequencyData(mol, xc, op)

    xkeys = []
    ykeys = []

    for i in range(d.num_states):
        xkeys.append("x" + str(i))
        ykeys.append("y" + str(i))

    mad_read = MadnessReader()
    freq = mad_read.freq_json[mol][xc][op]

    num_ran = len(d.converged)
    num_pass = sum(d.converged)

    num_freqs = len(freq)
    frequencies = list(d.num_iter_proto.keys())

    # print('num frequencies: ',num_freqs)
    # print('num pass: ',num_pass)
    # print('num ran: ',num_ran)

    sns.set_theme(style="darkgrid")
    sns.set_context("talk", font_scale=1.5, rc={"lines.linewidth": 2.5})
    for i in range(num_ran):

        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25, 9), constrained_layout=True)
        title = 'Polarizability Convergence: ' + mol + r'  $\omega({}/{})$'.format(i, num_freqs - 1)
        fig.suptitle(title)
        plot_norm_and_residual_i(d, i, ax)
        plotname = 'freq_{}'.format(i) + ".svg"
        if save:
            if not os.path.exists("convergence"):
                os.mkdir("convergence")
            if not os.path.exists('convergence/' + mol):
                os.mkdir("convergence/" + mol)
            if not os.path.exists('convergence/' + mol + '/' + xc):
                os.mkdir("convergence/" + mol + '/' + xc)
            plt.savefig("convergence/" + mol + '/' + xc + '/' + plotname)
    print(mol + "\n converged: ", d.converged)


def display_convergence_plots(mol, xc, rtype, save):
    d = None
    if rtype == "excited":
        d = ExcitedData(mol, xc)
    elif rtype == "dipole":
        d = FrequencyData(mol, xc, "dipole")

    xkeys = []
    ykeys = []
    for i in range(d.num_states):
        xkeys.append("x" + str(i))
        ykeys.append("y" + str(i))

    dconv = d.params["dconv"]
    if rtype == "excited":
        d.full_omega.loc[:, 0:].plot(title="Excited-State Frequency")
        omega_min = d.omega.iloc[0]
        omega_max = d.omega.iloc[-1]
        plt.vlines(
            d.num_iter_proto, omega_min, omega_max, colors="black", linestyle="dashed"
        )

        d.d_residuals.plot(logy=True, title="Density Residuals Excited")
        plt.vlines(d.num_iter_proto, 0, 1, colors="black", linestyle="dashed")
        plt.hlines(dconv, 0, d.num_iter_proto[-1], colors="black", linestyle="dashed")
        d.bsh_residuals.loc[:, xkeys].plot(logy=True, title="BSH X Residuals Excited")
        plt.vlines(d.num_iter_proto, 0, 1, colors="black", linestyle="dashed")
        plt.hlines(dconv, 0, d.num_iter_proto[-1], colors="black", linestyle="dashed")
        d.bsh_residuals.loc[:, ykeys].plot(logy=True, title="BSH Y Residuals Excited")
        plt.vlines(d.num_iter_proto, 0, 1, colors="black", linestyle="dashed")
        plt.hlines(dconv, 0, d.num_iter_proto[-1], colors="black", linestyle="dashed")

    elif rtype == "dipole":

        mad_read = MadnessReader()
        freq = mad_read.freq_json[mol][xc][rtype]
        num_freqs = len(freq)
        num_converged = len(d.num_iter_proto.keys())
        frequencies = list(d.num_iter_proto.keys())
        # print("number of frequencies: ", num_freqs)
        # print("number of frequencies converged: ", num_converged)
        f_labels = []
        for i in range(num_converged):
            f_labels.append(
                r"$[\omega=\omega_{{max}}({top}/{bot})]$".format(
                    top=i, bot=num_freqs - 1
                )
            )
        fig_len = 5
        fig = plt.figure(
            constrained_layout=True, figsize=(3 * fig_len, fig_len * num_converged)
        )
        fig.suptitle(mol + " Residuals", fontsize=15)
        rgb = ["r", "g", "b"]
        # create 3x1 subfigs
        subfigs = fig.subfigures(nrows=len(frequencies), ncols=1)
        freq_i = 0
        if len(frequencies) == 1:
            if freq_i == 0:
                f = frequencies[freq_i]
                axs = subfigs.subplots(nrows=1, ncols=3)
                d.d_residuals[f].plot(
                    logy=True,
                    ax=axs[0],
                    legend=False,
                    color=rgb,
                    title="Density",
                    marker="*",
                    grid=True,
                )
                d.bsh_residuals[f].loc[:, xkeys].plot(
                    logy=True,
                    ax=axs[1],
                    legend=False,
                    color=rgb,
                    title="BSH X",
                    marker="*",
                    grid=True,
                )
                d.bsh_residuals[f].loc[:, ykeys].plot(
                    logy=True,
                    ax=axs[2],
                    legend=False,
                    color=rgb,
                    title="BSH Y",
                    marker="*",
                    grid=True,
                )
            iters = d.num_iter_proto[f]
            # print(iters)
            for j in range(1, iters.__len__()):
                iters[j] = iters[j] + iters[j - 1]
            for j in range(0, iters.__len__()):
                if iters[j] != 1:
                    iters[j] -= 1
            # print(iters)

            for num_iter_pf in iters:
                axs[0].axvline(
                    x=num_iter_pf, ymin=0, ymax=1, c="black", linestyle="dashed"
                )
                axs[1].axvline(
                    x=num_iter_pf, ymin=0, ymax=1, c="black", linestyle="dashed"
                )
                axs[2].axvline(
                    x=num_iter_pf, ymin=0, ymax=1, c="black", linestyle="dashed"
                )
            for i in range(3):
                axs[i].axhline(
                    y=dconv,
                    xmin=0,
                    xmax=d.num_iter_proto[f][-1],
                    c="black",
                    linestyle="dashed",
                )
                axs[i].grid(which="both")
                axs[i].minorticks_on()
                axs[i].tick_params(
                    which="both",  # Options for both major and minor ticks
                    top="on",  # turn off top ticks
                    left="on",  # turn off left ticks
                    right="on",  # turn off right ticks
                    bottom="on",
                )
            labels = [
                r"$\Delta\gamma^{(x)}$",
                r"$\Delta\gamma^{(y)}$",
                r"$\Delta\gamma_{(z)}$",
            ]
            fig.legend(labels, loc="upper left")
        else:
            freq_i = 0
            for row, subfig in enumerate(subfigs):
                # print(freq_i)
                f = frequencies[freq_i]
                rowtitle = f_labels[freq_i] + " Converged: " + str(d.converged[f])
                subfig.suptitle(
                    rowtitle,
                )
                # create 1x3 subplots per subfig
                axs = subfig.subplots(nrows=1, ncols=3)
                if freq_i == 0:
                    d.d_residuals[f].plot(
                        logy=True,
                        ax=axs[0],
                        legend=False,
                        color=rgb,
                        title="Density",
                        marker="*",
                        grid=True,
                    )
                    d.bsh_residuals[f].loc[:, xkeys].plot(
                        logy=True,
                        ax=axs[1],
                        legend=False,
                        color=rgb,
                        title="BSH X",
                        marker="*",
                        grid=True,
                    )
                    d.bsh_residuals[f].loc[:, ykeys].plot(
                        logy=True,
                        ax=axs[2],
                        legend=False,
                        color=rgb,
                        title="BSH Y",
                        marker="*",
                        grid=True,
                    )
                else:
                    d.d_residuals[f].plot(
                        logy=True,
                        ax=axs[0],
                        legend=False,
                        color=rgb,
                        marker="*",
                        grid=True,
                    )
                    d.bsh_residuals[f].loc[:, xkeys].plot(
                        logy=True,
                        ax=axs[1],
                        legend=False,
                        color=rgb,
                        marker="*",
                        grid=True,
                    )
                    d.bsh_residuals[f].loc[:, ykeys].plot(
                        logy=True,
                        ax=axs[2],
                        legend=False,
                        color=rgb,
                        marker="*",
                        grid=True,
                    )
                iters = d.num_iter_proto[f]
                # print(iters)
                for j in range(1, iters.__len__()):
                    iters[j] = iters[j] + iters[j - 1]
                for j in range(0, iters.__len__()):
                    if iters[j] != 1:
                        iters[j] -= 1
                # print(iters)

                for num_iter_pf in iters:
                    axs[0].axvline(
                        x=num_iter_pf, ymin=0, ymax=1, c="black", linestyle="dashed"
                    )
                    axs[1].axvline(
                        x=num_iter_pf, ymin=0, ymax=1, c="black", linestyle="dashed"
                    )
                    axs[2].axvline(
                        x=num_iter_pf, ymin=0, ymax=1, c="black", linestyle="dashed"
                    )

                for i in range(3):
                    axs[i].axhline(
                        y=dconv,
                        xmin=0,
                        xmax=d.num_iter_proto[f][-1],
                        c="black",
                        linestyle="dashed",
                    )
                    axs[i].grid(which="both")
                    axs[i].minorticks_on()
                    axs[i].tick_params(
                        which="both",  # Options for both major and minor ticks
                        top="on",  # turn off top ticks
                        left="on",  # turn off left ticks
                        right="on",  # turn off right ticks
                        bottom="on",
                    )

                freq_i += 1
                labels = [
                    r"$\Delta\gamma^{(x)}$",
                    r"$\Delta\gamma^{(y)}$",
                    r"$\Delta\gamma_{(z)}$",
                ]
                fig.legend(labels, loc="upper left")

    plotname = mol + "_" + xc + ".svg"
    if save:
        if not os.path.exists("convergence"):
            os.mkdir("convergence")
        plt.savefig("convergence/" + plotname)
    print(mol + "\n converged: ", d.converged)
    return d


def create_polar_mol_series(mol, basis):
    data = FrequencyData(mol, "hf", "dipole")

    converged = data.converged
    freq = pd.Series(converged.keys())

    mra_keys = ["HF Energy"]
    diff_keys = [basis]
    conv_keys = []
    for f in range(freq.size):
        mra_keys.append("avg_{d}".format(d=f))
        diff_keys.append("diff_{d}".format(d=f))
        conv_keys.append("converged_{d}".format(d=f))

    xx = ["xx", "yy", "zz"]
    data = []
    for x in xx:
        data.append(create_polar_table(mol, "hf", [basis], x))
    average = (data[0] + data[1] + data[2]) / 3
    mra = average.loc["MRA"]
    basis_value = average.loc[basis]
    diff = mra - basis_value
    avg_diff = diff.mean()
    avg_diff = pd.Series(avg_diff)
    avg_diff.index = ["average diff"]
    mra.index = mra_keys
    diff.index = diff_keys
    converged.index = conv_keys
    new = pd.concat([freq, mra, pd.Series(avg_diff), converged], axis=0)

    return new


def polar_overview(basis, excluded):
    data = {}
    for g in glob.glob("molecules/*.mol"):
        m = g.split("/")
        mol = m[1].split(".")[0]
        print(mol)
        if mol not in excluded:
            data[mol] = create_polar_mol_series(mol, basis)

    return pd.DataFrame(data)


def create_basis_mol_data(basis_list, mol_list, data_dict):
    b_data = {}
    for b in basis_list:
        diff_dict = {}
        for mol in mol_list:
            diff_m = data_dict[mol] - data_dict[mol].loc["MRA"]
            diff_dict[mol] = diff_m.loc[b]
            diff_dict[mol].index = [
                "Total HF Energy",
                r"$\alpha(\omega_0)$",
                r"$\alpha(\omega_1)$",
                r"$\alpha(\omega_2)$",
                r"$\alpha(\omega_3)$",
                r"$\alpha(\omega_4)$",
            ]
        pdm = pd.DataFrame(diff_dict)
        pdm.name = b
        b_data[b] = pdm.T
    return b_data


def mean_and_std(basis_list, mol_list, data_dict):
    b_data = create_basis_mol_data(basis_list, mol_list, data_dict)
    mean_d = {}
    std_d = {}
    for b in basis_list:
        mean_d[b] = b_data[b].mean()
        std_d[b] = b_data[b].std()
    p_mean = pd.DataFrame(mean_d)
    p_std = pd.DataFrame(std_d)
    return p_mean, p_std


class MadRunner:
    def run_response(self, mol, xc, operator, prec):

        if operator == "dipole":
            mad_command = "mad-freq"
        elif operator == "excited-state":
            mad_command = "mad-excited"
        else:
            print("not implemented yet")
            return 1
        madnessCommand = " ".join([mad_command, mol, xc, prec])

        process = subprocess.Popen(madnessCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        return output, error


def run_madness_ground(self, mol, xc):
    mad_command = "database-moldft"
    madnessCommand = " ".join([mad_command, mol, xc])
    process = subprocess.Popen(madnessCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output, error
