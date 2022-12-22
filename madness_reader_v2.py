import subprocess

import seaborn as sns

import matplotlib.pyplot as plt
from setuptools import glob

from dalton import Dalton

import numpy as np

from madnessToDaltony import *


class MadnessReader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        with open(self.data_dir + "/molecules/frequency.json") as json_file:
            self.freq_json = json.loads(json_file.read())

    def __tensor_to_numpy(self, j):
        array = np.empty(j["size"])
        array[:] = j["vals"]
        return np.reshape(array, tuple(j["dims"]))

    def __read_protocol_excited_state_data(self, protocol_data: json, num_states, num_orbitals):
        num_protocols = protocol_data.__len__()
        polar_dfs = []
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
            omega_data = np.empty((num_iters, num_states))
            i = 0
            for iter in proto["property_data"]:
                # diagonalize the polarizability
                alpha = self.__tensor_to_numpy(iter["omega"]).flatten()
                # alpha=.5*(alpha+alpha.transpose())
                # w, v = LA.eig(alpha)
                # print("alpha : ",alpha)
                omega_data[i, :] = alpha
                i += 1
                iters.append(iter_p)
                iter_p += 1
            kproto_df = pd.DataFrame(kproto_array, columns=["k"])
            proto_df = pd.DataFrame(proto_array, columns=["thresh"])
            polar_df = pd.DataFrame(
                omega_data
            )
            polar_df = pd.concat([kproto_df, proto_df, polar_df], axis=1)
            polar_dfs.append(polar_df)
        iters_df = pd.DataFrame(iters, columns=["iterations"])
        final_polar = pd.concat(polar_dfs, ignore_index=True)
        final_polar = pd.concat([iters_df, final_polar], axis=1)

        return final_polar

    def __read_protocol_polarizability_data(self, protocol_data: json, num_states, num_orbitals):
        num_protocols = protocol_data.__len__()
        polar_dfs = []
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
            i = 0
            for iter in proto["property_data"]:
                # diagonalize the polarizability
                alpha = self.__tensor_to_numpy(iter["polar"]).flatten()
                # alpha=.5*(alpha+alpha.transpose())
                # w, v = LA.eig(alpha)
                # print("alpha : ",alpha)
                polar_data[i, :] = alpha
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
            polar_dfs.append(polar_df)

        iters_df = pd.DataFrame(iters, columns=["iterations"])
        final_polar = pd.concat(polar_dfs, ignore_index=True)
        final_polar = pd.concat([iters_df, final_polar], axis=1)

        return final_polar

    def __read_response_protocol_data(self, protocol_data: json, num_states, num_orbitals):

        # reading function data
        # x norms X1 ... Xm
        # x abs error aX1 ... aXM
        # x rel error rX1 ... rXM

        # xij norms x11....xmn y11 ymn size=s*m*n
        # xij abs error x11....xmn y11 ymn size=s*m*n

        # rho norms D1 ... DM
        # rho abs error aD1 ... aDM

        num_protocols = protocol_data.__len__()
        # create the data keys
        x_keys = []
        ax_keys = []
        rx_keys = []

        d_keys = []
        ad_keys = []

        for i in range(num_states):
            x_keys.append("X" + str(i))
            ax_keys.append("abs_X" + str(i))
            rx_keys.append("rel_X" + str(i))

            d_keys.append("D" + str(i))
            ad_keys.append("abs_D" + str(i))

        xij_keys = []
        axij_keys = []

        for i in range(num_states):
            for j in range(num_orbitals):
                xij_keys.append('x' + str(i) + str(j))
                axij_keys.append('abs_x' + str(i) + str(j))

        for i in range(num_states):
            for j in range(num_orbitals):
                xij_keys.append('y' + str(i) + str(j))
                axij_keys.append('abs_y' + str(i) + str(j))
        # print(xij_keys)
        # print(axij_keys)

        x_norm_dfs = []
        x_abs_error_dfs = []
        x_rel_error_dfs = []

        d_norm_dfs = []
        d_abs_error_dfs = []

        xij_norm_dfs = []
        xij_error_dfs = []

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

            x_norms = np.empty((num_iters, num_states))
            x_abs_error = np.empty((num_iters, num_states))

            d_norms = np.empty((num_iters, num_states))
            d_abs_error = np.empty((num_iters, num_states))

            xij_norms = np.empty((num_iters, num_states * 2 * num_orbitals))
            xij_abs_error = np.empty((num_iters, num_states * 2 * num_orbitals))

            i = 0
            for iter in proto["iter_data"]:
                x_norms[i, :] = self.__tensor_to_numpy(iter["x_norms"]).flatten()
                x_abs_error[i, :] = self.__tensor_to_numpy(iter["x_abs_error"]).flatten()

                d_norms[i, :] = self.__tensor_to_numpy(iter["rho_norms"]).flatten()
                d_abs_error[i, :] = self.__tensor_to_numpy(iter["rho_abs_error"]).flatten()

                xij_norms[i, :] = self.__tensor_to_numpy(iter["xij_norms"]).flatten()
                xij_abs_error[i, :] = self.__tensor_to_numpy(iter["xij_abs_error"]).flatten()

                i += 1
                iters.append(iter_p)
                iter_p += 1

            # num_iters_per_protocol[-1] -= 1

            x_norm_dfs.append(pd.DataFrame(x_norms, columns=x_keys))
            x_abs_error_dfs.append(pd.DataFrame(x_abs_error, columns=ax_keys))

            d_norm_dfs.append(pd.DataFrame(d_norms, columns=d_keys))
            d_abs_error_dfs.append(pd.DataFrame(d_abs_error, columns=ad_keys))

            xij_norm_dfs.append(pd.DataFrame(xij_norms, columns=xij_keys))
            xij_error_dfs.append(pd.DataFrame(xij_abs_error, columns=axij_keys))
        for j in range(1, num_iters_per_protocol.__len__()):
            num_iters_per_protocol[j] = (num_iters_per_protocol[j] + num_iters_per_protocol[j - 1])

        x1 = pd.concat(x_norm_dfs)
        xa = pd.concat(x_abs_error_dfs)

        d1 = pd.concat(d_norm_dfs)
        da = pd.concat(d_abs_error_dfs)

        f1 = pd.concat(xij_norm_dfs)
        fa = pd.concat(xij_error_dfs)

        iters_df = pd.Series(iters)
        iters_df.name = "iterations"
        full = pd.concat([x1, xa, d1, da, f1, fa], axis=1)
        full = pd.concat([iters_df, full.reset_index(drop=True)], axis=1)
        full.index += 1

        # final_bsh_norms = pd.concat([iters_df,full_df], ignore_index=True)

        return num_iters_per_protocol, full

    def __open_ground_json(self, mol, xc):

        moldir = self.data_dir + "/" + xc + "/" + mol
        jsonf = "moldft.calc_info.json"

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
        moldir = self.data_dir + "/" + xc + "/" + mol
        dfile = "excited-" + str(num_states)
        jsonf = "response_base.json"

        path = "/".join([moldir, dfile, jsonf])

        with open(path) as json_file:
            response_j = json.loads(json_file.read())

        return response_j

    def get_excited_data(self, mol, xc):
        num_states = self.freq_json[mol][xc]["excited-state"]
        rbasej = self.__open_excited_rbj(mol, xc, num_states)
        num_orbitals = rbasej["parameters"]["num_orbitals"]
        converged = rbasej["converged"]
        num_iters_per_protocol, function_data = self.__read_response_protocol_data(
            rbasej["protocol_data"], num_states, num_orbitals
        )
        omega = self.__read_protocol_excited_state_data(
            rbasej["protocol_data"], num_states, num_orbitals
        )

        params = rbasej["parameters"]
        wall_time = rbasej["wall_time"]

        return params, wall_time, converged, num_iters_per_protocol, rbasej, function_data, omega

    def __open_frequency_rbj(self, mol, xc, operator, freq):

        sfreq = "%f" % freq
        # first number before decimal
        f1 = sfreq.split(".")[0]
        # second number after decimal
        f2 = sfreq.split(".")[1]

        moldir = self.data_dir + "/" + xc + "/" + mol
        dfile = operator + "_" + xc + "_" + f1 + "-" + f2
        jsonf = "response_base.json"

        path = "/".join([moldir, dfile, jsonf])
        # print(path)

        with open(path) as json_file:
            response_j = json.loads(json_file.read())

        return response_j

    def __get_polar_data(self, rbase_j):
        params = rbase_j["parameters"]
        num_orbitals = params["num_orbitals"]
        num_states = params["states"]

        (
            num_iters_per_protocol,
            full_function_data,
        ) = self.__read_response_protocol_data(rbase_j["protocol_data"], num_states, num_orbitals)
        polarizability_data = self.__read_protocol_polarizability_data(rbase_j['protocol_data'], num_states,
                                                                       num_orbitals)
        polarizability_data.index += 1
        return params, num_iters_per_protocol, full_function_data, polarizability_data

    # TODO get the ground data
    def get_polar_result(self, mol, xc, operator):
        freq = self.freq_json[mol][xc][operator]
        polar_data = {}
        last_polar_freq = {}
        function_data = {}
        time_data = {}
        converged = {}
        num_iter_proto = {}
        full_params = {}
        full_response_base = {}
        for f in freq:
            try:
                rbasej = self.__open_frequency_rbj(mol, xc, operator, f)
                full_response_base[str(f)] = rbasej
                converged_f = rbasej["converged"]
                # print(converged_f)
                params, num_iters_per_protocol, full_function_data, polarizability_data = self.__get_polar_data(
                    rbasej)
                full_params[str(f)] = params
                polar_data[str(f)] = polarizability_data
                # print(polarizability_data)
                last_polar_freq[str(f)] = polarizability_data.iloc[-1, :]

                num_iter_proto[str(f)] = num_iters_per_protocol
                function_data[str(f)] = pd.DataFrame(full_function_data)
                # fdata[str(f)] = full_function_data.iloc[-1, :]
                converged[str(f)] = converged_f

                time_data[str(f)] = rbasej["time_data"]

            except FileNotFoundError as not_found:
                print(f, " not found:", not_found)
                pass

        return (
            full_params,
            time_data,
            pd.Series(converged),
            num_iter_proto,
            full_response_base,
            function_data,
            polar_data,
            pd.DataFrame(last_polar_freq).T
        )


def get_function_keys(num_states, num_orbitals):
    x_keys = []
    ax_keys = []

    d_keys = []
    ad_keys = []

    for i in range(num_states):
        x_keys.append("X" + str(i))
        ax_keys.append("abs_X" + str(i))

        d_keys.append("D" + str(i))
        ad_keys.append("abs_D" + str(i))

    xij_keys = []
    axij_keys = []

    for i in range(num_states):
        for j in range(num_orbitals):
            xij_keys.append('x' + str(i) + str(j))
            axij_keys.append('abs_x' + str(i) + str(j))

    for i in range(num_states):
        for j in range(num_orbitals):
            xij_keys.append('y' + str(i) + str(j))
            axij_keys.append('abs_y' + str(i) + str(j))

    return {"x_norms": x_keys, "x_abs_error": ax_keys, "d_norms": d_keys,
            "d_abs_error": ad_keys, "xij_norms": xij_keys, "xij_abs_error": axij_keys}


class ResponseCalc:
    def __init__(self, mol, xc, operator, data_dir):
        self.data_dir = data_dir
        self.ground_info = None
        self.mol = mol
        self.xc = xc
        self.operator = operator
        mad_reader = MadnessReader(self.data_dir)
        (
            self.ground_params,
            self.ground_scf_data,
            self.ground_timing,
            self.ground_info,
        ) = mad_reader.get_ground_scf_data(mol, xc)
        e_name_list = ["e_coulomb", "e_kinetic", "e_local", "e_nrep", "e_tot"]
        self.ground_e = {}
        for e_name in e_name_list:
            self.ground_e[e_name] = self.ground_scf_data[e_name][-1]
        (
            self.params,
            self.time_data,
            self.converged,
            self.num_iter_proto,
            self.response_base,
            self.function_data,
            self.full_polar_data,
            self.polar_data,
        ) = mad_reader.get_polar_result(mol, xc, operator)
        self.num_states = self.params['0.0']["states"]
        self.num_orbitals = self.params['0.0']["num_orbitals"]
        self.function_keys = self.get_function_keys()
        self.data = {"ground": self.__get_ground_data()}
        fdata, pdata = self.__get_response_data()
        self.data["response"] = {}
        self.data["response"]["function"] = fdata
        self.data["response"]["polarizability"] = pdata

    def get_function_keys(self):
        x_keys = []
        ax_keys = []
        rx_keys = []

        d_keys = []
        ad_keys = []

        for i in range(self.num_states):
            x_keys.append("X" + str(i))
            ax_keys.append("abs_X" + str(i))

            d_keys.append("D" + str(i))
            ad_keys.append("abs_D" + str(i))

        xij_keys = []
        axij_keys = []

        for i in range(self.num_states):
            for j in range(self.num_orbitals):
                xij_keys.append('x' + str(i) + str(j))
                axij_keys.append('abs_x' + str(i) + str(j))

        for i in range(self.num_states):
            for j in range(self.num_orbitals):
                xij_keys.append('y' + str(i) + str(j))
                axij_keys.append('abs_y' + str(i) + str(j))

        return {"x_norms": x_keys, "x_abs_error": ax_keys, "d_norms": d_keys,
                "d_abs_error": ad_keys, "xij_norms": xij_keys, "xij_abs_error": axij_keys}

    def __get_ground_precision(self):
        gprec = self.ground_info["precision"]
        ground_data = {}
        for key, val in gprec.items():
            if key == "dconv":
                ground_data["g-dconv"] = val
            elif key == "thresh":
                ground_data["g-thresh"] = val
            elif key == "k":
                ground_data["g-k"] = val
            elif key == "eprec":
                ground_data["g-eprec"] = val
        return pd.Series(ground_data)

    def __get_response_precision(self, omega):
        p_dict = self.response_base[omega]
        r_prec = {}
        for key, val in p_dict.items():
            if key == "dconv":
                r_prec["r-dconv"] = val
            elif key == "thresh":
                r_prec["r-thresh"] = val
            elif key == "k":
                r_prec["r-k"] = val
        return pd.Series(r_prec, dtype=float)

    def __get_response_data(self):
        polar_keys = ['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz']
        ff = []
        pp = []
        g_precision = self.__get_ground_precision()

        for om, data in self.response_base.items():
            if data["converged"]:
                fdo = self.function_data[om].iloc[-1, 1:]
                pdo = self.polar_data.loc[om, polar_keys]
                fd = {}
                fd['frequency'] = om
                freq = pd.Series(fd)
                r_prec = self.__get_response_precision(om)
                ff.append(pd.concat([freq, g_precision, r_prec, fdo]))
                pp.append(pd.concat([freq, g_precision, r_prec, pdo]))
        fdata = pd.DataFrame(ff)
        pdata = pd.DataFrame(pp)
        return fdata, pdata

    def __get_ground_data(self):
        ground_data = self.__get_ground_precision()
        ground_data.update(self.ground_e)
        e_data = pd.Series(ground_data)
        return e_data

    def __plot_norm_and_residual_freq(self, num_i, ax):
        fkeys = get_function_keys(self.num_states, self.num_orbitals)
        abs_keys = fkeys["x_abs_error"]
        chi_norms_keys = fkeys["x_norms"]

        f_key = list(self.function_data.keys())[num_i]
        dconv = self.params[f_key]["dconv"]

        chi_norms_i = self.function_data[f_key][chi_norms_keys]
        bsh_residuals_i = self.function_data[f_key][abs_keys]

        chi_norms_i.plot(ax=ax[0], logy=False, legend=True, colormap='Accent', title='Chi Norms', marker='*',
                         markersize=12, grid=True)
        bsh_residuals_i.plot(ax=ax[1], logy=True, legend=True, colormap='Accent', title='Absolute Residuals',
                             marker='*', markersize=12,
                             grid=True)

        iters = self.num_iter_proto[f_key]

        for pc in iters:
            for i in range(2):
                ax[i].axvline(x=pc, ymin=0, ymax=1, c="black", linestyle="dashed")
        for i in range(2):
            if i != 0:
                ax[i].axhline(y=dconv, xmin=0, xmax=iters[-1], c="black", linestyle="dashed", )
            ax[i].grid(which="both")
            ax[i].minorticks_on()
            ax[i].tick_params(which="both", top="on", left="on", right="on", bottom="on", )

    def plot_freq_norm_and_residual(self, save, only_static=False):
        xkeys = []
        ykeys = []
        for i in range(self.num_states):
            xkeys.append("x" + str(i))
            ykeys.append("y" + str(i))
        freq = list(self.response_base.keys())
        num_ran = len(self.converged)
        num_freqs = len(freq)
        sns.set_theme(style="darkgrid")
        sns.set_context("talk", font_scale=1.5, rc={"lines.linewidth": 2.5})
        num_plot = num_ran
        if only_static:
            num_plot = 1
        for i in range(num_plot):
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(25, 9), constrained_layout=True)
            title = 'Polarizability Convergence: ' + self.mol + r'  $\omega({}/{})$'.format(i, num_freqs - 1)
            fig.suptitle(title)
            self.__plot_norm_and_residual_freq(i, ax)
            plotname = 'freq_{}'.format(i) + ".svg"
            if save:
                if not os.path.exists("convergence"):
                    os.mkdir("convergence")
                if not os.path.exists('convergence/' + self.mol):
                    os.mkdir("convergence/" + self.mol)
                if not os.path.exists('convergence/' + self.mol + '/' + self.xc):
                    os.mkdir("convergence/" + self.mol + '/' + self.xc)
                plt.savefig("convergence/" + self.mol + '/' + self.xc + '/' + plotname)
        print(self.mol + "\n converged: ", self.converged)


class ExcitedData:
    def __init__(self, mol, xc):
        self.mol = mol
        self.xc = xc
        mad_reader = MadnessReader()
        (
            self.ground_params,
            self.ground_scf_data,
            self.ground_timing,
            self.ground_info
        ) = mad_reader.get_ground_scf_data(mol, xc)
        e_name_list = ["e_coulomb", "e_kinetic", "e_local", "e_nrep", "e_tot"]
        self.ground_e = {}
        for e_name in e_name_list:
            self.ground_e[e_name] = self.ground_scf_data[e_name][-1]
        (
            self.params,
            self.wall_time,
            self.converged,
            self.num_iter_proto,
            self.response_base,
            self.function_data,
            self.omega
        ) = mad_reader.get_excited_data(mol, xc)
        self.num_states = self.params["states"]
        self.num_orbitals = self.params["num_orbitals"]

    def compare_dalton(self, basis, base_dir):
        dalton_reader = Dalton(base_dir)
        ground_dalton, response_dalton = dalton_reader.get_excited_result(self.mol, self.xc, basis, True)

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


def create_polar_table(mol, xc, basis_list, xx, database_dir):
    dalton_reader = Dalton(database_dir)
    ground_dalton, response_dalton = dalton_reader.get_frequency_result(
        mol, xc, "dipole", basis_list[0]
    )
    freq = response_dalton["frequencies"]
    g_data = {}
    xx_data = []
    for i in range(len(freq)):
        xx_data.append({})
    for basis in basis_list:
        ground_dalton, response_dalton = dalton_reader.get_frequency_result(
            mol, xc, "dipole", basis
        )
        for i in range(len(freq)):
            xx_data[i][basis] = response_dalton[xx].iloc[i]
        g_data[basis] = ground_dalton["totalEnergy"]
    g_df = pd.Series(g_data)
    g_df.name = "Total HF Energy"
    names = []
    for f in freq:
        raw_f = r"{}".format(str(f))
        # names.append(r'$$\alpha_{xx}('+raw_f+r')$$')
        names.append("a(" + "{:.3f}".format(f) + ")")
    print(xx_data)
    r_dfs = []
    for i in range(len(freq)):
        r_dfs.append(pd.Series(xx_data[i]))
        r_dfs[i].name = names[i]
    dalton_df = pd.concat([g_df] + r_dfs, axis=1)

    moldata = ResponseCalc(mol, "hf", "dipole")
    mad_data_e = {}
    mad_data_r = {}
    mad_data_e["Total HF Energy"] = moldata.ground_e["e_tot"]

    for i in range(len(names)):
        mad_data_r[names[i]] = moldata.polar_data[xx].iloc[i]

    mad_data_e = pd.Series(mad_data_e)
    mad_data_r = pd.Series(mad_data_r)

    mad_data = pd.concat([mad_data_e, mad_data_r], axis=0)
    mad_data.name = "MRA"
    mad_data.key = ["MRA"]
    data = pd.concat([dalton_df.T, mad_data.T], axis=1)
    return data.T


def create_data(mol, xc, basis_list, database_dir):
    xx = ["xx", "yy", "zz"]
    data = []
    for x in xx:
        data.append(create_polar_table(mol, xc, basis_list, x, database_dir))
    average = (data[0] + data[1] + data[2]) / 3

    diff_data = average - average.loc["MRA"]
    diff_data = diff_data.drop(index="MRA")

    polar_diff = diff_data.drop("Total HF Energy", axis=1)

    average.name = "Average Polarizability"
    energy_diff = diff_data["Total HF Energy"]

    return average, diff_data, energy_diff, polar_diff


def polar_plot(mol, xc, basis_list, ax, database_dir):
    data, diff_data, energy_diff, polar_diff = create_data(mol, xc, basis_list, database_dir)
    num_freq = len(list(polar_diff.keys()))
    sns.set_theme(style="darkgrid")
    sns.set_context("talk", font_scale=1.5, rc={"lines.linewidth": 2.5})

    btitle = basis_list[0].replace('D', 'X')

    polar_diff.iloc[:, :].plot(marker="v", linestyle="solid", markersize=12, linewidth=4,
                               colormap='magma', ax=ax, title=btitle)

    legend = []
    for i in range(num_freq):
        legend.append(r'$\omega_{}$'.format(i))

    ax.axhline(linewidth=2, ls="--", color="k")
    ax.legend(legend)
    # ax.set_xticks(rotation=20)
    yl = r" $\Delta\alpha=[\alpha($BASIS$) -\alpha($MRA$)]$"
    ax.set_ylabel(yl)
    ax.set_xlabel(ax.get_xlabel(), rotation=45)


def create_polar_diff_subplot(mol, xc, blist, dlist, database_dir):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(25, 9), constrained_layout=True)
    title = 'Polarizability Convergence: ' + mol
    fig.suptitle(title)

    polar_plot(mol, xc, blist, ax[0], database_dir)
    polar_plot(mol, xc, dlist, ax[1], database_dir)

    btitle = blist[0].replace('D', 'X')
    save = mol + "-" + btitle
    if not os.path.exists("acs_mols"):
        os.mkdir("acs_mols")
    save = "acs_mols/" + save + ".svg"
    fig.savefig(save)
    return fig


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


def create_polar_mol_series(mol, basis):
    data = ResponseCalc(mol, "hf", "dipole")

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


def plot_norm_and_residual_excited(d, num_i, ax):
    fkeys = get_function_keys(d.num_states, d.num_orbitals)
    abs_keys = fkeys["x_abs_error"]
    rel_keys = fkeys["x_rel_error"]
    chi_norms_keys = fkeys["x_norms"]

    print("chi norms key", chi_norms_keys)
    print("abs norms key", abs_keys)
    print("rel norms key", rel_keys)

    params = d.params
    dconv = params["dconv"]

    chi_norms_i = d.function_data[chi_norms_keys]
    bsh_residuals_i = d.function_data[abs_keys]
    rel_residuals_i = d.function_data[rel_keys]

    chi_norms_i.plot(ax=ax[0], logy=False, legend=True, colormap='magma', title='Chi Norms', marker='*', grid=True)
    bsh_residuals_i.plot(ax=ax[1], logy=True, legend=True, colormap='magma', title='Absolute Residuals', marker='*',
                         grid=True)
    rel_residuals_i.plot(ax=ax[2], logy=True, legend=True, colormap='magma', title='Relative Residuals', marker='*',
                         grid=True)

    iters = d.num_iter_proto
    print(iters)

    for pc in iters:
        for i in range(3):
            ax[i].axvline(x=pc, ymin=0, ymax=1, c="black", linestyle="dashed")
    for i in range(3):
        if i != 0:
            ax[i].axhline(y=dconv, xmin=0, xmax=iters[-1], c="black", linestyle="dashed", )
        ax[i].grid(which="both")
        ax[i].minorticks_on()
        ax[i].tick_params(which="both", top="on", left="on", right="on", bottom="on", )


def excited_state_norm_residual_plot(mol, xc, save):
    d = ExcitedData(mol, xc)
    xkeys = []
    ykeys = []
    for i in range(d.num_states):
        xkeys.append("x" + str(i))
        ykeys.append("y" + str(i))
    mad_read = MadnessReader()

    sns.set_theme(style="darkgrid")
    sns.set_context("talk", font_scale=1.5, rc={"lines.linewidth": 2.5})

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25, 9), constrained_layout=True)
    title = 'Excited_State Convergence: ' + mol
    fig.suptitle(title)
    plot_norm_and_residual_excited(d, i, ax)
    plotname = 'excited_{}'.format(i) + ".svg"
    if save:
        if not os.path.exists("convergence"):
            os.mkdir("convergence")
        if not os.path.exists('convergence/' + mol):
            os.mkdir("convergence/" + mol)
        if not os.path.exists('convergence/' + mol + '/' + xc):
            os.mkdir("convergence/" + mol + '/' + xc)
        plt.savefig("convergence/" + mol + '/' + xc + '/' + plotname)
    print(mol + "\n converged: ", d.converged)


def plot_norm_and_residual_freq(d, num_i, ax):
    fkeys = get_function_keys(d.num_states, d.num_orbitals)
    abs_keys = fkeys["x_abs_error"]
    rel_keys = fkeys["x_rel_error"]
    chi_norms_keys = fkeys["x_norms"]

    # print("chi norms key",chi_norms_keys)
    # print("abs norms key",abs_keys)
    # print("rel norms key",rel_keys)
    f_key = list(d.function_data.keys())[num_i]
    dconv = d.params[f_key]["dconv"]

    chi_norms_i = d.function_data[f_key][chi_norms_keys]
    bsh_residuals_i = d.function_data[f_key][abs_keys]
    rel_residuals_i = d.function_data[f_key][rel_keys]

    chi_norms_i.plot(ax=ax[0], logy=False, legend=True, colormap='magma', title='Chi Norms', marker='*', grid=True)
    bsh_residuals_i.plot(ax=ax[1], logy=True, legend=True, colormap='magma', title='Absolute Residuals', marker='*',
                         grid=True)
    rel_residuals_i.plot(ax=ax[2], logy=True, legend=True, colormap='magma', title='Relative Residuals', marker='*',
                         grid=True)

    iters = d.num_iter_proto[f_key]

    for pc in iters:
        for i in range(3):
            ax[i].axvline(x=pc, ymin=0, ymax=1, c="black", linestyle="dashed")
    for i in range(3):
        if i != 0:
            ax[i].axhline(y=dconv, xmin=0, xmax=iters[-1], c="black", linestyle="dashed", )
        ax[i].grid(which="both")
        ax[i].minorticks_on()
        ax[i].tick_params(which="both", top="on", left="on", right="on", bottom="on", )


def freq_norm_and_residual(mol, xc, op, save):
    d = ResponseCalc(mol, xc, op)

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
        plot_norm_and_residual_freq(d, i, ax)
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
