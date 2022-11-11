from madness_reader_v2 import *
import json
import seaborn as sns
import glob

sns.set_context("paper")
sns.set_theme(style="whitegrid")


def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]


def to_table(mol_list, n):
    mol_list.sort(),
    molecules = list(chunkify(mol_list, n))
    mol_table = pd.DataFrame(molecules)
    return mol_table


class QCDatabase:

    def __init__(self, database_dir):
        self.database_dir = database_dir
        self.mol_dir = self.database_dir + "/molecules"

        self.mol_list = []

        for g in glob.glob(self.mol_dir + '/*.mol'):
            m = g.split('/')
            mol = m[-1].split('.')[0]
            self.mol_list.append(mol)

        self.num_molecules = len(self.mol_list)
        self.dalton_reader = Dalton(database_dir, False)


class FrequencyDatabase2(QCDatabase):
    def __init__(self, database_dir, xc, op, num_freq):
        super().__init__(database_dir)
        self.xc = xc
        self.op = op
        self.num_freq = num_freq
        self.energy_string = 'Total ' + xc.upper() + ' Energy'
        self.mol_list = self.report_convergence()[0]  # only keep molecules that are completely converged

    def report_convergence(self):
        converged = []
        not_converged = []
        not_found = []
        type_error = []
        json_error = []
        for mol in self.mol_list:
            try:
                check_mol = ResponseCalc(mol, self.xc, self.op, self.database_dir)
                if check_mol.converged.all() and check_mol.converged.sum() == self.num_freq:
                    converged.append(mol)
                else:
                    not_converged.append(mol)
            except FileNotFoundError as f:
                not_found.append(mol)
            except TypeError as f:
                type_error.append(mol)
            except json.decoder.JSONDecodeError as j:
                json_error.append(mol)

        num_c = len(converged)
        num_n = len(not_converged)
        num_nf = len(not_found)
        num_json_e = len(json_error)
        num_type_e = len(type_error)
        total = num_c + num_n + num_nf + num_json_e + num_type_e
        not_converged = []
        part_converged = []
        if True:
            for mol in not_converged:
                check = ResponseCalc(mol, self.xc, self.op, self.database_dir)
                if check.converged.any():
                    # print(mol,'\n',check.converged)
                    part_converged.append(mol)
                else:
                    not_converged.append(mol)
        num_not_converged = len(not_converged)
        num_part_converged = len(part_converged)
        print("converged : ", num_c)
        print("not converged : ", num_n)
        print("not found : ", num_nf)
        print("json error : ", num_json_e)
        print("type error : ", num_type_e)
        print("total : ", total)
        print("fully not converged", num_not_converged)
        print("num partly fully converged", num_part_converged)
        return converged, part_converged, not_converged, not_found, type_error, json_error

    def compare_dalton(self, mol, basis):
        ground_dalton, response_dalton = self.dalton_reader.get_frequency_result(
            mol, self.xc, self.op, basis
        )
        r_calc = ResponseCalc(mol, self.xc, self.op, self.database_dir)
        ground_compare = pd.concat(
            [
                ground_dalton,
                pd.Series(r_calc.ground_timing, index=["wall-time"]),
                pd.Series(r_calc.ground_e),
            ]
        )
        freq = response_dalton.iloc[:, 0]
        polar_df = r_calc.polar_data.iloc[:, 3:].reset_index(drop=True)
        polar_df = pd.concat([freq, polar_df], axis=1)

        diff_df = pd.concat(
            [polar_df.iloc[:, 0], polar_df.iloc[:, 1:] - response_dalton.iloc[:, 1:]],
            axis=1,
        )

        return ground_compare, response_dalton, polar_df, diff_df

    def frequency_basis_error(self, mol, compare_basis):
        gd, rd, md, dd = self.compare_dalton(mol, compare_basis)
        avg_error = dd.loc[:, ['xx', 'yy', 'zz']].mean(1)
        return avg_error

    def frequency_basis_family_error(self, mol, basis_list):
        b_error = {}
        for basis in basis_list:
            b_error[basis] = self.frequency_basis_error(mol, basis)
        return pd.DataFrame(b_error)

    def create_polar_table(self, mol, basis_list, xx):
        ground_dalton, response_dalton = self.dalton_reader.get_frequency_result(
            mol, self.xc, self.op, basis_list[0]
        )
        freq = response_dalton["frequencies"]
        g_data = {}
        xx_data = []
        for i in range(len(freq)):
            xx_data.append({})
        for basis in basis_list:
            ground_dalton, response_dalton = self.dalton_reader.get_frequency_result(
                mol, self.xc, self.op, basis
            )
            for i in range(len(freq)):
                xx_data[i][basis] = response_dalton[xx][i]
            g_data[basis] = ground_dalton["totalEnergy"]
        g_df = pd.Series(g_data)
        g_df.name = self.energy_string
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

        r_calc = ResponseCalc(mol, self.xc, self.op, self.database_dir)
        mad_data_e = {}
        mad_data_r = {}
        mad_data_e[self.energy_string] = r_calc.ground_e["e_tot"]

        for i in range(len(names)):
            mad_data_r[names[i]] = r_calc.polar_data[xx][i]

        mad_data_e = pd.Series(mad_data_e)
        mad_data_r = pd.Series(mad_data_r)

        mad_data = pd.concat([mad_data_e, mad_data_r], axis=0)
        mad_data.name = "MRA"
        mad_data.key = ["MRA"]
        data = pd.concat([dalton_df.T, mad_data.T], axis=1)
        return data.T

    def get_basis_data(self, mol, basis_list):

        xx = ["xx", "yy", "zz"]
        data = []
        for x in xx:
            data.append(self.create_polar_table(mol, basis_list, x))
        average = (data[0] + data[1] + data[2]) / 3
        average.name = "Basis Data"

        return average

    def get_spherical_polarizability(self, mol, basis_list):
        average = self.get_basis_data(mol, basis_list)
        average = average.drop(self.energy_string, axis=1)
        return average

    def get_spherical_polarizability_error(self, mol, basis_list):

        polar = self.get_spherical_polarizability(mol, basis_list)
        polar_error = (polar - polar.loc["MRA"]) / polar.loc["MRA"]

        polar_error = polar_error.drop(index="MRA")
        return polar_error

    def create_polar_diff_plot(self, mol, basis_list):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), constrained_layout=True)
        title = r'Basis-set convergence of $\alpha(\omega)$ : ' + mol
        yl = r" $\Delta\alpha=\alpha($BASIS$) -\alpha($MRA$)$"
        polar_error = self.get_spherical_polarizability_error(mol, basis_list)
        num_freq = len(list(polar_error.keys()))
        polar_error.iloc[:, :].plot(marker="v", linestyle="solid", markersize=12, linewidth=4, colormap='magma', ax=ax)
        legend = []
        for i in range(num_freq):
            legend.append(r'$\omega_{}$'.format(i))
        plt.axhline(linewidth=5, ls="--", color="k")

        plt.legend(legend)
        plt.xticks(rotation=20, fontsize=18)
        plt.title(title)
        plt.ylabel(yl, fontsize=24)
        return fig

    def get_molecule_basis_data(self, mol, basis_list):

        col_idx = [self.energy_string]
        for i in range(self.num_freq):
            col_idx.append(r'$\alpha(\omega_{})$'.format(i))

        data = self.get_basis_data(mol, basis_list)
        data.columns = col_idx
        data.index.name = 'basis'
        mol_series = pd.Series([mol for i in range(len(data))])
        mol_series.name = 'molecule'
        mtable = mol_series.to_frame()
        mtable.index = data.index
        data = pd.concat([mtable, data], axis=1, ignore_index=False)
        return data

    def get_basis_set_data(self, mol_list, basis_list):
        data = pd.DataFrame()
        for mol in mol_list:
            mol_data = self.get_molecule_basis_data(mol, basis_list)
            data = pd.concat([data, mol_data])
        return data

    def get_molecule_basis_error(self, mol, basis_list):

        col_idx = [self.energy_string]
        for i in range(self.num_freq):
            col_idx.append(r'$\alpha(\omega_{})$'.format(i))

        data = self.get_basis_data(mol, basis_list)
        error = data - data.loc["MRA"]
        error.index.name = 'basis'
        error.columns = col_idx
        error = error.drop(index="MRA")
        mol_series = pd.Series([mol for i in range(len(error))])
        mol_series.name = 'molecule'
        mtable = mol_series.to_frame()
        mtable.index = error.index
        error = pd.concat([mtable, error], axis=1, ignore_index=False)
        return error

    def get_basis_set_error(self, mol_list, basis_list):

        data = pd.DataFrame()
        for mol in mol_list:
            try:
                mol_data = self.get_molecule_basis_error(mol, basis_list)
                # concatenate if there is no problem
                data = pd.concat([data, mol_data])
            except ValueError as v:
                # report molecules with a problem
                print(mol)
                print(v)
                pass
        return data

    def partition_basis_data(self, mol_list, basis_list):
        yes = []
        no = []
        for mol in mol_list:
            try:
                for b in basis_list:
                    if not self.dalton_reader.polar_json_exists(mol, self.xc, self.op, b):
                        raise ValueError(mol + " not found with basis " + b)
                yes.append(mol)
            except ValueError as v:
                print(v)
                no.append(mol)
                pass
        return yes


