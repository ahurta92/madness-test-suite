import copy
import os
import pandas as pd
from madnessReader import MadnessReader
from madnessReader import ExcitedData
from madnessReader import FrequencyData
from madnessReader import *
from daltonRunner import DaltonRunner
import matplotlib.pyplot as plt
import json
import seaborn as sns
import glob

sns.set_context("paper")
sns.set_theme(style="whitegrid")


def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]


def to_table(mol_list, n, series_name, database_dir):
    paper_dir = database_dir + "/paper"
    mol_list.sort(),
    molecules = list(chunkify(mol_list, n))
    mol_table = pd.DataFrame(molecules)
    m = mol_table.style.hide(axis=0).hide(axis=1)
    latex_table = paper_dir + "/" + series_name + ".tex"
    m.to_latex(latex_table)
    print(mol_table.to_markdown())


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


class FrequencyDatabase(QCDatabase):
    def __init__(self, database_dir, xc, op, num_freq):
        super().__init__(database_dir)
        self.xc = xc
        self.op = op
        self.num_freq = num_freq
        self.energy_string = 'Total ' + xc.upper() + ' Energy'

    def report_convergence(self):
        converged = []
        not_converged = []
        not_found = []
        type_error = []
        json_error = []
        for mol in self.mol_list:
            try:
                check_mol = FrequencyData(mol, self.xc, self.op, self.database_dir)
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
                check = FrequencyData(mol, self.xc, self.op, self.database_dir)
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

    def frequency_basis_error(self, mol, compare_basis):
        data = FrequencyData(mol, self.xc, self.op, self.database_dir)
        gd, rd, md, dd = data.compare_dalton(compare_basis, self.database_dir)
        avg_error = dd.loc[:, ['xx', 'yy', 'zz']].mean(1)
        return avg_error

    def frequency_basis_family_error(self, mol, basis_list):
        data = FrequencyData(mol, self.xc, self.op, self.database_dir)
        # display('madness',md.round(4))
        # display(compare_basis,rd.round(4))
        b_error = {}
        for basis in basis_list:
            gd, rd, md, dd = data.compare_dalton(basis, self.database_dir)
            avg_error = dd.loc[:, ['xx', 'yy', 'zz']].mean(1)
            b_error[basis] = avg_error
        return pd.DataFrame(b_error)

    def create_polar_table(self, mol, basis_list, xx):
        dalton_reader = DaltonRunner(self.database_dir)
        ground_dalton, response_dalton = dalton_reader.get_frequency_result(
            mol, self.xc, self.op, basis_list[0]
        )
        freq = response_dalton["frequencies"]
        g_data = {}
        xx_data = []
        for i in range(len(freq)):
            xx_data.append({})
        for basis in basis_list:
            ground_dalton, response_dalton = dalton_reader.get_frequency_result(
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

        moldata = FrequencyData(mol, "hf", "dipole", self.database_dir)
        mad_data_e = {}
        mad_data_r = {}
        mad_data_e[self.energy_string] = moldata.ground_e["e_tot"]

        for i in range(len(names)):
            mad_data_r[names[i]] = moldata.polar_df[xx][i]

        mad_data_e = pd.Series(mad_data_e)
        mad_data_r = pd.Series(mad_data_r)

        mad_data = pd.concat([mad_data_e, mad_data_r], axis=0)
        mad_data.name = "MRA"
        mad_data.key = ["MRA"]
        data = pd.concat([dalton_df.T, mad_data.T], axis=1)
        return data.T

    def get_basis_data(self, mol, basis_list):
        # Get energy and polarizability data for molecule

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
        polar_error = (polar - polar.loc["MRA"]) / polar.loc["MRA"] * 100

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
            mol_data = self.get_molecule_basis_error(mol, basis_list)
            data = pd.concat([data, mol_data])
        return data

    def partition_basis_data(self, mol_list, basis_list):
        dalton_reader = DaltonRunner(self.database_dir)
        yes = []
        no = []
        for mol in mol_list:
            try:
                for b in basis_list:
                    ground_dalton, response_dalton = dalton_reader.get_frequency_result(mol, self.xc, self.op, b)
                yes.append(mol)
            except:
                no.append(mol)
                pass
        return yes


class PolarBasisComparisonDatabase:
    def __init__(self, f_data: FrequencyDatabase, mol_list, basis_list):
        self.basis_list = basis_list
        self.mol_list = f_data.partition_basis_data(mol_list, basis_list)  # grab molecules with available Dalton data
        self.data = f_data.get_basis_set_error(self.mol_list, self.basis_list)
        self.data_list = list(self.data.keys())

    def from_data(self, data):
        new_data = copy.deepcopy(self)
        new_data.data = data
        new_data.mol_list = list(data.molecule.unique())
        new_data.basis_list = list(data.index.unique())
        new_data.data_list = list(self.data.keys())
        return new_data

    def get_outliers(self, basis, dtype, cuttoff):
        basis_data = self.data.loc[basis]

        outlier_df = basis_data.loc[basis_data.loc[:, dtype].abs() > cuttoff]
        outlier_list = list(outlier_df.loc[:, 'molecule'])

        out_idx = self.data.molecule.isin(outlier_list)

        keep = self.data.loc[~out_idx, :]
        remove = self.data.loc[out_idx, :]

        return self.from_data(keep), self.from_data(remove)

    def partition_data_by_type(self, blist, dlist, partition_string):
        d1 = self.data.loc[self.data.index.isin(blist)]  # not augmented
        num_data = len(list(d1.index))
        d2 = self.data.loc[self.data.index.isin(dlist)]  # augmented
        daug = [True for i in range(num_data)]

        aug_true = pd.Series(daug)
        aug_true.index = d1.index
        aug_true.name = partition_string

        aug_false = ~aug_true
        aug_false.index = d1.index
        aug_false.name = partition_string

        d2.reset_index(drop=True)
        d2.index = d1.index

        d2 = pd.concat([d2, aug_true], axis=1)
        d1 = pd.concat([d1, aug_false], axis=1)

        return self.from_data(pd.concat([d1, d2])).data


class PartitionedDatabase(PolarBasisComparisonDatabase):
    def __init__(self, f_data: FrequencyDatabase, basis_list, blist, dlist, partition_type):
        super().__init__(f_data, f_data.mol_list, basis_list)

        self.partition_type = partition_type
        self.basis1 = blist
        self.basis2 = dlist

        self.data = self.partition_data_by_type(blist, dlist, partition_type)


def comparison_violin_plot(db: PartitionedDatabase, data_id, hue_label, ax):
    p1 = sns.violinplot(x=db.data.index, y=db.data[data_id], hue=db.data[hue_label], ax=ax, split=True,
                        scale="count", inner="quartile", scale_hue=False)
    p11 = sns.swarmplot(x=db.data.index, y=db.data[data_id], hue=db.data[hue_label], ax=ax, size=3.2, dodge=True,
                        palette='dark:.2')
    ax.axhline(y=0, linewidth=4, ls="--", color="r")
    handles, labels = ax.get_legend_handles_labels()
    if hue_label == 'd-aug':
        ax.legend(handles[:2], ['aug', 'd-aug'])
    elif hue_label == 'polarized_core':
        ax.legend(handles[:2], ['non-polarized', 'polarized'])

    ax.set_title(data_id)  # fontsize=30)
    ax.set_xlabel('basis')  # , fontsize=28)
    ax.set_ylabel('relative Error')
    #ax.set_ylabel(r'$\Delta$' + data_id + ' : [(basis) - (MRA)]')  # , fontsize=28)

    # ax.set_title(data_id, fontsize=30)
    # ax.set_xlabel('Basis', fontsize=28)
    # ax.set_ylabel(r'$\Delta$' + data_id + ' : [(basis) - (MRA)]', fontsize=28)
    nblist = len(list(db.data.index.unique()))
    xticks = [i for i in range(nblist)]
    labels = ['D', 'T', 'Q', '5', '6']
    ax.set_xticks(xticks, labels=labels[0:nblist])  # fontsize=24)


