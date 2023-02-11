from qcdatabase import QCDatabase as qcb
from qcdatabase import FrequencyDatabase as FDatabase
from qcdatabase import PolarBasisComparisonDatabase as polar_basis_comparison_db
from qcdatabase import PartitionedDatabase as partioned_db
from qcdatabase import BasisErrorDataSet as BSE
from qcdatabase import *

# from qcdatabase import comparison_violin_plot
from qcdatabase import chunkify
from qcdatabase import to_table
from dalton import Dalton
from madnessReader import FrequencyData
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import dataframe_image as dfi


singly_augmented = ["aug-cc-pVDZ", "aug-cc-pVTZ", "aug-cc-pVQZ", "aug-cc-pV5Z"]
doubly_augmented = ["d-aug-cc-pVDZ", "d-aug-cc-pVTZ", "d-aug-cc-pVQZ", "d-aug-cc-pV5Z"]
singly_DTQ = ["aug-cc-pVDZ", "aug-cc-pVTZ", "aug-cc-pVQZ"]
doubly_DTQ = ["d-aug-cc-pVDZ", "d-aug-cc-pVTZ", "d-aug-cc-pVQZ"]
single_polarized = ["aug-cc-pCVDZ", "aug-cc-pCVTZ", "aug-cc-pCVQZ"]
double_polarized = ["d-aug-cc-pCVDZ", "d-aug-cc-pCVTZ", "d-aug-cc-pCVQZ"]
all_basis = singly_augmented + doubly_augmented + single_polarized + double_polarized
basis_list = single_polarized + double_polarized + singly_DTQ + doubly_DTQ


class PolarPaperFunctions:
    def __init__(self, data_dir, xc, op):
        self.data_dir = data_dir
        self.xc = xc
        self.op = op
        self.mra_data = FDatabase(self.data_dir, xc, op, 9)

    def get_E_relative_diff(self, mol, basis_list):
        def __init__(self, data_dir):
            self.data_dir = data_dir
            with open(self.data_dir + "/molecules/frequency.json") as json_file:
                self.freq_json = json.loads(json_file.read())

        dalton_reader = Dalton(self.data_dir, False)
        f_data = FrequencyData(mol, self.xc, self.op, self.data_dir)
        e_mad = f_data.ground_e["e_tot"]
        compare_dict = {}
        compare_dict["MRA"] = e_mad
        for basis in basis_list:
            try:
                ground_dalton, response_dalton = dalton_reader.get_frequency_result(
                    mol, self.xc, self.op, basis
                )
                energy = ground_dalton["totalEnergy"]
                compare_dict[basis] = energy
            except:
                compare_dict[basis] = np.nan
                pass
        e_df = pd.Series(compare_dict)
        rel_error = e_df.subtract(e_df["MRA"], axis=0)
        rel_error = rel_error.iloc[1:].copy()
        re = rel_error.reset_index(drop=True)
        re.name = "error"
        b_s = pd.Series(basis_list)
        b_s.name = "basis"
        mol_s = pd.Series([mol for i in range(len(basis_list))])
        mol_s.name = "molecule"
        return pd.concat([mol_s, b_s, re], axis=1)

    def get_polar_avg(self, mol, basis_list, ij_j_list):

        dalton_reader = Dalton(self.data_dir, False)
        f_data = FrequencyData(mol, self.xc, self.op, self.data_dir)

        compare_dict = {}
        polar_df = f_data.polar_df.iloc[:, 3:].reset_index(drop=True)
        mad_col = polar_df[ij_j_list].iloc[:]
        mad_col.name = "MRA"
        compare_dict["MRA"] = mad_col.mean(axis=1)
        for basis in basis_list:
            try:
                ground_dalton, response_dalton = dalton_reader.get_frequency_result(
                    mol, self.xc, self.op, basis
                )
                col = response_dalton[ij_j_list]
                col.name = basis
                compare_dict[basis] = col.mean(axis=1)
            except:
                compare_dict[basis] = pd.Series([np.nan for i in range(9)], name=basis)
                pass
        polar_df = pd.concat(compare_dict, axis=1)
        polar_df.index.name = "omega"
        return polar_df

    def get_polar_relative_diff(self, mol, basis_list, ij_j_list):

        dalton_reader = Dalton(self.data_dir, False)
        f_data = FrequencyData(mol, self.xc, self.op, self.data_dir)

        compare_dict = {}
        polar_df = f_data.polar_df.iloc[:, 3:].reset_index(drop=True)
        mad_col = polar_df[ij_j_list].iloc[:]
        mad_col.name = "MRA"
        compare_dict["MRA"] = mad_col.mean(axis=1)
        for basis in basis_list:
            try:
                ground_dalton, response_dalton = dalton_reader.get_frequency_result(
                    mol, self.xc, self.op, basis
                )
                col = response_dalton[ij_j_list]
                col.name = basis
                compare_dict[basis] = col.mean(axis=1)
            except:
                compare_dict[basis] = pd.Series([np.nan for i in range(9)], name=basis)
                pass
        polar_df = pd.concat(compare_dict, axis=1)
        polar_df.index.name = "omega"
        rel_error = (
            polar_df.subtract(polar_df["MRA"].values, axis=0).div(
                polar_df["MRA"].values, axis=0
            )
            * 100
        )
        rel_error.index.name = "omega"
        return rel_error.iloc[:, 1:]

    def get_polar_diff(self, mol, basis_list, ij_j_list):

        dalton_reader = Dalton(self.data_dir, False)
        f_data = FrequencyData(mol, self.xc, self.op, self.data_dir)

        compare_dict = {}
        polar_df = f_data.polar_df.iloc[:, 3:].reset_index(drop=True)
        mad_col = polar_df[ij_j_list].iloc[:]
        mad_col.name = "MRA"
        compare_dict["MRA"] = mad_col.mean(axis=1)
        for basis in basis_list:
            try:
                ground_dalton, response_dalton = dalton_reader.get_frequency_result(
                    mol, self.xc, self.op, basis
                )
                col = response_dalton[ij_j_list]
                col.name = basis
                compare_dict[basis] = col.mean(axis=1)
            except:
                compare_dict[basis] = pd.Series([np.nan for i in range(9)], name=basis)
                pass
        polar_df = pd.concat(compare_dict, axis=1)
        polar_df.index.name = "omega"
        rel_error = polar_df.subtract(
            polar_df["MRA"].values, axis=0
        )  # .div(polar_df['MRA'].values,axis=0)
        rel_error.index.name = "omega"
        return rel_error.iloc[:, 1:]

    def get_polar_col(self, mol, basis_list):
        polar_df = self.get_polar_avg(mol, basis_list, ["xx", "yy", "zz"])
        mol_df = pd.DataFrame(dtype=np.float64)
        for basis in polar_df.keys():
            basis_vals = polar_df[basis].copy()
            basis_series = pd.Series(
                [basis_vals.name for i in range(basis_vals.size)], name="basis"
            )
            basis_vals.name = "alpha"
            basis_df = pd.concat([basis_series, basis_vals], axis=1)
            mol_df = pd.concat([mol_df, basis_df])

        mol_series = pd.Series(
            [mol for i in range(mol_df.shape[0])], name="molecule", index=mol_df.index
        )
        mol_df = pd.concat([mol_df, mol_series], axis=1)
        return mol_df

    def get_polar_diff_col(self, mol, basis_list):
        polar_df = self.get_polar_diff(mol, basis_list, ["xx", "yy", "zz"])
        mol_df = pd.DataFrame(dtype=np.float64)
        for basis in polar_df.keys():
            basis_vals = polar_df[basis].copy()
            basis_series = pd.Series(
                [basis_vals.name for i in range(basis_vals.size)], name="basis"
            )
            basis_vals.name = "error"
            basis_df = pd.concat([basis_series, basis_vals], axis=1)
            mol_df = pd.concat([mol_df, basis_df])

        mol_series = pd.Series(
            [mol for i in range(mol_df.shape[0])], name="molecule", index=mol_df.index
        )
        mol_df = pd.concat([mol_df, mol_series], axis=1)
        mol_df["basis"] = mol_df["basis"].astype("category")
        return mol_df

    def get_polar_rel_diff_col(self, mol, basis_list):
        polar_df = self.get_polar_relative_diff(mol, basis_list, ["xx", "yy", "zz"])
        mol_df = pd.DataFrame(dtype=np.float64)
        for basis in polar_df.keys():
            basis_vals = polar_df[basis].copy()
            basis_series = pd.Series(
                [basis_vals.name for i in range(basis_vals.size)], name="basis"
            )
            basis_vals.name = "error"
            basis_df = pd.concat([basis_series, basis_vals], axis=1)
            mol_df = pd.concat([mol_df, basis_df])
        mol_series = pd.Series(
            [mol for i in range(mol_df.shape[0])], name="molecule", index=mol_df.index
        )
        mol_df = pd.concat([mol_df, mol_series], axis=1)
        mol_df["basis"] = mol_df["basis"].astype("category")
        mol_df["molecule"] = mol_df["molecule"].astype("category")
        return mol_df

    def get_polar_data(self, mol_list, basis_list):
        polar_data = pd.DataFrame()
        for mol in mol_list:
            mol_col = self.get_polar_col(mol, basis_list)
            polar_data = pd.concat([polar_data, mol_col])
        polar_data["basis"] = polar_data["basis"].astype("category")
        polar_data["molecule"] = polar_data["molecule"].astype("category")
        return polar_data

    # gets the relative error
    def get_polar_RE(self, mol_list, basis_list):
        rel_error = pd.DataFrame()
        for mol in mol_list:
            mol_col = self.get_polar_rel_diff_col(mol, basis_list)
            rel_error = pd.concat([rel_error, mol_col])

        rel_error["basis"] = rel_error["basis"].astype("category")
        rel_error["molecule"] = rel_error["molecule"].astype("category")
        return rel_error

    def get_polar_ERROR(self, mol_list, basis_list):
        error = pd.DataFrame()
        for mol in mol_list:
            mol_col = self.get_polar_diff_col(mol, basis_list)
            error = pd.concat([error, mol_col])
        error["basis"] = error["basis"].astype("category")
        error["molecule"] = error["molecule"].astype("category")
        return error

    def get_energy_RE(self, mol_list, basis_list):
        e_data = pd.DataFrame()
        for mol in mol_list:
            mol_col = self.get_E_relative_diff(mol, basis_list)
            e_data = pd.concat([e_data, mol_col])
        e_data["basis"] = e_data["basis"].astype("category")
        e_data["molecule"] = e_data["molecule"].astype("category")
        return e_data


class PartDataFunctions:
    def __init__(self, energy_RE, polar_data, polar_RE, polar_E):
        self.polar_data = polar_data
        self.energy_RE = energy_RE
        self.polar_RE = polar_RE
        self.polar_E = polar_E

    # All the tables
    def sort_by_first_basis(self, data, basis_sets, i, freq):
        er_aug_idx = data.error.abs() < 10
        freq_idx = data.index == freq
        out_mol = data[(data.basis.isin(basis_sets)) & er_aug_idx & freq_idx].molecule
        data_aug_out = {}
        for mol in out_mol:
            data_aug_out[mol] = {}
            for basis in basis_sets:
                mol_idx = data.molecule == mol
                b_idx = data.basis == basis
                data_aug_out[mol][basis] = data[mol_idx & b_idx & freq_idx].error[0]

        data_out = pd.DataFrame(data_aug_out).T
        out_out = (
            data_out.loc[:, basis_sets]
            .sort_values(by=[basis_sets[i]], ascending=False)
            .index
        )
        data_out = data_out.loc[out_out]
        data_out = data_out.iloc[:, :]

        return data_out

    def partition_and_sort(self, df, cz, sort_i, freq):
        data = df[df.index == 0]
        all_data = self.sort_by_first_basis(data, cz, sort_i, freq)
        first_row, second_row, Flist = get_mol_lists(all_data.index.to_list())
        # split the cells and append the MRA reference
        mra_data = self.polar_data[self.polar_data.basis == "MRA"]
        mra_data.index.name = "frequency"
        mra_data = mra_data.reset_index()
        mra_data = mra_data.set_index("molecule")
        mra_data_0 = mra_data[mra_data.frequency == 0]
        mra_data_0 = mra_data_0.drop(columns=["frequency"])
        mra_data.index.name = ""
        polar_mra = mra_data_0.loc[all_data.index, "alpha"]
        polar_mra.name = "MRA Ref"
        all_data = pd.concat([all_data, polar_mra], axis=1)
        print(first_row)
        data1 = all_data.loc[
            first_row + Flist,
        ]
        data2 = all_data.loc[
            second_row,
        ]
        return all_data, data1, data2


def make_full_data_pretty(styler, cz, EE):
    styler.format("{:.2e}", precision=4, subset=cz)
    styler.format("{:.4g}", precision=4, subset=["MRA Ref"])
    styler.background_gradient(
        axis=1, vmin=-EE, vmax=EE, cmap="coolwarm", subset=cz, low=0, high=0
    )
    return styler


def get_mol_lists(mol_list):
    Flist = []
    row2 = []
    rest = []
    seconds = ["Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ne", "Ar"]
    for mol_i in mol_list:
        if "F" in mol_i and not any([e2 in mol_i for e2 in seconds]):
            Flist.append(mol_i)
        elif any([e2 in mol_i for e2 in seconds]):
            row2.append(mol_i)
        else:
            rest.append(mol_i)
    return rest, row2, Flist


def highlight_positive(x):
    return np.where(
        x == x[7], "font-weight: bold;background-color: white; color: black;", None
    )


def make_pretty_summary(styler):
    styler.format("{:.3f}", precision=0)
    styler.background_gradient(
        axis=1, vmin=-EE, vmax=EE, cmap="coolwarm", low=0, high=0
    )
    styler.apply(highlight_positive, axis=0)
    return styler


def make_summary_polar(data, bs):
    subset = data.loc[:, data.columns.isin(bs)]
    num_positive = (subset > 0).sum()
    num_positive.name = "Postive"
    num_positive.convert_dtypes(int)
    summary = subset.describe()[1:][bs]
    summary = pd.concat([summary.T, num_positive], axis=1).T
    return summary


def sort_energy_by_first_basis(data, basis_sets, i):
    er_aug_idx = data.error.abs() < 10
    out_mol = data[(data.basis.isin(basis_sets)) & er_aug_idx].molecule
    data_aug_out = {}
    for mol in out_mol:
        data_aug_out[mol] = {}
        for basis in basis_sets:
            mol_idx = data.molecule == mol
            b_idx = data.basis == basis
            dval = data[mol_idx & b_idx].error.to_numpy()
            if dval.any():
                data_aug_out[mol][basis] = dval[0]
            else:
                data_aug_out[mol][basis] = np.nan

    data_out = pd.DataFrame.from_dict(data_aug_out).T
    out_out = (
        data_out.loc[:, basis_sets]
        .sort_values(by=[basis_sets[i]], ascending=False)
        .index
    )
    data_out = data_out.loc[out_out]
    data_out = data_out.iloc[:, :]

    return data_out


def make_e_pretty(styler):

    styler.format("{:.2e}", precision=0)
    styler.background_gradient(
        axis=1, vmin=-0.01, vmax=0.05, cmap="Blues", low=0, high=0
    )

    return styler


def decribe_basis(data, cz):
    subset = data.loc[:, data.columns.isin(cz)]
    summary = subset.describe()[1:][cz]
    return summary


# compare aug/d-aug DTQ
def aug_split_data(rel_error):
    aug_rename = ["cc-pVDZ", "cc-pVTZ", "cc-pVQZ", "cc-pV5Z"]
    i = 0
    full_aug = pd.DataFrame()
    for basis in singly_augmented:
        aug_data = rel_error[rel_error.basis == basis].copy()
        aug_data.basis.replace(basis, aug_rename[i], inplace=True)
        aug_data = pd.concat(
            [
                aug_data,
                pd.Series(
                    ["aug" for i in range(aug_data.shape[0])],
                    name="aug-type",
                    index=aug_data.index,
                    dtype=str,
                ),
            ],
            axis=1,
        )
        full_aug = pd.concat([full_aug, aug_data])
        i += 1
    i = 0
    full_daug = pd.DataFrame()
    for basis in doubly_augmented:
        aug_data = rel_error[rel_error.basis == basis].copy()
        aug_data.basis.replace(basis, aug_rename[i], inplace=True)
        aug_data = pd.concat(
            [
                aug_data,
                pd.Series(
                    ["d-aug" for i in range(aug_data.shape[0])],
                    name="aug-type",
                    index=aug_data.index,
                    dtype=str,
                ),
            ],
            axis=1,
        )
        full_daug = pd.concat([full_daug, aug_data])
        i += 1
    data = pd.concat([full_aug, full_daug])
    return data


# compare aug/d-aug DTQ
def pc_split_data(rel_error):
    aug_rename = [
        "aug-cc-pVDZ",
        "aug-cc-pVTZ",
        "aug-cc-pVQZ",
        "d-aug-cc-pVDZ",
        "d-aug-cc-pVTZ",
        "d-aug-cc-pVQZ",
    ]
    i = 0
    full_aug = pd.DataFrame()
    for basis in singly_DTQ + doubly_DTQ:
        aug_data = rel_error[rel_error.basis == basis].copy()
        aug_data.basis.replace(basis, aug_rename[i], inplace=True)
        aug_data = pd.concat(
            [
                aug_data,
                pd.Series(
                    [False for i in range(aug_data.shape[0])],
                    name="pC",
                    index=aug_data.index,
                ),
            ],
            axis=1,
        )
        full_aug = pd.concat([full_aug, aug_data])
        i += 1
    i = 0
    full_daug = pd.DataFrame()
    for basis in single_polarized + double_polarized:
        aug_data = rel_error[rel_error.basis == basis].copy()
        aug_data.basis.replace(basis, aug_rename[i], inplace=True)
        aug_data = pd.concat(
            [
                aug_data,
                pd.Series(
                    [True for i in range(aug_data.shape[0])],
                    name="pC",
                    index=aug_data.index,
                ),
            ],
            axis=1,
        )
        full_daug = pd.concat([full_daug, aug_data])
        i += 1
    data = pd.concat([full_aug, full_daug])
    return data
