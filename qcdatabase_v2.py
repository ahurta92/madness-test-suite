from madness_reader_v2 import *
import json
import seaborn as sns
import glob


class QCDatabase:

    def __init__(self, database_dir):
        self.database_dir = database_dir
        self.mol_dir = self.database_dir + "/molecules"

        self.mol_list = []

        for g in glob.glob(self.mol_dir + '/*.mol'):
            m = g.split('/')
            mol = m[-1].split('.')[0]
            self.mol_list.append(mol)

        self.num_mols = len(self.mol_list)

    def report_converged(self, xc, op, num_converged):
        converged = []
        not_converged = []
        not_found = []
        type_error = []
        key_error = []
        json_error = []
        for mol in self.mol_list:
            try:
                ben = FrequencyData(mol, xc, op, self.database_dir)
                if ben.converged.all() and ben.converged.sum() == num_converged:
                    converged.append(mol)
                else:
                    not_converged.append(mol)
            except FileNotFoundError as f:
                not_found.append(mol)
            except TypeError as f:
                type_error.append(mol)
            except KeyError as f:
                key_error.append(mol)
            except json.decoder.JSONDecodeError as j:
                json_error.append(mol)

        num_c = len(converged)
        num_n = len(not_converged)
        num_nf = len(not_found)
        num_json_e = len(json_error)
        num_key_e = len(key_error)
        num_type_e = len(type_error)
        total = num_c + num_n + num_nf + num_json_e + num_type_e+num_key_e
        non_converged = []
        part_converged = []
        if True:
            for mol in not_converged:
                check = FrequencyData(mol, xc, op, self.database_dir)
                if check.converged.any():
                    # print(mol,'\n',check.converged)
                    part_converged.append(mol)
                else:
                    non_converged.append(mol)
        num_not_converged = len(non_converged)
        num_part_converged = len(part_converged)
        print("converged : ", num_c)
        print("not converged : ", num_n)
        print("not found : ", num_nf)
        print("json error : ", num_json_e)
        print("type error : ", num_type_e)
        print("total : ", total)
        print("fully not converged", num_not_converged)
        print("num partly fully converged", num_part_converged)
        return converged, part_converged, non_converged, not_found, type_error, json_error
