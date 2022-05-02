import os
import pandas as pd
import numpy as np
from numpy import linalg as LA
import json

PROOT = os.getcwd()
pt = pd.read_csv(PROOT + '/periodic_table.csv')
with open(PROOT + '/molecules/frequency.json') as json_file:
    freq_json = json.loads(json_file.read())


def get_atom_dict(mol):
    madmol_f = "molecules/" + mol + ".mol"
    with open(madmol_f) as file:  # Use file to refer to the file object
        lines = file.readlines()
        atom_dict = {}
        for line in lines:
            split = line.strip().split(' ')
            while '' in split:
                split.remove('')
            if (split[0] == 'units'):
                units = split[1]
            # madness key words
            skeys = ['geometry', 'eprec', 'units', 'end']
            skey = split[0]
            # if not a key word then it's probably an atom
            if not skey in skeys:
                geometry = ' '.join(split[1:])
                if skey not in atom_dict:
                    atom_dict[skey] = []
                    atom_dict[skey].append(geometry)
                else:
                    atom_dict[skey].append(geometry)
    return atom_dict


def line2array(line):
    num_line = []
    for l in line.split():
        num_line.append(float(l))
    return np.array(num_line)


def atom_dict_to_coomatrix(atom_dict):
    atom_list = []
    coordinate_list = []
    for atom, coordinate_lines in atom_dict.items():
        for line in coordinate_lines:
            atom_list.append(atom)
            coordinate_list.append(line2array(line))

    return atom_list, np.array(coordinate_list)


def create_distance_matrix(coo):
    num_atoms = coo.shape[0]
    d_matrix = np.zeros((num_atoms, num_atoms))

    for i in range(num_atoms):
        for j in range(i, num_atoms):
            d_matrix[i, j] = LA.norm(coo[i, :] - coo[j, :])
    d_matrix = d_matrix + d_matrix.T

    return d_matrix


def compute_com(atoms, coor):
    num_atoms = coor.shape[0]
    rx = 0;
    ry = 0;
    rz = 0;
    M = 0;
    for i in range(num_atoms):
        # Look up the mass
        mi = float(pt[pt['Symbol'] == atoms[i]]['AtomicMass'])
        M += mi
        rx += mi * coor[i, 0]
        ry += mi * coor[i, 1]
        rz += mi * coor[i, 2]

    com = np.array([rx, ry, rz])
    return com / M


def compute_moment_tensor(atoms, coor):
    num_atoms = coor.shape[0]
    I = np.zeros((3, 3))

    for i in range(num_atoms):
        mi = float(pt[pt['Symbol'] == atoms[i]]['AtomicMass'])

        xi = coor[i, 0]
        yi = coor[i, 1]
        zi = coor[i, 2]

        Ixx = mi * (yi ** 2 + zi ** 2)
        Iyy = mi * (xi ** 2 + zi ** 2)
        Izz = mi * (xi ** 2 + yi ** 2)

        Ixy = -mi * (xi * yi)
        Ixz = -mi * (xi * zi)
        Iyz = -mi * (yi * zi)

        I[0, 0] += Ixx
        I[0, 1] += Ixy
        I[0, 2] += Ixz

        I[1, 0] += Ixy
        I[1, 1] += Iyy
        I[1, 2] += Iyz

        I[2, 0] += Ixz
        I[2, 1] += Iyz
        I[2, 2] += Izz

    return I


def get_symmetry_inputs(mol):
    atom_dict = get_atom_dict(mol)
    atoms, coordinates = atom_dict_to_coomatrix(atom_dict)
    atoms = np.array(atoms)
    I = compute_moment_tensor(atoms, coordinates)
    d_matrix = create_distance_matrix(coordinates)
    r = compute_com(atoms, coordinates)

    return {"atoms": atoms, "com": r, "I": I, "coor": coordinates}


def getSeaInfo(D):
    uD = []
    for i in range(D.shape[0]):
        uD.append(float(np.unique(D[:, i]).sum()))
    Dcol = D.sum(axis=1).round(4)
    uniqD = np.unique(Dcol)
    ks = []
    idx = []
    for i in range(len(uniqD)):
        ks.append(len(Dcol[uniqD[i] == Dcol]))
        np.where(uniqD[i] == Dcol)[0]
        idx.append(np.where(uniqD[i] == Dcol)[0])
    return ks, idx


def Rotation(n, theta):
    n1 = n[0]
    n2 = n[1]
    n3 = n[2]
    r1 = [np.cos(theta) + n1 ** 2 * (1 - np.cos(theta)), n1 * n2 * (1 - np.cos(theta)) - n3 * np.sin(theta),
          n1 * n3 * (1 - np.cos(theta)) + n2 * np.sin(theta)]
    r2 = [n1 * n2 * (1 - np.cos(theta)) + n3 * np.sin(theta), np.cos(theta) + n2 ** 2 * (1 - np.cos(theta)),
          n2 * n3 * (1 - np.cos(theta)) - n1 * np.sin(theta)]
    r3 = [n1 * n3 * (1 - np.cos(theta)) - n2 * np.sin(theta), n3 * n2 * (1 - np.cos(theta)) + n1 * np.sin(theta),
          np.cos(theta) + n3 ** 2 * (1 - np.cos(theta))]

    return np.array([r1, r2, r3])


class SymmetryData:

    def __init__(self, mol):

        self.SEA_atom_index = None
        self.SEA_sizes = None
        self.atom_dict = get_atom_dict(mol)
        self.atoms, self.coordinates = atom_dict_to_coomatrix(self.atom_dict)

        self.atoms = np.array(self.atoms)
        self.com = compute_com(self.atoms, self.coordinates)

        if LA.norm(self.com) > .00005:
            self.coordinates = self.coordinates - self.com
            self.com = np.array([0, 0, 0])

        self.I = compute_moment_tensor(self.atoms, self.coordinates)
        self.D = create_distance_matrix(self.coordinates)
        self.Is = self.get_all_moment_tensors()

    def get_all_moment_tensors(self):
        self.SEA_sizes, self.SEA_atom_index = getSeaInfo(self.D)

        Is = []
        for k, atom_idx in zip(self.SEA_sizes, self.SEA_atom_index):

            if k > 1:
                atoms = self.atoms[atom_idx]
                coordinates = self.coordinates[atom_idx]
                com = compute_com(atoms, coordinates)
                Ia, Ivec = LA.eig(compute_moment_tensor(atoms, coordinates - com))
                sort = np.argsort(Ia)
                Is.append((Ia[sort], Ivec[:, sort]))
            else:
                Is.append((np.empty((0, 0)), np.zeros((0, 0))))
        return Is

    def get_all_rotations(self):
        rotations = []
        rotation_k = []
        for k, I, atom_idx in zip(self.SEA_sizes, self.Is, self.SEA_atom_index):
            if k > 2:
                principals = I[0]
                vectors = I[1]
                print(principals, "\n", vectors)
            else:
                if k == 2:
                    principals = I[0]
                    vectors = I[1]
                    print(principals, "\n", vectors)
                    pass
                pass
