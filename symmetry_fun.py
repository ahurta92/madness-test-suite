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
    return np.array(num_line, dtype=np.float128).T


def atom_dict_to_coomatrix(atom_dict):
    atom_list = []
    coordinate_list = []
    for atom, coordinate_lines in atom_dict.items():
        for line in coordinate_lines:
            atom_list.append(atom)
            coordinate_list.append(line2array(line))

    coords = np.array(coordinate_list).T
    print('coords ', coords)
    print('coords ', coords.shape)
    return np.array(atom_list), coords


def create_distance_matrix(coo):
    num_atoms = coo.shape[1]
    d_matrix = np.zeros((num_atoms, num_atoms))

    for i in range(num_atoms):
        for j in range(i, num_atoms):
            d_matrix[i, j] = LA.norm(coo[:, i] - coo[:, j])
    d_matrix = d_matrix + d_matrix.T

    return d_matrix


def compute_com(atoms, coor):
    num_atoms = coor.shape[1]
    rx = 0
    ry = 0
    rz = 0
    M = 0
    for i in range(num_atoms):
        # Look up the mass
        mi = float(pt[pt['Symbol'] == atoms[i]]['AtomicMass'])
        M += mi
        rx += mi * coor[0, i]
        ry += mi * coor[1, i]
        rz += mi * coor[2, i]

    com = np.array([[rx, ry, rz]]).T
    return com / M


def compute_moment_tensor(atoms, coor):
    num_atoms = coor.shape[1]
    I = np.zeros((3, 3))

    for i in range(num_atoms):
        mi = float(pt[pt['Symbol'] == atoms[i]]['AtomicMass'])
        xi = coor[0, i]
        yi = coor[1, i]
        zi = coor[2, i]

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
        idx.append(np.where(uniqD[i] == Dcol)[0])
    return ks, idx


def selectPerp(n):
    # normalize
    print("n: ", n)
    print("norm: ", LA.norm(n))
    n = n / LA.norm(n)
    print("n: ", n)
    # define the unit vectors
    nx = np.array([1, 0, 0])
    ny = np.array([0, 1, 0])
    nz = np.array([0, 0, 1])
    unit = [nx, ny, nz]
    selected = None
    min_dot = 10000
    for ni in unit:
        dot = np.dot(ni, n)
        if dot == 0:
            return ni
        elif dot <= min_dot:
            selected = ni
            min_dot = dot
    nPerp = selected - np.dot(selected, n) * n
    return nPerp


def Rotation(n, theta):

    n1 = n[0]
    n2 = n[1]
    n3 = n[2]

    R = np.zeros((3, 3), dtype=np.float128)

    R[0, 0] = np.cos(theta) + n1 ** 2 * (1 - np.cos(theta))
    R[1, 1] = np.cos(theta) + n2 ** 2 * (1 - np.cos(theta))
    R[2, 2] = np.cos(theta) + n3 ** 2 * (1 - np.cos(theta))

    R[0, 1] = n1 * n2 * (1 - np.cos(theta)) - n3 * np.sin(theta)
    R[1, 0] = n1 * n2 * (1 - np.cos(theta)) + n3 * np.sin(theta)

    R[0, 2] = n1 * n3 * (1 - np.cos(theta)) + n2 * np.sin(theta)
    R[2, 0] = n1 * n3 * (1 - np.cos(theta)) - n2 * np.sin(theta)

    R[1, 2] = n2 * n3 * (1 - np.cos(theta)) - n1 * np.sin(theta)
    R[2, 1] = n3 * n2 * (1 - np.cos(theta)) + n1 * np.sin(theta)

    return R


class SymmetryData:

    def __init__(self, mol):

        self.rotor_types = None
        self.rotor_type = None
        self.SEA_atom_index = None
        self.SEA_sizes = None
        self.atom_dict = get_atom_dict(mol)
        self.atoms, self.coordinates = atom_dict_to_coomatrix(self.atom_dict)

        self.num_atoms = len(self.atoms)
        self.com = compute_com(self.atoms, self.coordinates)

        if LA.norm(self.com) > .00005:
            self.coordinates = self.coordinates - self.com
            self.com = np.array([[0, 0, 0]]).T

        self.I = compute_moment_tensor(self.atoms, self.coordinates)
        self.D = create_distance_matrix(self.coordinates)
        self.Is = self.get_all_moment_tensors()

    def get_all_moment_tensors(self):
        self.SEA_sizes, self.SEA_atom_index = getSeaInfo(self.D)

        Is = []
        for k, atom_idx in zip(self.SEA_sizes, self.SEA_atom_index):
            print(k)
            if k > 1:
                atoms = self.atoms[atom_idx]
                print(atoms)
                coordinates = self.coordinates[:, atom_idx]
                com = compute_com(atoms, coordinates)
                print("coordinates: ", coordinates, "\n com: ", com)
                Ia, Ivec = LA.eig(compute_moment_tensor(atoms, coordinates - com))
                sort = np.argsort(Ia)
                Is.append((Ia[sort], Ivec[:, sort]))
            else:
                Is.append((np.empty((0, 0)), np.zeros((0, 0))))
        return Is

    def get_all_rotations(self):
        rotations = []
        rotation_k = []
        for k, eigs, atom_idx in zip(self.SEA_sizes, self.Is, self.SEA_atom_index):
            if k > 2:
                I = eigs[0]
                vec = eigs[1]

                IA = I[0]
                IB = I[1]
                IC = I[2]

                if IA + IB == IC:
                    pass

            else:
                if k == 2:
                    principals = eigs[0]
                    vectors = I[1]
                    print(principals, "\n", vectors)
                    pass
                pass

    def check_rotation(self, R):
        num_atoms = self.num_atoms
        print("num atoms ", num_atoms)
        X = self.coordinates
        print("x shape", X.shape)
        Y = np.matmul(R, X)
        print("y shape", Y.shape)
        print("X: ", X)
        print("Y: ", X)
        for i in range(num_atoms):
            yi = Y[:, [i]]
            atom = self.atoms[i]
            atoms_idx = np.argwhere(atom == self.atoms)
            found = False
            for j in atoms_idx:
                xj = X[:, [j[0]]]
                if LA.norm(yi - xj) < .000006:
                    print("sub between yi and xj: ", LA.norm(yi - xj))
                    found = True
            if not found:
                return False
        return True

    def find_rotor_type(self):

        I, V = LA.eigh(self.I)

        IA = I[0]
        IB = I[1]
        IC = I[2]
        print(I)

        def is_equal(IA, IB):
            return np.abs(IA - IB) < .00005

        if IA == 0 and is_equal(IB, IC):
            self.rotor_type = "linear"
        if is_equal(IA, IB) and is_equal(IB, IC):
            self.rotor_type = "spherical"
        if is_equal(IA, IB) and not is_equal(IB, IC) or not is_equal(IA, IB) and is_equal(IB, IC):
            self.rotor_type = "symmetric"
        if not is_equal(IA, IB) and not is_equal(IB, IC) and not is_equal(IA, IC):
            self.rotor_type = "asymmetric"

    def compute_rotor_type(self, I):

        IA = I[0]
        IB = I[1]
        IC = I[2]
        print(I)

        def is_equal(IA, IB):
            return np.abs(IA - IB) < .00005

        if IA == 0 and is_equal(IB, IC):
            return "linear"
        if is_equal(IA, IB) and is_equal(IB, IC):
            return "spherical"
        if is_equal(IA, IB) and not is_equal(IB, IC) or not is_equal(IA, IB) and is_equal(IB, IC):
            return "symmetric"
        if not is_equal(IA, IB) and not is_equal(IB, IC) and not is_equal(IA, IC):
            return "asymmetric"

    def find_sea_rotor_type(self):
        self.rotor_types = []
        for k, atom_idx, II in zip(self.SEA_sizes, self.SEA_atom_index, self.Is):
            if k>1:

                I=II[0]
                print(I)
                self.rotor_types.append(self.compute_rotor_type(I))
            else:
                self.rotor_types.append("atom")

