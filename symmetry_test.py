import os
import pandas as pd
import numpy as np
from numpy import linalg as LA
from numpy import matmul as matmul
import json

from symmetry_fun import SymmetryData
from symmetry_fun import *

allene = SymmetryData('allene')
h2o = SymmetryData('H2O')
NH3 = SymmetryData('NH3ex')
benzene = SymmetryData('C6H6')
ccl4 = SymmetryData('CCl4')
nh2cl = SymmetryData('NH2Cl')

nz = np.array([[0, 0, 1]]).T
print("nz: ", nz)
R = Rotation(nz, 2 * np.pi / 3)
print(benzene.check_rotation(R))

benzene.find_rotor_type()
print(benzene.rotor_type)

benzene.find_sea_rotor_type()
print(benzene.rotor_types)
