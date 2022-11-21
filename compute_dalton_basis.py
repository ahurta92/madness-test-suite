import os
import sys
from madnessReader import *
from dalton import Dalton

mol = sys.argv[1]
num_proc = sys.argv[2]

#basis_list = ['aug-cc-pV5Z-uc','aug-cc-pV6Z']  # , 'aug-cc-pV5Z', 'aug-cc-pV6Z']
#d_basis_list = ['d-aug-cc-pV5Z-uc','d-aug-cc-pV6Z']  # , 'd-aug-cc-pV5Z', 'd-aug-cc-pV6Z']
basis_list = ['q-aug-cc-pVDZ','q-aug-cc-pVTZ','q-aug-cc-pVQZ','q-aug-cc-pVD5',]  # , 'd-aug-cc-pV5Z', 'd-aug-cc-pV6Z']
base_dir='/gpfs/projects/rjh/adrian/post_watoc/august'

runner = Dalton(base_dir,True)
runner.Np = num_proc

for basis in basis_list:
    try:
        result = runner.get_polar_json(mol, 'hf', 'dipole', basis)
    except:
        pass
