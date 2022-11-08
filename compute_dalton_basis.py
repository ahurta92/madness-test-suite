import os

os.chdir("/gpfs/projects/rjh/adrian/post_watoc/august/")
from madnessReader import *
from dalton import Dalton

mol = sys.argv[1]
num_proc = sys.argv[2]

basis_list = ['aug-cc-pV5Z']  # , 'aug-cc-pV5Z', 'aug-cc-pV6Z']
d_basis_list = ['d-aug-cc-pV5Z']  # , 'd-aug-cc-pV5Z', 'd-aug-cc-pV6Z']

runner = Dalton()
runner.Np = num_proc

for basis in basis_list + d_basis_list:
    try:
        result = runner.get_polar_json(mol, 'hf', 'dipole', basis)
    except:
        pass
