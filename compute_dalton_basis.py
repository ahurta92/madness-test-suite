import os

os.chdir("/gpfs/projects/rjh/adrian/post_watoc/august/")
from madnessReader import *
from daltonRunner import DaltonRunner

mol = sys.argv[1]
num_proc = sys.argv[2]

basis_list = ['aug-cc-pCVDZ', 'aug-cc-pCVTZ', 'aug-cc-pCVQZ', 'aug-cc-pV5Z']  # , 'aug-cc-pV5Z', 'aug-cc-pV6Z']
d_basis_list = ['d-aug-cc-pCVDZ', 'd-aug-cc-pCVTZ', 'd-aug-cc-pCVQZ',
                'd-aug-cc-pV5Z']  # , 'd-aug-cc-pV5Z', 'd-aug-cc-pV6Z']

runner = DaltonRunner()

runner.Np = num_proc

for basis in basis_list + d_basis_list:
    try:
        result = runner.get_polar_json(mol, 'hf', 'dipole', basis)
    except:
        pass
