import sys
from daltonRunner import DaltonRunner

mol = sys.argv[1]
num_proc = sys.argv[2]

basis_list = ['aug-cc-pVDZ', 'aug-cc-pVTZ', 'aug-cc-pVQZ', 'aug-cc-pV5Z', 'aug-cc-pV6Z']
d_basis_list = ['d-aug-cc-pVDZ', 'd-aug-cc-pVTZ', 'd-aug-cc-pVQZ', 'd-aug-cc-pV5Z', 'd-aug-cc-pV6Z']

runner = DaltonRunner()

runner.Np = num_proc

for basis in basis_list + d_basis_list:
    try:
        result = runner.get_polar_json(mol, 'hf', 'dipole', basis)
        print(result[basis]['response']['frequencies'])
        print(result[basis]['response']['values']['xx'])
    except:
        pass
