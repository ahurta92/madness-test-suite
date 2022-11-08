import sys
from dalton import Dalton

mol = sys.argv[1]
xc = sys.argv[2]
op = sys.argv[3]
basis = sys.argv[4]
num_proc = sys.argv[5]

runner = Dalton()

runner.Np = num_proc

result = runner.get_polar_json(mol, xc, op, basis)

print(result[basis]['response']['frequencies'])
print(result[basis]['response']['values']['xx'])

