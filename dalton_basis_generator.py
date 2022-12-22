import numpy as np


def orbital_type_line(f_type):
    return '$ ' + f_type.capitalize() + '-TYPE FUNCTION\n'


def basis_contraction_line(n):
    space = ' ' * 4
    return 'H' + space + str(n) + space + str(n) + '\n'


def create_orbital_functions(betas):
    num_functions = len(betas)
    eye = np.identity(num_functions)
    i = 0
    lines = [basis_contraction_line(num_functions)]
    for beta in betas:
        b = '{:>15.8f}'.format(beta) + '   '
        coeffs = eye[i, :]
        # if num_functions > 6:
        if False:
            c1 = coeffs[:6]
            c2 = coeffs[6:]
            c1_line = np.array2string(c1, precision=2, separator='  ',
                                      formatter={'float_kind': lambda x: "%.3f" % x},
                                      suppress_small=True)[1:-1]
            c2_line = np.array2string(c2, precision=2, separator='  ',
                                      formatter={'float_kind': lambda x: "%.3f" % x},
                                      suppress_small=True)[1:-1]
            b_line = b + c1_line
            lines.append(b_line + ' ')
            lines.append('   ' + c2_line)
        else:
            c1 = coeffs
            c1_line = np.array2string(c1, precision=2, separator='    ',
                                      formatter={'float_kind': lambda x: "%.8f" % x},
                                      floatmode='fixed',
                                      max_line_width=1000,
                                      suppress_small=True)[1:-1]
            b_line = b + c1_line
            lines.append('  ' + b_line + '\n')
        i += 1

    return lines


class EvenTemperedBasis:
    def __init__(self, basis_dir):
        self.elements_supported = {'H': 1, 'He': 2, 'Be': 4, 'B': 5, 'C': 6, 'O': 8, }
        self.basis_name = 'EVT-'
        self.basis_dir = basis_dir
        self.element_dict = {'He': 'HELIUM',
                             'H': 'HYDROGEN',
                             'Be': 'Be',
                             'C': 'CARBON', 'O': 'OXYGEN'}

    def write_basis(self, suffix, elements_dictionary):
        b_write = self.basis_dir + '/' + self.basis_name + suffix
        with open(b_write, 'w') as basis_file:
            basis_file.write('$Basis = ' + self.basis_name + suffix + '\n')
            basis_file.write('$\n')
            basis_file.write(
                '$****************************************************************************************\n')
            basis_file.write('$ ELEMENTS supported \n')
            basis_file.write('$ ' + ' '.join(list(self.elements_supported.keys())) + ' $\n')
            for element, orbital_dict in elements_dictionary.items():
                basis_file.write('a ' + '{}\n'.format(self.elements_supported[element]))
                basis_file.write('$ ' + '{}\n'.format(self.element_dict[element]))
                for orbital, orb_betas in orbital_dict.items():
                    basis_file.write(orbital_type_line(orbital))
                    blines = create_orbital_functions(orb_betas)
                    basis_file.writelines(blines)
                    # for line in blines:
                    #    basis_file.write("%s\n" % line)

            basis_file.write('\n')
            basis_file.write('$ END OF BASIS')
