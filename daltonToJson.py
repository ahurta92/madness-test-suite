import sys
import string
import itertools
import json
from collections import OrderedDict

from matplotlib.pyplot import polar


class daltonToJson:

    def __init__(self):
        self.calcSetup = {}
        self.calcRes = {}
        self.calcTask = {}
        self.simulationTime = {'cpuTime': 0.0,
                               'wallTime': 0.0, 'units': 'second'}
        self.calculations = []
        self.taskNumber = 0
        self.setupCount = 0
        self.subTask = False

        self.xx = 0
        self.yy = 0
        self.zz = 0

    def convert(self, streamIn):
        # Takes in the file and converts to JSON
        # This ordered dictionary allows me to read in one module at a time
        my_tasks = OrderedDict([
            ('a direct, restricted step, second order MCSCF program  ', self.readScfDft),
            ('Linear Response calculation', self.readResponse),
            ('Singlet electronic excitation energies', self.readExcited),
        ])
        collectingInput = False
        line = streamIn.readline()
        while line:
            for myKey in my_tasks.keys():
                if line.find(myKey) >= 0:
                    # print(myKey)
                    myIndex = list(my_tasks).index(myKey)
                    if myIndex >= 0:
                        self.calcTask = {}
                        self.calcSetup = {}
                        self.calcRes = {}
                        self.taskNumber += 1
                        self.setupCount += 1
                    my_tasks[myKey](line, streamIn)
            if line.find('Total CPU  time used in DALTON:') >= 0:
                ##print("Capturing TIME!")
                line2 = streamIn.readline()
                self.simulationTime = self.readTiming(line, line2)
                break
            line = streamIn.readline()

        return json.dumps({'simulation': {
            'calculations': self.calculations,
            'simulationTime': self.simulationTime}},
            indent=2, separators=(',', ': '), ensure_ascii=False)

    def readScfDft(self, line, streamIn):
        #print('READING SCF')
        self.calcSetup['numberOfElectrons'] = 0
        self.calcSetup['molecularSpinMultiplicity'] = 1

        # for each one of these functions I read in a line and capture the value.
        # Value gets places in correct dictionary
        def closedShell(line):
            self.calcSetup['numberOfElectrons'] = int(line.split()[6])

        def doCharge(line):
            self.calcSetup['charge'] = int(float(line.split()[6]))

        def waveFuncSCF(line):
            if line.split()[5] == 'HF':
                self.calcSetup['waveFunctionType'] = 'HF'
                self.calcSetup['waveFunctionTheory'] = 'Hartree-Fock'
            elif line.split()[5] == 'KS-DFT':
                self.calcSetup['waveFunctionType'] = 'KS-DFT'
                self.calcSetup['waveFunctionTheory'] = 'DFT'

        def totalEn(line):
            self.calcRes['totalEnergy'] = {'value': float(
                line.split(':')[1]), 'units': 'Hartree'}

        def totalELectronEn(line):
            self.calcRes['electronEnergy'] = {
                'value': float(line.split(':')[1]), 'units': 'Hartree'}

        def nucEn(line):
            self.calcRes['nuclearRepulsionEnergy'] = {'value': float(line.split(':')[1]),
                                                      'units': 'Hartree'}

        self.calcTask['calculationType'] = 'energyCalculation'
        # Here I can define what I use to match what I want to capture
        # if the line matches to one of these inputs we will take the line
        # and capture value and save in dictionary using above functions
        scfInp = {

            '@    Wave function type': waveFuncSCF,
            '@    Number of closed shell electrons': closedShell,
            '@    Total charge of the molecule': doCharge,
            '@    Final DFT energy': totalEn,
            '@    Final HF energy:': totalEn,
            '@    Nuclear repulsion:': nucEn,
            '@    Electronic energy:': totalELectronEn,
        }
        line = streamIn.readline()
        while line:
            for scfKey in scfInp.keys():
                if line.find(scfKey) >= 0:
                    # print(scfKey)
                    scfInp[scfKey](line)
            if line.find('CPU and wall time for SCF :') >= 0:
                # print(line)
                time = line.split(':')[1]
                time = time.split()
                self.calcTask['calculationTime'] = {'cpuTime': float(time[0]),
                                                    'wallTime': float(time[1]),
                                                    'units': 'second'}
            elif line.find('End of Wave Function Section') >= 0:
                # print(line)
                break
                # print(time)
            line = streamIn.readline()
        self.calcTask['calculationResults'] = self.calcRes
        self.calcTask['calculationSetup'] = self.calcSetup
        self.calculations.append(self.calcTask)

    def readResponse(self, line, streamIn
                     ):
        self.calcTask['calculationType'] = 'LinearResponse'
        #print("READING RESPONSE")
        
        self.xx = 0
        self.xy = 0
        self.xz = 0
        self.yx = 0
        self.yy = 0
        self.yz = 0
        self.zx = 0
        self.zy = 0
        self.zz = 0

        def frequencies(line):
            #print("READING FREQ")

            freq = line.split()[2:]
            num_freq = len(freq)
            self.calcSetup['frequencies'] = []
            self.calcSetup['numFrequencies'] = int(num_freq)
            # empty list of size 3
            self.calcRes['secondOrderProp'] = {}
            self.calcRes['secondOrderProp']['values'] = []

            self.polar_dict={}
            self.polar_dict["xx"]=[0]*int(num_freq)
            self.polar_dict["xy"]=[0]*int(num_freq)
            self.polar_dict["xz"]=[0]*int(num_freq)
            self.polar_dict["yx"]=[0]*int(num_freq)
            self.polar_dict["yy"]=[0]*int(num_freq)
            self.polar_dict["yz"]=[0]*int(num_freq)
            self.polar_dict["zx"]=[0]*int(num_freq)
            self.polar_dict["zy"]=[0]*int(num_freq)
            self.polar_dict["zz"]=[0]*int(num_freq)

            # need to get the values at frequency
            for f in freq:
                self.calcSetup['frequencies'].append(
                    float(f.replace('D', 'E')))

        def secondOrderProp(line):
            lineSP = line.split()
            opA = lineSP[2]
            opB = lineSP[4]

            opKey=' ; '.join([opA,opB])
            

            def grab_val(line):
                return float(line[7]) 

            def grab_operator(line):
                return ' '.join(line)[1:6]


            if opKey == "XDIPLEN ; XDIPLEN":
                self.polar_dict['xx'][self.xx]=grab_val(lineSP)
                self.xx+=1

            elif opKey == "XDIPLEN ; YDIPLEN":
                self.polar_dict['xy'][self.xy]=grab_val(lineSP)
                self.xy+=1

            elif opKey == "XDIPLEN ; ZDIPLEN":
                self.polar_dict['xz'][self.xz]=grab_val(lineSP)
                self.xz+=1
            elif opKey == "YDIPLEN ; XDIPLEN":
                self.polar_dict['yx'][self.yx]=grab_val(lineSP)
                self.yx+=1
            elif opKey == "YDIPLEN ; YDIPLEN":
                self.polar_dict['yy'][self.yy]=grab_val(lineSP)
                self.yy+=1
            elif opKey == "YDIPLEN ; ZDIPLEN":
                self.polar_dict['yz'][self.yz]=grab_val(lineSP)
                self.yz+=1
            elif opKey == "ZDIPLEN ; XDIPLEN":
                self.polar_dict['zx'][self.zx]=grab_val(lineSP)
                self.zx+=1
            elif opKey == "ZDIPLEN ; YDIPLEN":
                self.polar_dict['zy'][self.zy]=grab_val(lineSP)
                self.zy+=1
            elif opKey == "ZDIPLEN ; ZDIPLEN":
                self.polar_dict['zz'][self.zz]=grab_val(lineSP)
                self.zz+=1


        scfInp = OrderedDict([
            ('B-frequencies', frequencies),
            ('>> =', secondOrderProp,),])


        # if the line matches to one of these inputs we will take the line
        line = streamIn.readline()
        while line:
            # SCF converged
            if line.find('Total CPU  time used in RESPONSE:') >= 0:
                line2 = streamIn.readline()
                self.calcTask['calculationTime'] = self.readTaskTimes(
                    line, line2)
                break
            else:
                for scfKey in scfInp.keys():
                    if line.find(scfKey) >= 0:
                        scfInp[scfKey](line)
            line = streamIn.readline()
        self.calcTask['calculationResults'] = self.polar_dict
        self.calcTask['calculationSetup'] = self.calcSetup
        self.calculations.append(self.calcTask)

    def readExcited(self, line, streamIn):
        #print("READING RESPONSE")
        self.calcTask['calculationType'] = 'SingletExcitationEnergy'
        # if the line matches to one of these inputs we will take the line
        line = streamIn.readline()
        while line:
            # SCF converged
            if line.find('Total CPU  time used in ABACUS:') >= 0:
                line2 = streamIn.readline()
                self.calcTask['calculationTime'] = self.readTaskTimes(
                    line, line2)
                break
            # find the first line to start reading
            if line.find('==============') >= 0:
                self.calcRes['frequencies'] = []
                self.calcRes['Sym'] = []
                self.calcRes['Mode'] = []
                line = streamIn.readline()
                while line:
                    if line.find('==============') >= 0:
                        break
                    # find the first ---- line
                    if line.find('------------') >= 0:
                        self.calcRes['frequencies'].append([])
                        self.calcRes['Sym'].append([])
                        self.calcRes['Mode'].append([])
                        line = streamIn.readline()
                        while line:
                            if line.find('-----------') >= 0:
                                self.calcRes['frequencies'].append([])
                                self.calcRes['Sym'].append([])
                                self.calcRes['Mode'].append([])
                            elif line.find('========') == -1:
                                self.calcRes['frequencies'][-1].append(
                                    float(line.split()[2]))
                                self.calcRes['Sym'][-1].append(
                                    int(line.split()[0]))
                                self.calcRes['Mode'][-1].append(
                                    int(line.split()[1]))
                            else:
                                break
                            line = streamIn.readline()
                    if line.find('==============') >= 0:
                        break
                    line = streamIn.readline()
                    # read if not ------
            line = streamIn.readline()
        self.calcTask['calculationResults'] = self.calcRes
        self.calculations.append(self.calcTask)

    def readTiming(self, line1, line2):
        vars = line1.split(':')[1]
        vars = vars.split()[0]
        vars2 = line2.split(':')
        vars2 = vars.split()[0]

        return {'cpuTime': float(vars), 'wallTime': float(vars2),
                'units': 'second'}

    def readTaskTimes(self, line1, line2):
        vars = line1.split(':')[1]
        vars = vars.split()[0]
        vars2 = line2.split(':')
        vars2 = vars.split()[0]

        return {'cpuTime': float(vars), 'wallTime': float(vars2),
                'units': 'second'}
