import subprocess

import numpy as np
import shutil
from madnessToDaltony import madnessToDalton
import os
import pandas as pd
import json

from daltonToJson import daltonToJson


class Dalton:
    dalton_dir = None

    @classmethod
    def __init__(self, base_dir, run_new):
        self.run = run_new
        self.base_dir = base_dir  # what is my base directory?
        self.dalton_dir = os.path.join(self.base_dir, 'dalton')
        # here I can change PROOT to my directory of chocse
        if shutil.which("mpirun") != None:
            self.use_mpi = True
            self.Np = int(os.cpu_count() / 8)
        else:
            self.use_mpi = False
            self.Np = 1

        # where ever I run I can assume that the dalton directory will be one above cwd
        if not os.path.exists("dalton"):
            os.mkdir("dalton")
        with open(self.base_dir + "/molecules/frequency.json") as json_file:
            self.freq_json = json.loads(json_file.read())

        # with open(DALROOT + '/dalton-dipole.json') as json_file:
        #    self.dipole_json = json.loads(json_file.read())
        # with open(DALROOT + '/dalton-excited.json') as json_file:
        #    self.excited_json = json.loads(json_file.read())

    @staticmethod
    def __write_polar_input(self, madmol, xc, operator, basis):
        """writes the polar input to folder"""
        # DALTON INPUT
        molname = madmol.split(".")[0]
        dalton_inp = []
        dalton_inp.append("**DALTON INPUT")
        dalton_inp.append(".RUN RESPONSE")
        dalton_inp.append(".DIRECT")
        if basis.split("-")[-1] == "uc":
            dalton_inp.append("*MOLBAS ")
            dalton_inp.append(".UNCONT ")
        dalton_inp.append("**WAVE FUNCTIONS")
        # HF or DFT
        if xc == "hf":
            dalton_inp.append(".HF")
        else:
            dalton_inp.append(".DFT")
            dalton_inp.append(xc.capitalize())
        # RESPONSE
        dalton_inp.append("**RESPONSE")
        # LINEAR
        dalton_inp.append("*LINEAR")
        # looks like it's the only option for a response calculation
        if operator == "dipole":
            dalton_inp.append(".DIPLEN")
            freq = self.freq_json[molname][xc][operator]
            num_freq = len(freq)
            dalton_inp.append(".FREQUENCIES")
            dalton_inp.append(str(num_freq))

            freq_s = []
            for f in freq:
                freq_s.append(str(f))
            dalton_inp.append(" ".join(freq_s))

        dalton_inp.append("**END OF DALTON INPUT")
        dalton_inp = "\n".join(dalton_inp)
        run_dir = self.dalton_dir + "/" + xc + "/" + molname + "/" + operator
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        # Here I read the madness mol file from the molecules directory
        madmolfile = self.base_dir + "/molecules/" + madmol + ".mol"
        mad_to_dal = madnessToDalton(self.base_dir)
        if basis.split("-")[-1] == "uc":
            mol_input = mad_to_dal.madmol_to_dalmol(madmolfile, "-".join(basis.split("-")[:-1]))
        else:
            mol_input = mad_to_dal.madmol_to_dalmol(madmolfile, basis)
        dal_run_file = run_dir + "/freq.dal"
        with open(dal_run_file, "w") as file:  # Use file to refer to the file object
            file.write(dalton_inp)
        mol_file = run_dir + "/" + molname + "-" + basis.replace("*", "S") + ".mol"
        with open(mol_file, "w") as file:  # Use file to refer to the file object
            file.write(mol_input)
        molname = mol_file.split("/")[-1]
        dalname = dal_run_file.split("/")[-1]
        return run_dir, dalname, molname

    def __run_dalton(self, rdir, dfile, mfile):
        dalton = shutil.which('dalton')
        # Change to run directory
        os.chdir(rdir)
        # dalton [.dal] [.mol]
        if self.use_mpi:
            daltonCommand = (
                    dalton + " -N " + str(self.Np) + " " + dfile + " " + mfile
            )
            print(daltonCommand)
        else:
            daltonCommand = "dalton " + dfile + " " + mfile
            print(daltonCommand)
        process = subprocess.Popen(daltonCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        os.chdir(self.base_dir)
        print("Changed Directory to ", self.base_dir)
        return output, error

    @staticmethod
    def __write_excited_input(self, madmol, xc, basis, num_states):
        # Given a molecule, exchange correlation functional, basis an number of states
        # generates a dalton .dal file and writes it in corresponding directory
        # /dalton/[xc]/[madmol]/excited-state

        molname = madmol.split(".")[0]
        dalton_inp = []
        dalton_inp.append("**DALTON INPUT")
        dalton_inp.append(".RUN PROPERTIES")
        dalton_inp.append(".DIRECT")
        if basis.split("-")[-1] == "uc":
            dalton_inp.append("*MOLBAS ")
            dalton_inp.append(".UNCONT ")
        dalton_inp.append("**WAVE FUNCTIONS")
        if xc == "hf":
            dalton_inp.append(".HF")
        else:
            dalton_inp.append(".DFT")
            dalton_inp.append(xc.capitalize())
        dalton_inp.append("**PROPERTIES")
        dalton_inp.append(".EXCITA")
        dalton_inp.append("*EXCITA")
        dalton_inp.append(".NEXCIT")

        states = []
        for i in range(10):
            states.append(str(num_states))
        dalton_inp.append("   " + "  ".join(states))

        dalton_inp.append("**END OF DALTON INPUT")
        dalton_inp = "\n".join(dalton_inp)
        run_dir = self.dalton_dir + xc + "/" + molname + "/" + "excited-state"
        mad_to_dal = madnessToDalton(self.base_dir)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        madmolfile = self.base_dir + "/molecules/" + madmol + ".mol"
        if basis.split("-")[-1] == "uc":
            mol_input = mad_to_dal.madmol_to_dalmol(madmolfile, "-".join(basis.split("-")[:-1]))
        else:
            mol_input = mad_to_dal.madmol_to_dalmol(madmolfile, basis)
        dal_run_file = run_dir + "/excited.dal"
        with open(dal_run_file, "w") as file:  # Use file to refer to the file object
            file.write(dalton_inp)
        mol_file = run_dir + "/" + molname + "-" + basis.replace("*", "S") + ".mol"
        with open(mol_file, "w") as file:  # Use file to refer to the file object
            file.write(mol_input)
        molname = mol_file.split("/")[-1]
        dalname = dal_run_file.split("/")[-1]
        return run_dir, dalname, molname

    @staticmethod
    def __create_frequency_json(outfile, basis):

        rdata = {
            "xx": [],
            "xy": [],
            "xz": [],
            "yx": [],
            "yy": [],
            "yz": [],
            "zx": [],
            "zy": [],
            "zz": [],
        }
        r_dict = {}
        r_dict["frequencies"] = []
        e_data = json.loads(outfile)["simulation"]["calculations"][1]
        r_data = json.loads(outfile)["simulation"]["calculations"][2]

        dipole_data = json.loads(outfile)["simulation"]["calculations"][0]

        p_data = r_data["calculationResults"]
        f_data = r_data["calculationSetup"]["frequencies"]

        r_dict["frequencies"] = f_data
        r_dict["values"] = p_data
        r_dict["calculationTime"] = r_data["calculationTime"]

        return {basis: {"ground": e_data, "response": r_dict, "dipole": dipole_data}}

    @staticmethod
    def __create_excited_json(outfile, basis):
        # generate tables given name of output files and basis_list used to generate output files
        data = {"totalEnergy": [], "nuclearRepulsionEnergy": [], "electronEnergy": []}
        s_dict = {}
        e_data = json.loads(outfile)["simulation"]["calculations"][0]
        r_data = json.loads(outfile)["simulation"]["calculations"][1]
        for dkeys in data.keys():
            data[dkeys].append(float(e_data["calculationResults"][dkeys]["value"]))

        # sort the frequencies
        freq = r_data["calculationResults"]["frequencies"]
        freq = [f for l in freq for f in l]
        freq = np.array(freq)
        sort_f = np.argsort(freq)
        freq = freq[sort_f]

        # Sort by frequency
        Mode = r_data["calculationResults"]["Mode"]
        Mode = [m for mo in Mode for m in mo]
        Mode = np.array(Mode)
        Mode = Mode[sort_f]
        Sym = r_data["calculationResults"]["Sym"]
        Sym = [s for so in Sym for s in so]
        Sym = np.array(Sym)
        Sym = Sym[sort_f]

        s_dict["Sym"] = Sym.tolist()
        s_dict["Mode"] = Mode.tolist()
        s_dict["freq"] = freq.tolist()
        s_dict["calculationTime"] = r_data["calculationTime"]

        return {basis: {"ground": e_data, "response": s_dict}}

    def polar_json_exists(self, mol, xc, operator, basis):

        run_directory, dal_input, mol_input = self.__write_polar_input(self,
                                                                       mol, xc, operator, basis
                                                                       )
        outfile = "/freq_" + "-".join([mol, basis]) + ".out"
        outfile = run_directory + outfile
        try:
            with open(outfile, "r") as daltonOutput:
                dipole_j = self.get_polar_json(mol, xc, operator, basis)
                if dipole_j is None:
                    raise TypeError('polar json does not exist for ' + mol + ' ' + xc + ' ' + operator + ' ' + basis)
                return True
        except (FileNotFoundError, KeyError, IndexError, TypeError) as e:
            print(e)
            return False

    def get_polar_json(self, mol, xc, operator, basis):
        run_directory, dal_input, mol_input = self.__write_polar_input(self,
                                                                       mol, xc, operator, basis
                                                                       )
        outfile = "/freq_" + "-".join([mol, basis]) + ".out"
        outfile = run_directory + outfile
        data = None
        try:
            with open(outfile, "r") as daltonOutput:
                dj = daltonToJson()
                data = self.__create_frequency_json(dj.convert(daltonOutput), basis)
        except (FileNotFoundError, KeyError, IndexError) as e:
            print(e)
            if self.run:
                print("Try and run molecule ", mol)
                d_out, d_error = self.__run_dalton(run_directory, dal_input, mol_input)
                print("Finshed running  ", mol, " in ", run_directory)
                print(d_error)
                try:

                    with open(outfile, "r") as daltonOutput:
                        dj = daltonToJson()
                        data = self.__create_frequency_json(dj.convert(daltonOutput), basis)
                except (IndexError) as e:
                    print("most likely BASIS not found", d_out)
                    pass
            else:

                print("Did not find ", basis, " data for", mol, "and Dalton is not set to run")
                pass
        return data

    def get_excited_json(self, mol, xc, basis, run):
        """get the json output given mol xc and basis
        :param run:
        """
        num_states = self.freq_json[mol][xc]["excited-state"]
        run_directory, dal_input, mol_input = self.__write_excited_input(self,
                                                                         mol, xc, basis, num_states
                                                                         )
        # First look for the output file and try and convert it to a json
        outfile = "/excited_" + "-".join([mol, basis]) + ".out"
        outfile = run_directory + outfile
        data = None
        try:
            # open the output file
            with open(outfile, "r") as daltonOutput:
                dj = daltonToJson()
                data = self.__create_excited_json(dj.convert(daltonOutput), basis)
        except:

            print("did not find output file")
            if run:

                print("Try and run molecule ", mol)
                d_out, d_error = self.__run_dalton(run_directory, dal_input, mol_input)
                # print(d_out, d_error)
                with open(outfile, "r") as daltonOutput:
                    dj = daltonToJson()
                    data = self.__create_excited_json(dj.convert(daltonOutput), basis)
                pass
            else:
                print("Not trying to run dalton for ", mol)
                pass
        return data

    def get_excited_result(self, mol, xc, basis, run):

        excited_j = self.get_excited_json(mol, xc, basis, run)
        if excited_j is not None:

            time = excited_j[basis]["ground"]["calculationTime"]
            results = excited_j[basis]["ground"]["calculationResults"]
            gR = {}
            gR["basis"] = basis
            # results
            gkeys = ["totalEnergy", "nuclearRepulsionEnergy", "electronEnergy"]
            for g in gkeys:
                gR[g] = float(results[g]["value"])
            # timings
            tkeys = ["cpuTime", "wallTime"]

            for t in tkeys:
                gR["g" + t] = float(time[t])
            rtime = excited_j[basis]["response"]["calculationTime"]
            for t in tkeys:
                gR["r" + t] = float(rtime[t])

            # number of electrons
            skeys = ["numberOfElectrons"]
            setup = excited_j[basis]["ground"]["calculationSetup"]
            for s in skeys:
                gR[s] = setup[s]

            gSeries = pd.Series(gR)
            rresults = excited_j[basis]["response"]
            ekeys = ["Sym", "Mode", "freq"]

            rR = {}
            for e in ekeys:
                rR[e] = rresults[e]
            rDf = pd.DataFrame.from_dict(rR)

            return gSeries, rDf
        else:
            return None

    def get_frequency_result(self, mol, xc, operator, basis):

        dipole_j = self.get_polar_json(mol, xc, operator, basis)[basis]
        time = dipole_j["ground"]["calculationTime"]
        results = dipole_j["ground"]["calculationResults"]
        ground_dipole = dipole_j["dipole"]["calculationResults"]
        gR = {}
        gR["basis"] = basis
        gR["dipole"] = pd.Series(ground_dipole)
        # results
        gkeys = ["totalEnergy", "nuclearRepulsionEnergy", "electronEnergy"]
        for g in gkeys:
            gR[g] = float(results[g]["value"])
        # timings
        tkeys = ["cpuTime", "wallTime"]
        for t in tkeys:
            gR["g" + t] = float(time[t])
        rtime = dipole_j["response"]["calculationTime"]
        for t in tkeys:
            gR["r" + t] = float(rtime[t])
        # number of electrons
        skeys = ["numberOfElectrons"]
        setup = dipole_j["ground"]["calculationSetup"]
        for s in skeys:
            gR[s] = setup[s]
        # ground results
        gSeries = pd.Series(gR)

        # response results
        rresults = dipole_j["response"]
        rkeys = ["xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"]
        rdict = {}
        rdict["frequencies"] = rresults["frequencies"]

        for r in rkeys:
            rdict[r] = rresults["values"][r]
        rdf = pd.DataFrame(rdict)

        return gSeries, rdf
