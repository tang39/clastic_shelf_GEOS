from subprocess import call
import os
import numpy as np
import multiprocessing
import glob
from wrapper import hdf5_wrapper
import shutil


def read_value(region, k):
    v = region[k]['__values__']
    s = region[k]['__dimensions__']
    return np.reshape(v, s, order='C')


def collect_data(n):
    dirName = os.getcwd()
    for i in range(n):
        pressure = np.load(dirName + '/runs/Run' + str(i) + '/' + 'pressure_all.npy')
        saturation = np.load(dirName + '/runs/Run' + str(i) + '/' + 'saturation_all.npy')
        print(pressure.shape)
        if i == 0:
            pressure_all = pressure
            saturation_all = saturation
        else:
            pressure_all = np.append(pressure_all, pressure, axis=0)
            saturation_all = np.append(saturation_all, saturation, axis=0)
    return pressure_all, saturation_all


class GEOSX_Ensemble():
    """
    GEOSX_Ensemble class for generating geosx simulation ensemble

    Attributes:
        nx/ny/nz: reservoir domain dimension
        Nx/Ny/Nz: formation domain dimension being considered for boundary condition
        X/Y/Z : grids in x, y, z direction
        rocktype: rocktype for high perm facies (sand)
    """

    def __init__(self, X, Y, Z, Init_p, Init_z, geosx_dir, input_file, **kwargs):
        self.nx = 64
        self.ny = 64
        self.nz = 28
        self.boundary = 5
        self.Nx = self.nx + self.boundary * 2
        self.Ny = self.ny + self.boundary * 2
        self.Nz = self.nz
        self.X = X
        self.Y = Y
        self.Z = Z
        self.rocktype = 3
        # relative permeability curve parameters
        self.minvolfrac1 = [0, 0]
        self.minvolfrac2 = [0.605, 0.294]
        self.relpermexp1 = [5.01, 5.01]
        self.relpermexp2 = [6.36, 1.825]
        self.relpermmax1 = [0.015, 0.55]
        self.relpermmax2 = [1, 1]
        self.init_p = Init_p
        self.init_z = Init_z
        self.geosx_dir = geosx_dir
        self.input_file = input_file
        self.input_dir = './geosx_input'
        # get a list of all predefined values directly from __dict__
        allowed_keys = list(self.__dict__.keys())
        # Update __dict__ but only for keys that have been predefined
        # (silently ignore others)
        self.__dict__.update((key, value)
                             for key, value in kwargs.items() if key in allowed_keys)
        # to not silently ignore rejected keys
        rejected_keys = set(kwargs.keys()) - set(allowed_keys)
        if rejected_keys:
            raise ValueError(
                "Invalid arguments in constructor: {}".format(rejected_keys))

    def create_folder(self, n):
        dirName = os.getcwd()
        runNumber = n
        dirNames = [f'{self.input_dir}/geo_tables', f'{self.input_dir}/pvt_tables']
        for j in range(runNumber):
            os.makedirs('runs', exist_ok=True)
            os.makedirs('%s/runs/Run%d' % (dirName, j), exist_ok=True)
            call('cp -f ' + f'{self.input_dir}/{self.input_file}' + '.xml ' + dirName + '/runs/Run' + str(j), shell=True)
            for k in range(len(dirNames)):
                call('cp -r ' + dirNames[k] + ' ' + dirName + '/runs/Run' + str(j), shell=True)
        os.chdir(dirName)

    def generate_properties(self, permeability, porosity, rocktype):
        perm_mean = np.mean(permeability)
        poro_mean = np.mean(porosity)
        permeability_new = perm_mean * np.ones((self.Nx, self.Ny, self.Nz))
        porosity_new = poro_mean * np.ones((self.Nx, self.Ny, self.Nz))
        rocktype_new = self.rocktype * np.ones((self.Nx, self.Ny, self.Nz))
        permeability_new[self.boundary:(self.nx + self.boundary), self.boundary:(self.ny + self.boundary), :] = permeability
        porosity_new[self.boundary:(self.nx + self.boundary), self.boundary:(self.ny + self.boundary), :] = porosity
        rocktype_new[self.boundary:(self.nx + self.boundary), self.boundary:(self.ny + self.boundary), :] = rocktype
        minvolfrac_1 = self.minvolfrac1[0] * np.ones((self.Nx, self.Ny, self.Nz))
        minvolfrac_2 = self.minvolfrac2[0] * np.ones((self.Nx, self.Ny, self.Nz))
        relpermexp_1 = self.relpermexp1[0] * np.ones((self.Nx, self.Ny, self.Nz))
        relpermexp_2 = self.relpermexp2[0] * np.ones((self.Nx, self.Ny, self.Nz))
        relpermmax_1 = self.relpermmax1[0] * np.ones((self.Nx, self.Ny, self.Nz))
        relpermmax_2 = self.relpermmax2[0] * np.ones((self.Nx, self.Ny, self.Nz))
        minvolfrac_2[rocktype_new == self.rocktype] = self.minvolfrac2[1]
        relpermexp_2[rocktype_new == self.rocktype] = self.relpermexp2[1]
        relpermmax_1[rocktype_new == self.rocktype] = self.relpermmax1[1]

        os.makedirs('geo_tables', exist_ok='True')
        dirName = os.getcwd() + '/geo_tables'
        dataList = [permeability_new, porosity_new, permeability_new * 0.1, minvolfrac_1,
                    minvolfrac_2, relpermexp_1, relpermexp_2, relpermmax_1, relpermmax_2]
        datanames = ['Permeability', 'Porosity', 'Permeability_z', 'minvolfrac_1',
                     'minvolfrac_2', 'relpermexp_1', 'relpermexp_2', 'relpermmax_1', 'relpermmax_2']
        writepath = []
        for i in range(len(dataList)):
            writepath.append(dirName + '/' + str(datanames[i]) + '.Table')

        count = 0
        for path in writepath:
            print("writing " + str(count) + " GEOS initial condition tables")
            with open(writepath[count], 'w') as myfile:
                for z in range(0, self.Nz):
                    for col in range(0, self.Ny):
                        for row in range(0, self.Nx):
                            myfile.write(
                                str(dataList[count][row, col, self.Nz - z - 1]) + '\n')
                            # This is due to the difference between GEOSX and CMG,
                            # GEOSX takes upward direction as positive z direction, while CMG takes downward as positive.
            count += 1

        myfileP = open(dirName + '/x.Table', 'w')
        for i in range(0, len(self.X) - 1):
            myfileP.write(str((self.X[i] + self.X[i + 1]) / 2.) + '\n')
        myfileP.close()

        myfileT = open(dirName + '/y.Table', 'w')
        for j in range(0, len(self.Y) - 1):
            myfileT.write(str((self.Y[j] + self.Y[j + 1]) / 2.) + '\n')
        myfileT.close()

        myfileT = open(dirName + '/z.Table', 'w')
        Z = []
        for j in range(0, len(self.Z) - 1):
            Z.append((self.Z[j] + self.Z[j + 1]) / 2)
            myfileT.write(str((self.Z[j] + self.Z[j + 1]) / 2.) + '\n')
        myfileT.close()

        Z = np.array(Z)
        Init_p_interp = np.interp(Z, self.init_z, self.init_p)
        f = open(dirName + '/Init_Pres.Table', 'w')
        for i in range(0, self.Nz):
            f.write(str(Init_p_interp[i]) + '\n')
        f.close()
        print('Finish writing perm/poro table')

    def run_simulator(self, Run_i, num_per_run, permeability_all, porosity_all, facies_all):
        commandroot = self.geosx_dir + ' -i '
        dirName = os.getcwd()
        os.chdir(dirName + '/runs/Run' + str(Run_i))
        saturation_all = []
        pressure_all = []
        for j in range(0, num_per_run):
            rank = j + Run_i * num_per_run
            porosity = porosity_all[j]
            permeability = permeability_all[j]
            facies = facies_all[j]
            self.generate_properties(permeability, porosity, facies)
            commandinput = commandroot + self.input_file + '.xml >out_' + str(rank)
            returncode = call(commandinput, shell=True)
            print('Finish run' + str(rank) + ' ', returncode)
            # collect simulation data
            restart_root_pool = sorted(glob.glob(self.input_file + '_restart_*/'))
            target_region = 'Problem/domain/MeshBodies/mesh1/Level0/ElementRegions/elementRegionsGroup/Region1/elementSubRegions/cb1/'
            sat_key = 'phaseVolumeFraction'
            location_key = 'elementCenter'
            press_key = 'pressure'

            restart_root = restart_root_pool[-1]
            fname = glob.glob('%s/*hdf5' % (restart_root))[0]
            tmp_location = []
            for fname in sorted(glob.glob('%s/*hdf5' % (restart_root))):
                with hdf5_wrapper(fname) as data:
                    region = data[target_region]
                    tmp_location.append(read_value(region, location_key))
            location = np.concatenate(tmp_location, axis=0)

            ind_X = np.digitize(location[:, 0], self.X) - 1
            ind_Y = np.digitize(location[:, 1], self.Y) - 1
            ind_Z = np.digitize(location[:, 2], self.Z) - 1
            la = tuple([ind_X, ind_Y, ind_Z])
            saturation = []
            pressure = []
            for i in range(1, len(restart_root_pool) - 1):
                saturation_i = np.zeros((self.Nx, self.Ny, self.Nz))
                pressure_i = np.zeros((self.Nx, self.Ny, self.Nz))
                restart_root = restart_root_pool[i]
                tmp_sat = []
                tmp_press = []
                for fname in sorted(glob.glob('%s/*hdf5' % (restart_root))):
                    print(fname)
                    with hdf5_wrapper(fname) as data:
                        region = data[target_region]
                        tmp_sat.append(read_value(region, sat_key))
                        tmp_press.append(read_value(region, press_key))

                sat = np.concatenate(tmp_sat, axis=0)
                press = np.concatenate(tmp_press, axis=0)
                saturation_i[la] = sat[:, 0]
                pressure_i[la] = press
                saturation_i = saturation_i[self.boundary:(self.nx + self.boundary), self.boundary:(self.ny + self.boundary), :]
                pressure_i = pressure_i[self.boundary:(self.nx + self.boundary), self.boundary:(self.ny + self.boundary), :]
                pressure.append(pressure_i)
                saturation.append(saturation_i)

            pressure = np.array(pressure)
            saturation = np.array(saturation)
            pressure_all.append(pressure)
            saturation_all.append(saturation)

            # clean restart file
            for filename in glob.glob(f'{self.input_file}_restart_*'):
                if os.path.isfile(filename):
                    os.remove(filename)
                else:
                    shutil.rmtree(filename)

        pressure_all = np.array(pressure_all)
        saturation_all = np.array(saturation_all)

        np.save(dirName + '/runs/Run' + str(Run_i) + '/saturation_all.npy', saturation_all)
        np.save(dirName + '/runs/Run' + str(Run_i) + '/pressure_all.npy', pressure_all)

        return pressure_all, saturation_all

    def run_simulator_par_all(self, n, num_per_run, permeability_all, porosity_all, facies_all):
        num = np.arange(0, n, 1)
        processes = []
        for i in range(0, n):
            porosity = porosity_all[i * num_per_run:(i + 1) * num_per_run]
            permeability = permeability_all[i * num_per_run:(i + 1) * num_per_run]
            facies = facies_all[i * num_per_run:(i + 1) * num_per_run]

            p = multiprocessing.Process(target=self.run_simulator, args=(
                num[i], num_per_run, permeability, porosity, facies))
            os.system("taskset -p -c %d %d" %
                      (i % multiprocessing.cpu_count(), os.getpid()))
            p.start()
            processes.append(p)

        for process in processes:
            process.join()

        pressure_all, saturation_all = collect_data(n)

        return pressure_all, saturation_all

    def clean_folder(self, n):
        dirName = os.getcwd()
        for i in range(n):
            for filename in glob.glob(f'{dirName}/runs/Run{i}/{self.input_file}_restart_*'):
                if os.path.isfile(filename):
                    os.remove(filename)
                else:
                    shutil.rmtree(filename)
