import time
import numpy as np
import os
from geosxensemble import GEOSX_Ensemble

if __name__ == '__main__':

    # genearte simulation results for 10 years of CO2 injection.
    # Will run for ~280min on quartz.

    porosity_all = np.load('./geosx_input/porosity.npy')
    permeability_all = np.load('./geosx_input/permeability.npy')
    facies_all = np.load('./geosx_input/facies.npy')
    porosity_all = porosity_all.transpose((0, 3, 2, 1))
    permeability_all = permeability_all.transpose((0, 3, 2, 1))
    facies_all = facies_all.transpose((0, 3, 2, 1))

    # shape nr, nx, ny, nz
    start = time.perf_counter()
    NPar = 2
    num_per_run = 2
    N = NPar * num_per_run

    geosx_dir = '/usr/workspace/tang39/GEOSX/GEOSX/build-quartz-gcc@8.1.0-release/bin/geosx'
    input_file = 'large_four_CO2_wells_3d_64_28'
    input_dir = 'geosx_input'
    nx = 64
    ny = 64
    nz = 28
    X = np.concatenate((np.arange(-10000, 0, 2000), np.linspace(0, 32156.4, nx + 1), np.arange(34156.4, 42157, 2000)), axis=0)
    Y = X
    Z = np.linspace(0, 85.344, nz + 1)

    Init_HydroPressure = np.array([1800.9, 1805.4, 1809.8, 1814.3, 1818.8, 1823.2, 1827.7, 1832.1, 1836.6, 1841.1,
                                   1845.5, 1850, 1854.5, 1858.9, 1863.4, 1867.8, 1872.3, 1876.8, 1881.2, 1885.7, 1890.1,
                                   1894.6, 1899, 1903.5, 1908, 1912.4, 1916.9, 1921.3, 1925.8, 1930.3])
    psi2pa = 6894.76
    Init_p = Init_HydroPressure * psi2pa
    Init_p = np.flip(Init_p)
    Z_init = np.linspace(0, 91.44, 31)
    Init_z = [(Z_init[i] + Z_init[i + 1]) / 2 for i in range(0, len(Z_init) - 1)]

    ensemble = GEOSX_Ensemble(nx=nx, ny=ny, nz=nz, X=X, Y=Y, Z=Z, input_dir=input_dir,
                              Init_p=Init_p, Init_z=Init_z, geosx_dir=geosx_dir, input_file=input_file)
    ensemble.create_folder(NPar)
    pressure_all, saturation_all = ensemble.run_simulator_par_all(
        NPar, num_per_run, permeability_all[:N], porosity_all[:N], facies_all[:N])

    result_dir = 'geosx_simulation_results'
    os.makedirs(f'./{result_dir}', exist_ok=True)
    np.save(f'./{result_dir}/pressure_all.npy', pressure_all)
    np.save(f'./{result_dir}/resusaturation_all.npy', saturation_all)
    np.save(f'./{result_dir}/permeability_all.npy', permeability_all)
    np.save(f'./{result_dir}/porosity_all.npy', porosity_all)
    np.save(f'./{result_dir}/facies_all.npy', facies_all)

    finish = time.perf_counter()
    print(f'Finished in {round(finish-start,2)/60} (min)')

