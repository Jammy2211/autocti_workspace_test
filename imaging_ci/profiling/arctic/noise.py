import numpy as np
from astropy.io import fits
from os import path
import time

from arcticpy.src import cti, traps, ccd, roe


def numpy_array_2d_via_fits_from(
    file_path: str, hdu: int, do_not_scale_image_data: bool = False
):
    """
    Read a 2D NumPy array from a .fits file.

    After loading the NumPy array, the array is flipped upside-down using np.flipud. This is so that the structures
    appear the same orientation as .fits files loaded in DS9.

    Parameters
    ----------
    file_path
        The full path of the file that is loaded, including the file name and ``.fits`` extension.
    hdu
        The HDU extension of the array that is loaded from the .fits file.
    do_not_scale_image_data
        If True, the .fits file is not rescaled automatically based on the .fits header info.

    Returns
    -------
    ndarray
        The NumPy array that is loaded from the .fits file.

    Examples
    --------
    array_2d = numpy_array_2d_via_fits_from(file_path='/path/to/file/filename.fits', hdu=0)
    """
    hdu_list = fits.open(file_path, do_not_scale_image_data=do_not_scale_image_data)

    return np.array(hdu_list[hdu].data).astype("float64")


norm = 10000
total_columns = 10

dataset_type = "imaging_ci"
dataset_label = "uniform"
dataset_name = "parallel_x1"
dataset_size = f"columns_{total_columns}"

dataset_path = path.join(
    "dataset", dataset_type, dataset_label, dataset_name, dataset_size
)

parallel_trap_0 = traps.TrapInstantCapture(density=0.07275, release_timescale=0.8)

parallel_trap_list = [parallel_trap_0]

parallel_ccd = ccd.CCD(
    well_fill_power=0.58, well_notch_depth=0.0, full_well_depth=200000.0
)


image_pre_cti = numpy_array_2d_via_fits_from(
    file_path=path.join(dataset_path, f"norm_{int(norm)}", "image.fits"), hdu=0
)

start = time.time()

cti.add_cti(
    image=image_pre_cti,
    parallel_traps=parallel_trap_list,
    parallel_ccd=parallel_ccd,
    parallel_roe=roe.ROEChargeInjection(),
    parallel_express=2,
    parallel_prune_n_electrons=1e-18,
    parallel_prune_frequency=20,
)


print(f"Clocking Time = {(time.time() - start)}")


image_pre_cti = numpy_array_2d_via_fits_from(
    file_path=path.join(dataset_path, f"image_{int(norm)}_no_noise.fits"), hdu=0
)

start = time.time()

cti.add_cti(
    image=image_pre_cti,
    parallel_traps=parallel_trap_list,
    parallel_ccd=parallel_ccd,
    parallel_roe=roe.ROEChargeInjection(),
    parallel_express=2,
    parallel_prune_n_electrons=1e-18,
    parallel_prune_frequency=20,
)

print(f"Clocking Time No Noise = {(time.time() - start)}")
