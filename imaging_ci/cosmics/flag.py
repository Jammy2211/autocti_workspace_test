import lacosmic

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from os import path

import autocti as ac

"""
__Path__

Grab the relative path to this file for loading the charge injection imaging which includes cosmic rays.
"""
dir_path = path.dirname(path.realpath(__file__))

"""
__Dataset Paths__

The 'dataset_type' describes the type of data being simulated (in this case, imaging data). They define the folder 
the dataset is output to on your hard-disk:

 - The image will be output to '/autocti_workspace/dataset/dataset_type/image.fits'.
 - The noise-map will be output to '/autocti_workspace/dataset/dataset_type/noise_map.fits'.
 - The pre_cti_data will be output to '/autocti_workspace/dataset/dataset_type/pre_cti_data.fits'.
"""
# dataset_type = "imaging_noise"
# dataset_type = "imaging_ci"
dataset_type = "imaging_ci_non_uniform"

"""
Returns the path where the dataset will be output, which in this case is
'/autocti_workspace/dataset/imaging_ci/uniform_cosmic_rays/parallel_x1'
"""
dataset_path = path.join(dir_path, "dataset", dataset_type)

"""
__Layout__

The 2D shape of the image.
"""
shape_native = (500, 500)

"""
The locations (using NumPy array indexes) of the parallel overscan, serial prescan and serial overscan on the image.
"""
parallel_overscan = ac.Region2D((2066, 2086, 51, shape_native[1] - 29))
serial_prescan = ac.Region2D((0, 2086, 0, 51))
serial_overscan = ac.Region2D((0, 2066, shape_native[1] - 29, shape_native[1]))

"""
Specify the charge injection regions on the CCD, which in this case is 5 equally spaced rectangular blocks.
"""
regions_list = [
    #   (0, 200, serial_prescan[3], serial_overscan[2]),
    #   (400, 600, serial_prescan[3], serial_overscan[2]),
    #   (800, 1000, serial_prescan[3], serial_overscan[2]),
    #   (1200, 1400, serial_prescan[3], serial_overscan[2]),
    #   (1600, 1800, serial_prescan[3], serial_overscan[2]),
]

"""
The normalization of every charge injection image, which determines how many images are simulated.
"""
norm_list = [100, 5000, 25000, 200000]

"""
Create the layout of the charge injection pattern for every charge injection normalization.
"""
layout_list = [
    ac.Layout2DCI(
        shape_2d=shape_native,
        region_list=regions_list,
        norm=norm,
        parallel_overscan=parallel_overscan,
        serial_prescan=serial_prescan,
        serial_overscan=serial_overscan,
    )
    for norm in norm_list
]

"""
We can now load every image, noise-map and pre-CTI charge injection image as instances of the `ImagingCI` object.
"""
imaging_ci_list = [
    ac.ImagingCI.from_fits(
        image_path=path.join(dataset_path, f"image_{int(layout.norm)}.fits"),
        noise_map_path=path.join(dataset_path, f"noise_map_{int(layout.norm)}.fits"),
        pre_cti_data_path=path.join(
            dataset_path, f"pre_cti_data_{int(layout.norm)}.fits"
        ),
        cosmic_ray_map_path=path.join(
            dataset_path, f"cosmic_ray_map_{int(layout.norm)}.fits"
        ),
        layout=layout,
        pixel_scales=0.1,
    )
    for layout in layout_list
]

"""
__LACosmic Cosmic Ray Flagging__

Use the LACosmic algorithm to flag cosmic rays in the data.
"""
for imaging_ci in imaging_ci_list:

    clean_data, cr_flag_mask = lacosmic.lacosmic(
        data=imaging_ci.data.native,
        contrast=1.0,
        cr_threshold=4.0,
        neighbor_threshold=4.0,
        error=imaging_ci.noise_map.native,
        maxiter=8,
    )

    figsize = (50, 40)
    norm = colors.Normalize(vmin=0.0, vmax=norm)

    plt.figure(figsize=figsize)
    plt.imshow(imaging_ci.data.native, norm=norm)
    plt.colorbar()
    plt.show()

    plt.figure(figsize=figsize)
    plt.imshow(clean_data, norm=norm)
    plt.colorbar()
    plt.show()

    plt.figure(figsize=figsize)
    plt.imshow(cr_flag_mask)
    plt.colorbar()
    plt.show()

    cr_mask_true = imaging_ci.cosmic_ray_map.native > 0.0

    plt.figure(figsize=figsize)
    plt.imshow(cr_mask_true)
    plt.colorbar()
    plt.show()

    total_cr_true = np.sum(cr_mask_true)
    total_cr_flagged = np.sum(cr_flag_mask)

    cr_flagged_correctly_map = cr_mask_true * cr_flag_mask
    total_cr_flagged_correctly = np.sum(cr_flagged_correctly_map)

    cr_flagged_incorrectly_map = np.logical_and(cr_flag_mask, np.invert(cr_mask_true))
    total_cr_flagged_incorrectly = np.sum(cr_flagged_incorrectly_map)

    cr_unflagged_map = np.logical_and(np.invert(cr_flag_mask), cr_mask_true)
    total_cr_unflagged = np.sum(cr_unflagged_map)

    max_unflagged_signal_to_noise = np.max(
        imaging_ci.absolute_signal_to_noise_map.native[cr_unflagged_map == True]
    )

    print(f"NORMALIZATION {norm}\n")

    print(f"CR True  = {total_cr_true}")
    print(f"CR Flagged = {total_cr_flagged}")

    print("\nFlagging Values: \n")

    print(f"CR Correct Flags = {total_cr_flagged_correctly}")
    print(f"CR Incorrect Flags = {total_cr_flagged_incorrectly}")
    print(f"CR Unflags = {total_cr_unflagged}")

    print("\nFlagging Percentages: \n")

    print(f"CR Correct Flags = {100.0*total_cr_flagged_correctly / total_cr_true}")
    print(f"CR Incorrect Flags = {100.0*total_cr_flagged_incorrectly / total_cr_true}")
    print(f"CR Unflags = {100.0*total_cr_unflagged / total_cr_true}")

    print("\n Other Stats: \n")

    print(f"Max Unflagged S/N = {max_unflagged_signal_to_noise}")

    print()
