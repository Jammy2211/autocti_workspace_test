import lacosmic

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from os import path

import autocti as ac
import autocti.plot as aplt

"""
__Dataset Paths__

The 'dataset_type' describes the type of data being simulated (in this case, imaging data). They define the folder 
the dataset is output to on your hard-disk:

 - The image will be output to '/autocti_workspace/dataset/dataset_type/image.fits'.
 - The noise-map will be output to '/autocti_workspace/dataset/dataset_type/noise_map.fits'.
 - The pre_cti_data will be output to '/autocti_workspace/dataset/dataset_type/pre_cti_data.fits'.
"""
dataset_name = "no_ci_or_cti"
# dataset_name = "no_ci_or_cti__no_noise"

method = "lax"
method = "thresh"
cr_threshold = 4.0

"""
Returns the path where the dataset will be output, which in this case is
'/autocti_workspace/dataset/imaging_ci/uniform_cosmic_rays/parallel_x1'
"""
dataset_path = path.join("imaging_ci", "cosmics", "dataset", dataset_name)

"""
__Layout__

The 2D shape of the image.
"""
shape_native = (2086, 2128)

"""
The locations (using NumPy array indexes) of the parallel overscan, serial prescan and serial overscan on the image.
"""
parallel_overscan = ac.Region2D((2066, 2086, 51, shape_native[1] - 29))
serial_prescan = ac.Region2D((0, 2086, 0, 51))
serial_overscan = ac.Region2D((0, 2066, shape_native[1] - 29, shape_native[1]))

"""
The normalization of every charge injection image, which determines how many images are simulated.
"""
norm = 1

"""
Create the layout of the charge injection pattern for every charge injection normalization.
"""
layout = ac.Layout2DCI(
    shape_2d=shape_native,
    region_list=[(0, 1, 0, 1)],
    parallel_overscan=parallel_overscan,
    serial_prescan=serial_prescan,
    serial_overscan=serial_overscan,
)

"""
We can now load every image, noise-map and pre-CTI charge injection image as instances of the `ImagingCI` object.
"""
imaging_ci = ac.ImagingCI.from_fits(
    image_path=path.join(dataset_path, f"norm_{int(norm)}", f"data.fits"),
    noise_map_path=path.join(dataset_path, f"norm_{int(norm)}", f"noise_map.fits"),
    pre_cti_data_path=path.join(
        dataset_path, f"norm_{int(norm)}", f"pre_cti_data.fits"
    ),
    cosmic_ray_map_path=path.join(
        dataset_path, f"norm_{int(norm)}", f"cosmic_ray_map.fits"
    ),
    pixel_scales=0.1,
    layout=layout,
)

"""
__LACosmic Cosmic Ray Flagging__

Use the LACosmic algorithm to flag cosmic rays in the data.
"""
if method == "thresh":

    cr_flag_mask = imaging_ci.data.native > cr_threshold * imaging_ci.noise_map.native

elif method == "lax":

    import lacosmicx as lax

    pad = 2

    y_pixels = imaging_ci.data.shape_native[0]
    x_pixels = imaging_ci.data.shape_native[1]

    data = imaging_ci.data.resized_from(
        new_shape=(y_pixels + (2 * pad), x_pixels + (2 * pad))
    )

    cr_flag_mask, clean_data = lax.lacosmicx(
        indat=np.asarray(data.native).astype("float32"),
        readnoise=4.0,
        sigclip=4.0,
        sigfrac=0.1,
        niter=8,
        verbose=True,
    )

    clean_data = ac.Array2D.no_mask(
        values=clean_data, pixel_scales=imaging_ci.pixel_scales
    ).native
    clean_data = clean_data.resized_from(new_shape=(y_pixels, x_pixels))

    cr_flag_mask = ac.Array2D.no_mask(
        values=cr_flag_mask, pixel_scales=imaging_ci.pixel_scales
    ).native
    cr_flag_mask = cr_flag_mask.resized_from(new_shape=(y_pixels, x_pixels))

cr_flag_mask = ac.Array2D.no_mask(
    values=cr_flag_mask, pixel_scales=imaging_ci.pixel_scales
).native
cr_flag_mask = np.asarray(cr_flag_mask).astype("bool")

figsize = (50, 40)
norm = colors.Normalize(vmin=0.0, vmax=norm)

output_path = path.join(
    "imaging_ci", "cosmics", "output", method, "data_1_no_ci_or_cti"
)

mat_plot_2d = aplt.MatPlot2D(
    cmap=aplt.Cmap(vmax=10.0, vmin=0.0),
    output=aplt.Output(path=output_path, filename="data", format="png"),
)

array_2d_plotter = aplt.Array2DPlotter(
    array=imaging_ci.data.native, mat_plot_2d=mat_plot_2d
)
array_2d_plotter.figure_2d()


mat_plot_2d = aplt.MatPlot2D(
    output=aplt.Output(path=output_path, filename="cr_mask", format="png")
)

cr_flag_mask_plot = ac.Array2D.no_mask(
    values=cr_flag_mask,
    pixel_scales=imaging_ci.pixel_scales,
).native
array_2d_plotter = aplt.Array2DPlotter(array=cr_flag_mask_plot, mat_plot_2d=mat_plot_2d)
array_2d_plotter.figure_2d()


cr_mask_true = imaging_ci.cosmic_ray_map.native > 0.0

mat_plot_2d = aplt.MatPlot2D(
    output=aplt.Output(path=output_path, filename="cr_mask_true", format="png")
)

array_2d_plotter = aplt.Array2DPlotter(array=cr_mask_true, mat_plot_2d=mat_plot_2d)
array_2d_plotter.figure_2d()

total_cr_true = np.sum(cr_mask_true)
total_cr_flagged = np.sum(cr_flag_mask)

cr_flagged_correctly_map = cr_mask_true * cr_flag_mask
total_cr_flagged_correctly = np.sum(cr_flagged_correctly_map)

cr_flagged_incorrectly_map = np.logical_and(cr_flag_mask, np.invert(cr_mask_true))
total_cr_flagged_incorrectly = np.sum(cr_flagged_incorrectly_map)

cr_unflagged_map = np.logical_and(np.invert(cr_flag_mask), cr_mask_true)
total_cr_unflagged = np.sum(cr_unflagged_map)


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
max_unflagged_signal_to_noise = np.max(
    imaging_ci.absolute_signal_to_noise_map.native[cr_unflagged_map == True]
)
print(f"Max Unflagged S/N = {max_unflagged_signal_to_noise}")

print()
