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
dataset_name = "cti_no_ci"
norm = 1

dataset_name = "with_ci"
norm = 100000

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

parallel_overscan = ac.Region2D((2066, 2086, 51, shape_native[1] - 29))
serial_prescan = ac.Region2D((0, 2086, 0, 51))
serial_overscan = ac.Region2D((0, 2066, shape_native[1] - 29, shape_native[1]))

"""
The normalization of every charge injection image, which determines how many images are simulated.
"""
regions_2d_list = [
    (16, 436, serial_prescan[3], serial_overscan[2]),
    (536, 956, serial_prescan[3], serial_overscan[2]),
    (1056, 1476, serial_prescan[3], serial_overscan[2]),
    (1576, 1996, serial_prescan[3], serial_overscan[2]),
]

"""
Create the layout of the charge injection pattern for every charge injection normalization.
"""
layout = ac.Layout2DCI(
    shape_2d=shape_native,
    region_list=regions_2d_list,
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
__Correct Serial CTI (So charge injection FPRs are unmixed)__
"""
clocker_2d = ac.Clocker2D(
    parallel_express=5,
    parallel_roe=ac.ROEChargeInjection(),
    serial_express=5,
    serial_prune_n_electrons=1e-7,
    serial_prune_frequency=10,
    iterations=1,
)

serial_trap_0 = ac.TrapInstantCapture(density=0.07275, release_timescale=0.8)
serial_trap_1 = ac.TrapInstantCapture(density=0.21825, release_timescale=4.0)
serial_trap_2 = ac.TrapInstantCapture(density=6.54804, release_timescale=20.0)

serial_trap_list = [serial_trap_0, serial_trap_1, serial_trap_2]

serial_ccd = ac.CCDPhase(
    well_fill_power=0.58, well_notch_depth=0.0, full_well_depth=200000.0
)

cti_2d = ac.CTI2D(
    serial_trap_list=serial_trap_list,
    serial_ccd=serial_ccd,
)

image_corrected = clocker_2d.remove_cti(data=imaging_ci.data, cti=cti_2d)

"""
__Charge Injection Estimate__
"""
injection_norm_list = layout.extract.parallel_fpr.median_list_from(
    array=image_corrected, pixels=(400, 420)
)

pre_cti_data = layout.pre_cti_data_non_uniform_from(
    injection_norm_list=injection_norm_list,
    pixel_scales=imaging_ci.image.pixel_scales,
)

"""
__Charge Injection Add CTI (So EPER / FPR subtract correctly from data for flagging).
"""
parallel_trap_0 = ac.TrapInstantCapture(density=0.214, release_timescale=1.25)
parallel_trap_1 = ac.TrapInstantCapture(density=0.412, release_timescale=4.4)
parallel_trap_list = [parallel_trap_0, parallel_trap_1]

parallel_ccd = ac.CCDPhase(
    well_fill_power=0.58, well_notch_depth=0.0, full_well_depth=200000.0
)

cti_2d = ac.CTI2D(
    parallel_trap_list=parallel_trap_list,
    parallel_ccd=parallel_ccd,
    serial_trap_list=serial_trap_list,
    serial_ccd=serial_ccd,
)

pre_cti_data_with_cti = clocker_2d.add_cti(data=pre_cti_data, cti=cti_2d)

"""
__Charge Injection Subtract__
"""
image_ci_subtracted = imaging_ci.image.native - pre_cti_data_with_cti.native

"""
__Cosmic Ray Flagging__
"""

cosmic_ray_mask = (
    image_ci_subtracted.native > cr_threshold * imaging_ci.noise_map.native
)

figsize = (50, 40)
norm = colors.Normalize(vmin=0.0, vmax=norm)

output_path = path.join("imaging_ci", "cosmics", "output", dataset_name)

"""
__Data__
"""
mat_plot_2d = aplt.MatPlot2D(
    cmap=aplt.Cmap(vmax=10.0, vmin=0.0),
    output=aplt.Output(path=output_path, filename="data", format="png"),
)

array_2d_plotter = aplt.Array2DPlotter(
    array=imaging_ci.data.native, mat_plot_2d=mat_plot_2d
)
array_2d_plotter.figure_2d()

"""
__Data (CI Subtracted)__
"""
mat_plot_2d = aplt.MatPlot2D(
    cmap=aplt.Cmap(vmax=10.0, vmin=0.0),
    output=aplt.Output(path=output_path, filename="data_ci_subtracted", format="png"),
)

array_2d_plotter = aplt.Array2DPlotter(
    array=image_ci_subtracted, mat_plot_2d=mat_plot_2d
)
array_2d_plotter.figure_2d()

"""
__Cosmic Ray Mask (True)_
"""
cosmic_ray_mask_true = imaging_ci.cosmic_ray_map.native > 0.0

mat_plot_2d = aplt.MatPlot2D(
    output=aplt.Output(path=output_path, filename="cosmic_ray_mask_true", format="png")
)

array_2d_plotter = aplt.Array2DPlotter(
    array=cosmic_ray_mask_true, mat_plot_2d=mat_plot_2d
)
array_2d_plotter.figure_2d()

"""
__Cosmic Ray Mask_
"""
mat_plot_2d = aplt.MatPlot2D(
    output=aplt.Output(path=output_path, filename="cosmic_ray_mask", format="png")
)

array_2d_plotter = aplt.Array2DPlotter(
    array=ac.Array2D.no_mask(
        values=cosmic_ray_mask,
        pixel_scales=imaging_ci.pixel_scales,
    ).native,
    mat_plot_2d=mat_plot_2d,
)
array_2d_plotter.figure_2d()

"""
__Cosmic Ray Mask (Pad for EPERs)__
"""
cosmic_ray_parallel_buffer = 5

cosmic_ray_mask = ac.Mask2D.from_cosmic_ray_map_buffed(
    cosmic_ray_map=cosmic_ray_mask,
    settings=ac.SettingsMask2D(cosmic_ray_parallel_buffer=cosmic_ray_parallel_buffer),
)

"""
__Statistics__
"""
total_cr_true = np.sum(cosmic_ray_mask_true)
total_cr_flagged = np.sum(cosmic_ray_mask)

cr_flagged_correctly_map = cosmic_ray_mask_true * cosmic_ray_mask
total_cr_flagged_correctly = np.sum(cr_flagged_correctly_map)

cr_flagged_incorrectly_map = np.logical_and(
    cosmic_ray_mask, np.invert(cosmic_ray_mask_true)
)
total_cr_flagged_incorrectly = np.sum(cr_flagged_incorrectly_map)

cr_unflagged_map = np.logical_and(np.invert(cosmic_ray_mask), cosmic_ray_mask_true)
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
