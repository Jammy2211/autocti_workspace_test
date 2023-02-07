"""
Simulator: Uniform Charge Injection With Cosmic Rays
====================================================

This script simulates charge injection imaging with CTI, where:

 - Parallel CTI is added to the image using a 2 `Trap` species model.
 - The volume filling behaviour in the parallle direction using the `CCD` class.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import json
from os import path
import autocti as ac
import autocti.plot as aplt

"""
__Columns__

Simulate datasets with variable numbers of columns, to profile the run-times of a LH evaluation as a function of 
data quantity.
"""
total_columns_list = [10, 100, 1000, 2000]

for total_columns in total_columns_list:

    """
    __Dataset Paths__

    The 'dataset_label' describes the type of data being simulated (in this case, imaging data) and 'dataset_name'
    gives it a descriptive name. They define the folder the dataset is output to on your hard-disk:

     - The image will be output to '/autocti_workspace/dataset/dataset_label/dataset_name/image.fits'.
     - The noise-map will be output to '/autocti_workspace/dataset/dataset_label/dataset_name/noise_map.fits'.
     - The pre_cti_data will be output to '/autocti_workspace/dataset/dataset_label/dataset_name/pre_cti_data.fits'.
    """
    dataset_type = "imaging_ci"
    dataset_label = "uniform"
    dataset_name = "parallel_x1_with_noise"
    dataset_size = f"columns_{total_columns}"

    dataset_path = path.join(
        "dataset", dataset_type, dataset_label, dataset_name, dataset_size
    )

    """
    __Layout__
    
    The 2D shape of the image.
    """
    shape_native = (2000, total_columns)

    """
    The locations (using NumPy array indexes) of the parallel overscan, serial prescan and serial overscan on the image.
    """
    parallel_overscan = ac.Region2D(
        (shape_native[0] - 1, shape_native[0], 1, shape_native[1] - 1)
    )
    serial_prescan = ac.Region2D((0, shape_native[0], 0, 1))
    serial_overscan = ac.Region2D(
        (0, shape_native[0] - 1, shape_native[1] - 1, shape_native[1])
    )

    """
    Specify the charge injection regions on the CCD, which in this case is 5 equally spaced rectangular blocks.
    """
    regions_list = [
        (30, 60, serial_prescan[3], serial_overscan[2]),
        (360, 390, serial_prescan[3], serial_overscan[2]),
        (690, 720, serial_prescan[3], serial_overscan[2]),
        (1020, 1050, serial_prescan[3], serial_overscan[2]),
        (1350, 1380, serial_prescan[3], serial_overscan[2]),
        (1680, 1710, serial_prescan[3], serial_overscan[2]),
    ]

    """
    The normalization of every charge injection image, which determines how many images are simulated.
    """
    norm_list = [100.0, 250.0, 500.0, 1000.0, 5000.0, 10000.0, 30000.0, 200000.0]

    """
    Create the layout of the charge injection pattern for every charge injection normalization.
    """
    layout_list = [
        ac.Layout2DCI(
            shape_2d=shape_native,
            region_list=regions_list,
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
        )
        for norm in norm_list
    ]

    """
    __Clocker__
    
    The `Clocker` models the CCD read-out, including CTI. 
    
    For parallel clocking, we use 'charge injection mode' which transfers the charge of every pixel over the full CCD.
    """
    clocker = ac.Clocker2D(
        parallel_express=2,
        parallel_roe=ac.ROEChargeInjection(),
        parallel_prune_frequency=0,
    )

    """
    __CTI Model__
    
    The CTI model used by arCTIc to add CTI to the input image in the parallel direction, which contains: 
    
     - 2 `Trap` species in the parallel direction.
     - A simple CCD volume beta parametrization.
    """
    parallel_trap_0 = ac.TrapInstantCapture(density=0.1, release_timescale=1.0)

    parallel_trap_list = [parallel_trap_0]

    parallel_ccd = ac.CCDPhase(
        well_fill_power=0.58, well_notch_depth=0.0, full_well_depth=200000.0
    )

    cti_2d = ac.CTI2D(parallel_trap_list=parallel_trap_list, parallel_ccd=parallel_ccd)

    """
    __Simulate__
    
    To simulate charge injection imaging, we pass the charge injection pattern to a `SimulatorImagingCI`, which adds CTI 
    via arCTIc and read-noise to the data.
    
    This creates instances of the `ImagingCI` class, which include the images, noise-maps and pre_cti_data images.
    """
    simulator_list = [
        ac.SimulatorImagingCI(read_noise=4.0, pixel_scales=0.1, norm=norm)
        for norm in norm_list
    ]

    """
    We now pass each charge injection pattern to the simulator. This generate the charge injection image of each exposure
    and before passing each image to arCTIc does the following:
    
     - Uses an input read-out electronics corner to perform all rotations of the image before / after adding CTI.
     - Stores this corner so that if we output the files to .fits,they are output in their original and true orientation.
     - Includes information on the different scan regions of the image, such as the serial prescan and serial overscan.
    """
    imaging_ci_list = [
        simulator.via_layout_from(clocker=clocker, layout=layout_ci, cti=cti_2d)
        for layout_ci, simulator in zip(layout_list, simulator_list)
    ]

    """
    __Output__

    Output subplots of the simulated dataset to the dataset path as .png files.
    """
    for imaging_ci, norm in zip(imaging_ci_list, norm_list):
        output = aplt.Output(
            path=path.join(dataset_path, f"norm_{int(norm)}"),
            filename="imaging_ci",
            format="png",
        )

        mat_plot_2d = aplt.MatPlot2D(output=output)

        imaging_ci_plotter = aplt.ImagingCIPlotter(
            dataset=imaging_ci, mat_plot_2d=mat_plot_2d
        )
        imaging_ci_plotter.subplot_imaging_ci()

    """
    __Output__
    
    Output the image, noise-map and pre cti image of the charge injection dataset to .fits files.
    """
    [
        imaging_ci.output_to_fits(
            image_path=path.join(dataset_path, f"norm_{int(norm)}", "image.fits"),
            noise_map_path=path.join(
                dataset_path, f"norm_{int(norm)}", "noise_map.fits"
            ),
            pre_cti_data_path=path.join(
                dataset_path, f"norm_{int(norm)}", "pre_cti_data.fits"
            ),
            overwrite=True,
        )
        for imaging_ci, norm in zip(imaging_ci_list, norm_list)
    ]

    """
    Determine the log likelihood of the true model and output with the data for reference.
    """
    log_likelihood_list = []

    for imaging_ci in imaging_ci_list:

        post_cti_data = clocker.add_cti(data=imaging_ci.pre_cti_data, cti=cti_2d)

        fit = ac.FitImagingCI(dataset=imaging_ci, post_cti_data=post_cti_data)

        log_likelihood_list.append(fit.figure_of_merit)

    fit_file = path.join(dataset_path, "fit.json")

    with open(fit_file, "w") as f:
        json.dump(log_likelihood_list, f)


"""
Finished.
"""
