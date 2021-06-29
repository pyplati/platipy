import numpy as np
import SimpleITK as sitk

from scipy.interpolate import griddata

from platipy.imaging.label.utils import vectorised_transform_index_to_physical_point


def evaluate_distance_on_surface(
    reference_volume, test_volume, abs_distance=True, reference_as_distance_map=False
):
    """
    Evaluates a distance map on a surface
    Input: reference_volume: binary volume SimpleITK image, or alternatively a distance map
           test_volume: binary volume SimpleITK image
    Output: theta, phi, values
    """
    if reference_as_distance_map:
        reference_distance_map = reference_volume
    else:
        if abs_distance:
            reference_distance_map = sitk.Abs(
                sitk.SignedMaurerDistanceMap(
                    reference_volume, squaredDistance=False, useImageSpacing=True
                )
            )

        else:
            reference_distance_map = sitk.SignedMaurerDistanceMap(
                reference_volume, squaredDistance=False, useImageSpacing=True
            )

    test_surface = sitk.LabelContour(test_volume)

    distance_image = sitk.Multiply(
        reference_distance_map, sitk.Cast(test_surface, sitk.sitkFloat32)
    )
    distance_array = sitk.GetArrayFromImage(distance_image)

    # Get centre of mass of reference volume
    reference_volume_array = sitk.GetArrayFromImage(reference_volume)
    reference_volume_locations = np.where(reference_volume_array == 1)
    com_index = reference_volume_locations.mean(axis=1)
    com_real = vectorised_transform_index_to_physical_point(reference_volume, com_index)

    # Calculate centre of mass in real coordinates
    test_surface_array = sitk.GetArrayFromImage(test_surface)
    test_surface_locations = np.where(test_surface_array == 1)
    test_surface_locations_array = np.array(test_surface_locations)

    # Calculate each point on the surface in real coordinates
    pts = test_surface_locations_array.T
    pts_real = vectorised_transform_index_to_physical_point(test_surface, pts)
    pts_diff = pts_real - com_real

    # Convert to spherical polar coordinates - base at north pole
    rho = np.sqrt((pts_diff * pts_diff).sum(axis=1))
    theta = np.pi / 2.0 - np.arccos(pts_diff.T[0] / rho)
    phi = -1 * np.arctan2(pts_diff.T[2], -1.0 * pts_diff.T[1])

    # Extract values
    values = distance_array[test_surface_locations]

    return theta, phi, values


def evaluate_distance_to_reference(reference_volume, test_volume, resample_factor=1):
    """
    Evaluates the distance from the surface of a test volume to a reference
    Input: reference_volume: binary volume SimpleITK image
           test_volume: binary volume SimpleITK image
    Output: values : the distance to each point on the reference volume surface
    """

    # TO DO
    # come up with a better resampling strategy
    # e.g. resample images prior to this process?

    # compute the distance map from the test volume surface
    test_distance_map = sitk.Abs(
        sitk.SignedMaurerDistanceMap(test_volume, squaredDistance=False, useImageSpacing=True)
    )

    # get the distance from the test surface to the reference surface
    ref_surface = sitk.LabelContour(reference_volume)
    ref_surface_pts = sitk.GetArrayFromImage(ref_surface) == 1
    surface_values = sitk.GetArrayFromImage(test_distance_map)[ref_surface_pts]

    # resample to keep the points to a reasonable amount
    values = surface_values[::resample_factor]

    return values


def regrid_spherical_data(theta, phi, values, resolution):
    """
    Re-grids spherical data
    Input: theta, phi, values
    Options: plot a figure (plotFig), save a figure (saveFig), case identifier (figName)
    Output: p_lat, p_long, grid_values (, fig)
    """
    # Re-grid:
    #  Set up grid
    d_radian = resolution * np.pi / 180
    p_long, p_lat = np.mgrid[-np.pi : np.pi : d_radian, -np.pi / 2.0 : np.pi / 2.0 : d_radian]

    # First pass - linear interpolation, works well but not for edges
    grid_values = griddata(
        list(zip(theta, phi)), values, (p_lat, p_long), method="linear", rescale=False
    )

    # Second pass - nearest neighbour interpolation
    grid_values_nn = griddata(
        list(zip(theta, phi)), values, (p_lat, p_long), method="nearest", rescale=False
    )

    # Third pass - wherever the linear interpolation isn't defined use nearest neighbour
    # interpolation
    grid_values[~np.isfinite(grid_values)] = grid_values_nn[~np.isfinite(grid_values)]

    return p_lat, p_long, grid_values
