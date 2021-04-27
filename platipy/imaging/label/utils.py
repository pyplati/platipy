import SimpleITK as sitk

from scipy.ndimage.measurements import center_of_mass


def get_com(label, real_coords=False):
    """
    Get centre of mass of a SimpleITK.Image
    """
    arr = sitk.GetArrayFromImage(label)
    com = center_of_mass(arr)

    if real_coords:
        com = label.TransformContinuousIndexToPhysicalPoint(com)

    else:
        com = [int(i) for i in com]

    return com


def vectorised_transform_index_to_physical_point(image, point_array, rotate=True):
    """
    Transforms a set of points from array indices to real-space
    """
    if rotate:
        spacing = image.GetSpacing()[::-1]
        origin = image.GetOrigin()[::-1]
    else:
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
    return point_array * spacing + origin


def vectorised_transform_physical_point_to_index(image, point_array, rotate=True):
    """
    Transforms a set of points from real-space to array indices
    """
    if rotate:
        spacing = image.GetSpacing()[::-1]
        origin = image.GetOrigin()[::-1]
    else:
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
    return (point_array - origin) / spacing
