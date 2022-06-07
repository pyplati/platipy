import SimpleITK as sitk
import numpy as np

from platipy.imaging.label.comparison import compute_surface_dsc, compute_surface_metrics

def test_surface_dsc():

    label_a = sitk.Image(100,100,100, sitk.sitkUInt8)
    label_a.SetSpacing((1,1,2))
    label_a[30:70, 30:70, 30:70] = 1

    # Check contour within acceptable variation
    label_b = sitk.Image(100,100,100, sitk.sitkUInt8)
    label_b.SetSpacing((1,1,2))
    label_b[30:71, 30:71, 30:71] = 1
    sd = compute_surface_dsc(label_a, label_b)
    assert sd == 1.0

    # Move parts of contour away
    label_b = sitk.Image(100,100,100, sitk.sitkUInt8)
    label_b.SetSpacing((1,1,2))
    label_b[35:71, 35:71, 35:71] = 1
    sd = compute_surface_dsc(label_a, label_b)
    assert np.allclose(sd, 0.5158373786407767)

    # Move parts of contour away to shift one dimension out of boundary
    label_b = sitk.Image(100,100,100, sitk.sitkUInt8)
    label_b.SetSpacing((1,1,2))
    label_b[35:72, 35:72, 35:72] = 1
    sd = compute_surface_dsc(label_a, label_b)
    assert np.allclose(sd, 0.39725541227966404)
    
    # Check other dimensions
    label_b = sitk.Image(100,100,100, sitk.sitkUInt8)
    label_b.SetSpacing((1,1,2))
    label_b[35:75, 35:75, 35:75] = 1
    sd = compute_surface_dsc(label_a, label_b)
    assert np.allclose(sd, 0.1258764241893076)

def test_surface_metrics():

    label_a = sitk.Image(100,100,100, sitk.sitkUInt8)
    label_a.SetSpacing((1,1,2))
    label_a[30:70, 30:70, 30:70] = 1

    # Check contour with small shift
    label_b = sitk.Image(100,100,100, sitk.sitkUInt8)
    label_b.SetSpacing((1,1,2))
    label_b[30:71, 30:71, 30:71] = 1
    metrics = compute_surface_metrics(label_a, label_b)
    assert np.allclose(metrics['hausdorffDistance'], 2.449489742783178)
    assert np.allclose(metrics['meanSurfaceDistance'], 0.6649174304423457)
    assert np.allclose(metrics['medianSurfaceDistance'], 0.574099183082580)
    assert np.allclose(metrics['maximumSurfaceDistance'], 2.4494898319244385)
    assert np.allclose(metrics['sigmaSurfaceDistance'], 101.78549149738755)
    assert np.allclose(metrics['surfaceDSC'], 1.0)

    # Check with larger shift
    label_b = sitk.Image(100,100,100, sitk.sitkUInt8)
    label_b.SetSpacing((1,1,2))
    label_b[35:71, 35:71, 35:71] = 1
    metrics = compute_surface_metrics(label_a, label_b)

    assert np.allclose(metrics['hausdorffDistance'], 12.24744871391589)
    assert np.allclose(metrics['meanSurfaceDistance'], 3.842314521867095)
    assert np.allclose(metrics['medianSurfaceDistance'], 3.5163573920726776)
    assert np.allclose(metrics['maximumSurfaceDistance'], 12.24744871391589)
    assert np.allclose(metrics['sigmaSurfaceDistance'], 392.57229390698296)
    assert np.allclose(metrics['surfaceDSC'], 0.5158373786407767)
