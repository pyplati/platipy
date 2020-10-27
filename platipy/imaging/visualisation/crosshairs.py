"""
Code to display orthogonal image slices with crosshairs displaying the slice position

"""

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

"""
Settings
Fill out this section
"""

# Define the image
im = sitk.ReadImage('/home/robbie/Work/3_ResearchProjects/Chinese Lung Cancer/1_data/Case_02/Images/Case_02_0%.nii.gz')

# Define the cut location - this is where the crosshairs will appear
# Given as [axial, coronal, sagittal] location
cut = [30, 220, 330]

# Define the figure size (inches), colormap, intensity windowing ([min, range])
figSize=6
cmap=plt.cm.Greys_r
window=[-250, 500]

# Output file name
fig_name = './test.png'

"""
Code - shouldn't need to edit this
"""

def returnSlice(axis, index):
    if axis == "x":
        s = (slice(None), slice(None), index)
    if axis == "y":
        s = (slice(None), index, slice(None))
    if axis == "z":
        s = (index, slice(None), slice(None))

    return s

nda = sitk.GetArrayFromImage(im)

# Get data for correct visualisation
(AxSize, CorSize, SagSize) = nda.shape
spPlane, _, spSlice = im.GetSpacing()
asp = (1.0 * spSlice) / spPlane


# Set up figure
fSize = (
    figSize,
    figSize * (asp * AxSize + CorSize) / (1.0 * SagSize + CorSize),
)

fig, ((axAx, blank), (axCor, axSag)) = plt.subplots(
    2,
    2,
    figsize=fSize,
    gridspec_kw={"height_ratios": [(CorSize) / (asp * AxSize), 1], "width_ratios": [SagSize, CorSize]},
);
blank.axis("off")

# Get slices
sAx = returnSlice("z", cut[0])
sCor = returnSlice("y", cut[1])
sSag = returnSlice("x", cut[2])

# Display image data
imAx = axAx.imshow(
    nda.__getitem__(sAx),
    aspect=1.0,
    interpolation=None,
    cmap=cmap,
    clim=(window[0], window[0] + window[1]),
)
imCor = axCor.imshow(
    nda.__getitem__(sCor),
    origin="lower",
    aspect=asp,
    interpolation=None,
    cmap=cmap,
    clim=(window[0], window[0] + window[1]),
)
imSag = axSag.imshow(
    nda.__getitem__(sSag),
    origin="lower",
    aspect=asp,
    interpolation=None,
    cmap=cmap,
    clim=(window[0], window[0] + window[1]),
)

# Display crosshairs
# Axial image, FAKE cut (just for label)
axAx.plot([0, 0], [0, 0], c='yellow', label=f'Axial slice: {cut[0]}')
# Axial image, coronal cut
axAx.plot([0, SagSize], [cut[1], cut[1]], c='r', label=f'Coronal slice: {cut[1]}')
# Axial image, sagittal cut
axAx.plot([cut[2], cut[2]], [0, CorSize], c='orange', label=f'Sagittal slice: {cut[2]}')

axAx.legend(loc='center left', bbox_to_anchor=(1.05,0.5))

# Sag image, ax cut
axSag.plot([0, CorSize], [cut[0], cut[0]], c='yellow')
# Sag image, cor cut
axSag.plot([cut[1], cut[1]], [0, AxSize], c='r')


# Cor image, ax cut
axCor.plot([0, SagSize], [cut[0], cut[0]], c='yellow')
# Cor image, sagittal cut
axCor.plot([cut[2], cut[2]], [0, AxSize], c='orange')


# Turn off axes
axAx.axis("off")
axCor.axis("off")
axSag.axis("off")

# Adjust spacing
fig.subplots_adjust(left=0, right=1, wspace=0.01, hspace=0.01, top=1, bottom=0)

# Save image
fig.savefig(f'{fig_name}', dpi=300, transparent=True)