import env_init  # This module is in current dir
import numpy as np
from time import perf_counter as time
from tifffile import TiffFile
from pycuda import gpuarray

t = time()

import ssnp

ANGLE_NUM = 8
ssnp.config.res = (0.1, 0.1, 0.1)
img = TiffFile("sample.tiff")

n = img.asarray(0)
if n.dtype.type == np.uint16:
    scale = 0.01 / 0xFFFF
elif n.dtype.type == np.uint8:
    scale = 0.01 / 0xFF
else:
    raise ValueError(f"Unknown data type {n.dtype.type} of input image")

NA = 0.65
u = ssnp.read("plane", np.complex128, shape=n.shape, gpu=False)
u = np.broadcast_to(u, (ANGLE_NUM, *u.shape)).copy()
beam = ssnp.BeamArray(u)

for num in range(ANGLE_NUM):
    xy_theta = num / ANGLE_NUM * 2 * np.pi
    c_ab = NA * np.cos(xy_theta), NA * np.sin(xy_theta)
    beam.forward[num] *= beam.multiplier.tilt(c_ab, trunc=True, gpu=True)

beam.backward = 0

for num in range(len(img.pages)):
    n = img.asarray(num).astype(np.double)
    n = gpuarray.to_gpu(n)
    n *= scale
    beam.ssnp(1, n)

beam.ssnp(-len(img.pages) / 2)
beam.backward = None
beam.binary_pupil(1.0001 * NA)

measurements = beam.forward.get()
print(time() - t)

ssnp.write("meas_sim.tiff", measurements, scale=0.5, pre_operator=lambda x: np.abs(x))
print("finished saving")