# import env_init  # This module is in current dir
import numpy as np
from time import perf_counter as time
from tifffile import TiffFile, imwrite
from pycuda import gpuarray

from torchvision.transforms import CenterCrop

import argparse
import tqdm

t = time()

import ssnp

def main():

    parser = argparse.ArgumentParser(description='Simulates the forward model of IDT using the ssnp framework')

    parser.add_argument('-n', '--angle_num', type=int, default=8, help='Number of angles to simulate')

    parser.add_argument('input_file',  type=argparse.FileType("rb"), help='Input file name')
    parser.add_argument('output_file', type=argparse.FileType("wb"), help='Output file name')

    simulate(**vars(parser.parse_args()))

def simulate(input_file, output_file, angle_num):

    ssnp.config.res = (0.1, 0.1, 0.1)

    img = TiffFile(input_file)

    z_slices = len(img.pages)

    n = img.asarray(0)

    if n.dtype.type == np.uint16:
        scale = 0.01 / 0xFFFF
    elif n.dtype.type == np.uint8:
        scale = 0.01 / 0xFF
    else:
        raise ValueError(f"Unknown data type {n.dtype.type} of input image")

    NA = 0.65
    u = ssnp.read("plane", np.complex128, shape=n.shape, gpu=False)
    u = np.broadcast_to(u, (angle_num, *u.shape)).copy()
    beam = ssnp.BeamArray(u)

    for num in range(angle_num):
        xy_theta = num / angle_num * 2 * np.pi
        c_ab = NA * np.cos(xy_theta), NA * np.sin(xy_theta)
        beam.forward[num] *= beam.multiplier.tilt(c_ab, trunc=True, gpu=True)

    beam.backward = 0

    # initializing orginal gpu slice
    n = gpuarray.to_gpu(img.asarray(0)).astype(np.double)

    for num in tqdm.tqdm(range(z_slices), total=z_slices, desc="Simulating z steps"):
        n.set(img.asarray(num).astype(np.double))
        n *= scale
        beam.ssnp(1, n)

    beam.ssnp(-z_slices / 2)
    beam.backward = None
    beam.binary_pupil(1.0001 * NA)

    measurements = np.abs(beam.forward.get())
    measurements -= measurements.min()
    measurements *= 0xFFFF / measurements.max()
    measurements = measurements.astype(np.uint16)

    imwrite(output_file, measurements, imagej=True)

    # ssnp.write(output_file, measurements, scale=0.5, pre_operator=lambda x: np.abs(x))

if __name__ == '__main__':
    main()