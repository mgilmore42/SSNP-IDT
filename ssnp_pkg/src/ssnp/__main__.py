# import env_init  # This module is in current dir
import numpy as np
from time import perf_counter as time
from tifffile import TiffFile, imwrite
from pycuda import gpuarray

import pycuda.driver as cuda

from concurrent.futures import ThreadPoolExecutor

from torchvision.transforms import CenterCrop

import argparse
import tqdm

import ssnp

def main():
    '''
        Logic to parse command line arguments and call the main function
    '''

    parser = argparse.ArgumentParser(description='Simulates the forward model of IDT using the ssnp framework')

    parser.add_argument('-n', '--angle_num', type=int, default=8, help='Number of angles to simulate')

    parser.add_argument('input_file',  type=argparse.FileType("rb"), help='Input file name')
    parser.add_argument('output_file', type=argparse.FileType("wb"), help='Output file name')

    simulate(**vars(parser.parse_args()))

def gen_data(img, scale=0.01 / 0xFFFF):
    '''
        Generator that yields the data from the input image one z slice at a time
    '''
    
    def to_gpu(arr_cpu):
        arr_gpu = gpuarray.to_gpu(arr_cpu)
        arr_final = arr_gpu.astype(np.double)
        arr_final *= scale
        return arr_final

    for i in range(len(img.pages)):
        yield to_gpu(img.asarray(i))

def simulate(input_file, output_file, angle_num):
    '''
        Simulates the forward model of IDT using the ssnp framework
    '''

    ssnp.config.res = (0.1, 0.1, 0.1)

    img = TiffFile(input_file)

    z_slices = len(img.pages)

    arr_init = img.asarray(0)

    if arr_init.dtype.type == np.uint16:
        scale = 0.01 / 0xFFFF
    elif arr_init.dtype.type == np.uint8:
        scale = 0.01 / 0xFF
    else:
        raise ValueError(f"Unknown data type {arr_init.dtype.type} of input image")

    NA = 0.65
    u = ssnp.read("plane", np.complex128, shape=arr_init.shape, gpu=False)
    u = np.broadcast_to(u, (angle_num, *u.shape)).copy()
    beam = ssnp.BeamArray(u)

    for num in range(angle_num):
        xy_theta = num / angle_num * 2 * np.pi
        c_ab = NA * np.cos(xy_theta), NA * np.sin(xy_theta)
        beam.forward[num] *= beam.multiplier.tilt(c_ab, trunc=True, gpu=True)

    beam.backward = 0

    for arr in tqdm.tqdm(gen_data(img, scale), total=z_slices, desc="Simulating z steps"):
        beam.ssnp(1, arr)

    beam.ssnp(-z_slices / 2)
    beam.backward = None
    beam.binary_pupil(1.0001 * NA)

    measurements = np.abs(beam.forward.get())
    measurements -= measurements.min()
    measurements *= 0xFFFF / measurements.max()
    measurements = measurements.astype(np.uint16)

    imwrite(output_file, measurements, imagej=True)

if __name__ == '__main__':
    main()