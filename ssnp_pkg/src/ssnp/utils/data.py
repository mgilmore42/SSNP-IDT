import sys

import numpy as np
from warnings import warn
import os
import csv
from pycuda import gpuarray
from pycuda import driver as cuda


def predefined_read(name, shape, dtype=None):
    """
    Get a predefined tensor
    ``shape`` can be a ``list`` of integers, a ``tuple`` of integers,
    or a 1-D ``Tensor`` of type ``int32``

    :param name: predefined image name
    :param shape: Dimensions of resulting tensor
    :param dtype: DType of the elements of the resulting tensor
    """
    if shape is None:
        raise ValueError("indeterminate shape")
    if name == "plane":
        img = np.ones(shape, dtype)
    else:
        raise ValueError(f"unknown image name {name}")
    return img


def tiff_read(path):
    from tifffile import TiffFile
    """
    Import a TIFF file as numpy array

    :param path: Target file path
    :return: A tensor constant
    """
    with TiffFile(path) as file:
        img = np.squeeze(np.stack([page.asarray() for page in file.pages]))
    if img.dtype.type == np.uint16:
        img = img.astype(np.double) / 65535
    elif img.dtype.type == np.uint8:
        warn("Importing uint8 image, please use uint16 if possible")
        img = img.astype(np.double) / 255
    else:
        raise TypeError(f"Unknown data type {img.dtype.type} of input image")
    return img


def np_read(path, *, key=None):
    ext = os.path.splitext(path)[-1]
    if key is None:
        key = 'arr_0'
    if ext == '.npy':
        img = np.load(path)
    elif ext == '.npz':
        with np.load(path) as f:
            try:
                img = f[key]
            except KeyError:
                warn(f"'{key}' is not a file in archive {path}. Reading first file instead.", stacklevel=2)
                img = f[f.files[0]]
    else:
        raise ValueError(f"unknown filename extension '{ext}'")
    return img


def csv_read(path: str, quoting=csv.QUOTE_NONNUMERIC, **kwargs):
    table = []
    with open(path, 'r', newline='') as file:
        reader = csv.reader(file, quoting=quoting, **kwargs)
        for row in reader:
            table.append(row)
    return np.array(table)


def mat_read(path, *, key=None):
    from scipy.io import loadmat, whosmat
    try:
        if key is None:
            key = whosmat(path)[0][0]
        img = loadmat(path)[key]
    except NotImplementedError as e:
        if 'HDF' in e.args[0]:
            return hdf5_read(path, key=key)
        else:
            raise e
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Unknown data type {type(img)} of input image")
    return img

def hdf5_read(path, *, key=None):
    import h5py
    with h5py.File(path) as f:
        if key is None:
            key = list(f.keys())[0]
        img = np.array(f[key])
    return img

def read(source, dtype=None, shape=None, *, scale=1., gpu=False, pagelocked=False, **kwargs):
    """
    Read a ``tf.Tensor`` from source. Method is auto-detected corresponding to the file extension.

    :param source: Target file path
    :param dtype: Data type of the elements of the resulting tensor
    :param shape:
    :param scale:
    :param gpu: whether load memory to gpu
    :param pagelocked: for cpu, whether make it pinned
    :return:
    """
    if source in {'plane'}:
        arr = predefined_read(source, shape, dtype)
    elif os.access(source, os.R_OK):
        ext = os.path.splitext(source)[-1].lower()
        if ext in {'.tiff', '.tif'}:
            arr = tiff_read(source)
        elif ext in {'.npy', '.npz'}:
            arr = np_read(source, **kwargs)
        elif ext == '.mat':
            arr = mat_read(source, **kwargs)
        elif ext == '.hdf5':
            arr = hdf5_read(source, **kwargs)
        elif ext == '.csv':
            arr = csv_read(source)
        else:
            raise ValueError(f"unknown filename extension '{ext}'")
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)

    else:
        raise FileNotFoundError(f"'{source}' is not a known illumination type nor a valid file path")
    if shape is not None and tuple(shape) != tuple(arr.shape):
        arr = arr.reshape(shape)
    rank = len(arr.shape)
    if rank not in {2, 3}:
        warn(f"Importing {rank}-D image with shape {arr.shape}", stacklevel=2)
    arr *= scale
    if gpu:
        arr = gpuarray.to_gpu(arr)
    else:
        if pagelocked:
            arr = cuda.register_host_memory(arr)
    return arr


def tiff_write(path, arr, *, scale=1, pre_operator: callable = None, dtype=np.uint16,
               compression='zlib', photometric=None):
    from tifffile import TiffWriter
    arr = np.squeeze(arr)
    shape = arr.shape
    if photometric == 'rgb':
        assert shape[-1] == 3, "rgb need exactly 3 channels"
        shape = shape[:-1]
    if len(shape) == 2:
        arr = [arr]
    elif len(shape) != 3:
        raise ValueError(f"Must export 2-D or 3-D array but got {len(arr.shape)}-D data, "
                         f"shape {arr.shape}.")
    with TiffWriter(path) as out_file:
        for i in arr:
            if pre_operator is not None:
                i = pre_operator(i)
            try:
                if scale is not None:
                    i = i * scale * {np.uint16: 65535, np.uint8: 255}[dtype]
                i = i.astype(np.int64)
                np.clip(i, 0, {np.uint16: 65535, np.uint8: 255}[dtype], out=i)
            except KeyError:
                raise TypeError(f"dtype should be either np.uint8 or np.uint16, but not {dtype}") from None
            i = i.astype(dtype)
            if compression is not None:
                out_file.write(i, compression=compression, predictor=True, photometric=photometric)
            else:
                out_file.write(i, photometric=photometric)


def np_write(path, arr, *, scale=None, pre_operator=None, dtype=None, compress=True):
    if pre_operator is not None:
        arr = pre_operator(arr)
    if scale is not None:
        arr = arr * scale
    if dtype is not None:
        arr = arr.astype(dtype)
    ext = os.path.splitext(path)[-1]
    if ext == '.npy':
        np.save(path, arr)
    elif ext == '.npz':
        np.savez_compressed(path, arr) if compress else np.savez(path, arr)


def binary_write(path, arr, *, scale=None, pre_operator=None, dtype=None, add_hint=False):
    if pre_operator is not None:
        arr = pre_operator(arr)
    if scale is not None:
        arr = arr * scale
    if dtype is not None:
        arr = arr.astype(dtype)
    if add_hint:
        path = os.path.splitext(path)
        path_list = [path[0]]
        for i in arr.shape:
            path_list.append(f"_{i}")
        path_list.append(f"_{arr.dtype}")
        path_list.append(f"_{sys.byteorder[0]}e")
        path_list.append(path[1])
        path = "".join(path_list)
    arr.tofile(path)


def csv_write(path: str, table, **kwargs):
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file, **kwargs)
        for row in table:
            writer.writerow(row)


def mat_write(path, arr, *, scale=None, pre_operator=None, dtype=None, compress=True, key="data"):
    from scipy.io import savemat
    if pre_operator is not None:
        arr = pre_operator(arr)
    if scale is not None:
        arr = arr * scale
    if dtype is not None:
        arr = arr.astype(dtype)
    savemat(path, {key: arr}, do_compression=compress)


def write(dest, array, **kwargs):
    if isinstance(array, np.ndarray):
        arr = array
    elif isinstance(array, gpuarray.GPUArray):
        arr = array.get()
    else:
        arr = [page for page in array]
        if isinstance(arr[0], np.ndarray):
            arr = np.stack(arr)
        elif isinstance(arr[0], gpuarray.GPUArray):
            arr = np.stack([gpu_arr.get() for gpu_arr in arr])
        else:
            try:
                warn("trying general np.array fallback", stacklevel=2)
                arr = np.array(array, copy=False)
            except Exception as ee:
                raise TypeError("unknown data type to write") from ee

    ext = os.path.splitext(dest)[-1]
    if ext in {'.tiff', '.tif'}:
        tiff_write(dest, arr, **kwargs)
    elif ext in {'.npy', '.npz'}:
        np_write(dest, arr, **kwargs)
    elif ext == '.mat':
        mat_write(dest, arr, **kwargs)
    elif ext == '.bin':
        binary_write(dest, arr, **kwargs)
    elif ext == '.csv':
        csv_write(dest, arr)
    else:
        raise ValueError(f"unknown filename extension '{ext}'")
