import math
import functools
from warnings import warn
from collections.abc import Iterable
import numpy as np
from pycuda import gpuarray
from numbers import Number


def _cache_array(func):
    @functools.wraps(func)
    def get_cache(self, *args, gpu=False, **kwargs):
        self: Multipliers
        key, calc = func(self, *args, **kwargs)
        if gpu:
            try:
                if not hasattr(self, "_gpu_cache_ru"):
                    raise KeyError()
                
                if self._gpu_cache_ru == key:
                    return self._gpu_cache[key]
                else:

                    if self._gpu_cache_ru != key:
                        # frees old cached array from gpu
                        key_old = self._gpu_cache_ru
                        arr_gpu = self._gpu_cache[key_old]
                        arr_cpu = arr_gpu.get(pagelocked=True)
                        arr_gpu.gpudata.free()
                        self._gpu_cache[key_old] = arr_cpu

                    # puts cached array to gpu
                    arr_cpu = self._gpu_cache[key]
                    arr_gpu = gpuarray.to_gpu_async(arr_cpu, stream=self.stream)
                    self._gpu_cache[key] = arr_gpu
                    self._gpu_cache_ru = key

                    return arr_gpu

            except KeyError:
                arr = gpuarray.to_gpu_async(get_cache(self, *args, gpu=False, **kwargs), stream=self.stream)
                self._gpu_cache[key] = arr
                self._gpu_cache_ru = key
                return arr
        try:
            return self._cache[key]
        except KeyError:
            arr = calc()
            assert arr.shape == self._shape
            self._cache[key] = arr
            return arr

    return get_cache


@functools.lru_cache
class Multipliers:
    def __init__(self, shape, res, stream=None):
        assert len(shape) == 2
        self._xy_size = (shape[1], shape[0])
        self._shape = shape
        self.res = res
        self._cache = {}
        self._gpu_cache = {}
        self.stream = stream

    @_cache_array
    def tilt(self, c_ab, *, trunc, periodic_params=None, c_ab_out=None):
        res = self.res
        xy_size = self._xy_size
        norm = tuple(xy_size[i] * res[i] for i in (0, 1))  # to be confirmed: * config.n0
        kernel = None
        if trunc:
            c_ab = [math.trunc(c_ab[i] * norm[i]) / norm[i] for i in (0, 1)]
        elif periodic_params is not None:
            kernel = self.gaussian(periodic_params[0], (0.5, 0.5), gpu=False)
            kernel = np.fft.fft2(np.fft.fftshift(kernel))
        c_ab = tuple(float(i) for i in c_ab)
        if c_ab_out is not None:
            for i in range(2):
                c_ab_out[i] = c_ab[i]

        key = ("t", c_ab, res, None if periodic_params is None else periodic_params[0])

        def calc():
            xr, yr = [np.arange(xy_size[i]) / xy_size[i] * c_ab[i] * norm[i] for i in (0, 1)]
            phase = np.mod(xr + yr[:, None], 1).astype(np.complex128)
            phase = np.exp(2j * np.pi * phase)
            if kernel is not None:
                phase = np.fft.fft2(phase)
                phase *= kernel
                phase = np.fft.ifft2(phase)
                # change fft default f-contiguous output to c-contiguous
                phase = np.ascontiguousarray(phase)
            # normalize by center point value
            phase /= phase[tuple(i // 2 for i in phase.shape)]
            np.testing.assert_almost_equal(phase[tuple(i // 2 for i in phase.shape)], 1)
            return phase

        return key, calc

    @_cache_array
    def binary_pupil(self, na, n0=1):
        res = self.res
        c_gamma = self.c_gamma()
        na /= n0  # NA = n0 sin(gamma), use NA' = NA/n0 to make the mask formula the same as air
        key = ("bp", round(na, 4), res)

        def calc():
            mask = np.greater(c_gamma, np.sqrt(1 - na ** 2))
            mask = mask.astype(np.complex128)
            return mask

        return key, calc

    @_cache_array
    def c_gamma(self, shift=0):
        """
        Calculate gamma as a constant array at frequency domain. Gamma is the ``cos`` value of the angle
        between the wave vector and z-axis. Note: ``NA**2 + gamma**2 == 1``

        The array is pre-shifted for later FFT operation.
        :param shift: 1 or 2 float numbers in [0, 1) (decimal part is used) marking the position of zero frequency.
        For example, 0.5 have similar effect as fftshift.
        :return: gamma as a numpy array
        """
        if isinstance(shift, Number):
            shift = (shift, shift)
        xy_size = self._xy_size
        res = self.res
        key = ("cg", res, *[round(s, 4) for s in shift])

        def calc():
            eps = 1E-8
            c_alpha, c_beta = [self._near_0(xy_size[i], shift[i]) / res[i] for i in (0, 1)]
            c_gamma = np.sqrt(np.maximum(1 - (np.square(c_alpha) + np.square(c_beta[:, None])), eps))
            return c_gamma

        return key, calc

    @staticmethod
    def _near_0(size, pos_0, edge=True):
        """
        Return ``f(x)``, which satisfies:

        ``f(x)`` range ``[-0.5, 0.5)``,

        ``f'(x) = 1`` (except for the discontinuity point jumping from 0.5 to -0.5),

        ``f(pos_0) = 0``

        The index of the returned array determines x (range is ``[0, 1)``)

        Example of x at size = 3:
          - Index as edge: x = [0, 1/3, 2/3]
          - Index as center (edge = False): x = [1/6, 3/6, 5/6]

        :param size: arr length
        :param pos_0: a number in [0, 1) (otherwise wrap around and use decimal part) marking the position of zero
        :param edge: use left edge as x (default), otherwise use center as x.
        :return: 1-D array of f(x)
        """
        return np.mod((np.arange(size).astype(np.double) + 0.5 * (not edge)) / size + 0.5 - pos_0, 1) - 0.5

    @_cache_array
    def soft_crop(self, width, *, total_slices=1, pos=0, strength=1):
        xy_size = self._xy_size
        width = float(width)
        if width >= 1 or width <= 0:
            raise ValueError("width should be a relative value in 0-1")
        key = ("cr", round(width, 4), round(pos, 4), round(total_slices), round(strength, 4))
        if strength < 0.01:
            warn("strength is too weak, change to 0.01", stacklevel=3)
            strength = 0.01

        def calc():
            x, y = [
                np.exp(-np.exp(-(self._near_0(xy_size[i], pos) * 2 / width) ** 2 * (np.log(100 * strength) + 0.8))
                       * 100 * strength / total_slices)
                for i in (0, 1)
            ]
            mask = x * y[:, None]
            return mask

        return key, calc

    @_cache_array
    def hard_crop(self, width):
        """
        Possible width specs:

        * 0~1 ``float``: relative crop value, same for each axis
        * ``int``: crop pixels, same for each axis
        * ``Iterable`` (list, tuple, ...): Indicate crop pixels of ``(left, right, top, bottom)``,
          or ``(x, y)`` for same pixels each side

        :param width: crop size specification
        """
        err_width = ValueError(f"invalid width spec {width}")
        xy_size = self._xy_size
        if isinstance(width, float):
            width = width / 2 + 1e-8
            x_size, y_size = [int(width * i) for i in xy_size]
            width = (x_size, x_size, y_size, y_size)
        elif isinstance(width, int):
            width = (width,) * 4
        elif isinstance(width, Iterable):
            width = tuple(width)
            if len(width) == 2:
                x_size, y_size = width
                width = (x_size, x_size, y_size, y_size)
            elif len(width) != 4:
                raise err_width
        else:
            raise err_width
        for i in width:
            if not (isinstance(i, int) and i >= 0):
                raise err_width
        key = ("hc", width)

        def calc():
            x, y = [
                np.zeros(xy_size[i], bool)
                for i in (0, 1)
            ]
            x1, x2, y1, y2 = width
            x[x1:-x2] = True
            y[y1:-y2] = True
            mask = np.logical_and(x, y[:, None])
            return mask.astype(np.double)

        return key, calc

    @_cache_array
    def gaussian(self, sigma, mu=(0, 0)):
        xy_size = self._xy_size
        sigma = float(sigma)
        mu = tuple(float(i) for i in mu)
        for i in mu:
            if i > 1 or i < 0:
                warn("mu value is out of normal range 0~1")
        key = ("ga", round(sigma, 4), tuple(round(i, 4) for i in mu))

        def calc():
            x, y = [
                np.exp(-(self._near_0(xy_size[i], mu[i])) ** 2
                       / 2 / (sigma / xy_size[i]) ** 2)
                for i in (0, 1)
            ]
            mask = x * y[:, None]
            return mask

        return key, calc
