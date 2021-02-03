# pvs-uebung-5

## dependencies

- ubuntu: `ocl-icd-opencl-dev ocl-icd-libopencl1 opencl-headers`
    - nvidia: `nvidia-libopencl1-384`
    - intel: `beignet-opencl-icd` (`beignet` on older versions)
    - additionally, `clinfo` might be useful

- arch: `ocl-icd opencl-headers opencl-clhpp`, and a
[runtime](https://wiki.archlinux.org/index.php/GPGPU#OpenCL_Runtime),
e.g. `opencl-nvidia`

## running

A Makefile has been provided.  To select an iteration of the kernel,
pass the `KERNEL` option.  E.g., to run and test the first
optimization of the kernel run:
```
make test KERNEL=1
```
