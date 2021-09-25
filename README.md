# CUDA-Fractals

*Student project*

CUDA-accelerated deep zooming into Mandelbrot and Julia using [perturbation theory](http://www.science.eclipse.co.uk/sft_maths.pdf).

# Requirements
- GCC
- GNU Make
- NVCC
- OpenGL
- OpenMP
# Compiling & Usage
```sh
make release && ./bin/main
```
# Media
![Julia set](media/julia.png "Julia set")|![Mandelbrot set](media/mandelbrot.png "Mandelbrot set")
-|-
![deep Mandelbrot set zoom](media/mosaic.png "mosaic in Mandelbrot set")|![mosaic in Mandelbrot set](media/mandelbrot2.png "deep Mandelbrot set zoom")