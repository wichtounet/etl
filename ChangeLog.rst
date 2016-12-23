In development - ETL 1.1
++++++++++++++++++++++++

* *Performance* Vectorization of signed integer operations
* *Performance* Faster CPU convolutions
* *Performance* Better parallelization of convolutions
* *Performance* Better usage of GPU contexts
* *Performance* Much better GEMM/GEMV/GEVM kernels (when BLAS not available)
* *Performance* Reduced overhead for 3D/4D matrices access by indices
* *Performance* Use of non-temporal stores for large matrices
* *Performance*: Forced alignment of matrices
* *Performance*: Force basic padding of vectors
* *Performance*: Better thread reuse
* *Performance*: Faster dot product
* *Performance*: Faster batched outer product
* *Performance*: Better usage of FMA
* *Performance*: SSE/AVX double-precision exponentiation
* *Performance*: Much faster Probabilistic Max Pooling
* *Feature* Pooling with stride is now supported
* *Feature*: Custom fast and dyn matrices support
* *Feature* Matrices and vectors slices view
* *Feature* Deeper pooling support
* *Misc* Lots of small fixes
* *Misc* Reduced duplications in the code base
* *Misc* Simplifications of the iterators to DMA expressions
* *Misc* Faster compilation of the test cases
* *Misc* Generalized SSE/AVX versions into VEC versions

ETL 1.0 - 02.09.2016
++++++++++++++++++++

Initial version (was rolling released before) with the following main features:

* Smart Expression Templates
* Matrix and vector (runtime-sized and compile-time-sized)
* Simple element-wise operations
* Reductions (sum, mean, max, ...)
* Unary operations (sigmoid, log, exp, abs, ...)
* Matrix multiplication
* Convolution (1D and 2D and higher variations)
* Max Pooling
* Fast Fourrier Transform
* Use of SSE/AVX to speed up operations
* Use of BLAS/MKL/CUBLAS/CUFFT/CUDNN libraries to speed up operations
* Symmetric matrix adapter (experimental)
* Sparse matrix (experimental)
