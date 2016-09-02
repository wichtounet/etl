ETL 1.0
+++++++

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
