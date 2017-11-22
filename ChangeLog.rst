ETL 1.3 - dev
*************

* *Feature* Support for embeddings and embedding gradients
* *Feature* Support for merging matrices together
* *Feature* Support for bias_batch_var_2d
* *Feature* GPU Support for uniform and normal generators
* *Performance* Vectorize hyperbolic functions
* *Bug* Fix fast_dyn_matrix with bool
* *Bug* Fix possible stack overflow with fast matrix and aliasing
* *Bug* Correctly handle aliasing in assignable (sub_view for instance)
* *Bug* Fix small compilation bug with sub_matrix

ETL 1.2 - 01.10.2017
********************

* *Feature* GPU support for basic expressions (such as c = 1.0 * b + d + e - 1.0)
* *Feature* GPU Support for unary and binary operators
* *Feature* Support for convolutions for matrices of different data types
* *Feature* Support for log2 / log10
* *Feature* Default selection of algorithms by default
* *Feature* Support for categorical cross entropy loss and error
* *Feature* Improve support for complex numbers and etl::complex
* *Performance* Improved performance of using parallel BLAS
* *Misc* Full cleanup of the traits
* *Misc* Use of variable templates (C++14) for the traits
* *Misc* Improved support for clang
* *Misc* Reduced compilation time for non-tests / non-benchmark code
* *Misc* Reduce durations of the tests
* *Misc* Preliminary C++17 if constexpr support
* *Bug* Fix bug in the GEMM kernel for CM = CM * CM
* *Bug* Vectorization bug for binary operations with different data types
* *Bug* GPU memory was not correctly handled when std::move is used

ETL 1.1 - 04.08.2017
++++++++++++++++++++

* *Performance* Better dispatching for alignment
* *Performance* Much faster multiplications between matrices of different major
* *Performance* Highly improved performed of multiplications with transpose
* *Performance* Vectorization of signed integer operations
* *Performance* Faster CPU convolutions
* *Performance* Better parallelization of convolutions
* *Performance* Much better GEMM/GEMV/GEVM kernels (when BLAS not available)
* *Performance* Reduced overhead for 3D/4D matrices access by indices
* *Performance* Use of non-temporal stores for large matrices
* *Performance* Forced alignment of matrices
* *Performance* Force basic padding of vectors
* *Performance* Better thread reuse
* *Performance* Faster dot product
* *Performance* Faster batched outer product
* *Performance* Better usage of FMA
* *Performance* SSE/AVX double-precision exponentiation
* *Performance* Much faster Pooling for various dimensions
* *Feature*: Sub matrices in 2D, 3D and 4D
* *Feature* Helpers for Machine Learning
* *Feature* Comparisons operators and functions equal, not_equal, almost_equal
* *Feature* Logical operators for boolean containers
* *Feature* Shuffle and noise can now operate on custom random engines
* *Feature* Pooling with stride is now supported
* *Feature* Custom fast and dyn matrices support
* *Feature* Matrices and vectors slices view
* *Feature* Deeper pooling support
* *Feature* bias_add (2D and 4D) (Machine Learning)
* *Feature* bias_batch_mean (2D and 4D) (Machine Learning)
* *Feature* Transposed convolution
* *GPU* Better usage of contexts
* *GPU* Pooling and Upsample support
* *GPU* batch_outer support
* *GPU* sigmoid and RELU and derivatives
* *GPU* Memory pool handling
* *GPU* Avoid a lot of temporaries
* *Misc* Reduced duplications in the code base
* *Misc* Simplifications of the iterators to DMA expressions
* *Misc* Faster compilation of the test cases
* *Misc* Generalized SSE/AVX versions into VEC versions
* *Misc* Reviewed completely temporary expressions
* *Bug* Lots of small fixes
* *Bug* Transpose on GPU was not working on column major matrix
* *Bug* 4D Pooling
* *Bug* Q/R Decomposition

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
