Expression Templates Library (ETL) 1.3
======================================

|logo|    |coverage| |jenkins| |license| |doc|

.. |logo| image:: logo_small.png
.. |coverage| image:: https://img.shields.io/sonar/https/sonar.baptiste-wicht.ch/etl/coverage.svg
.. |jenkins| image:: https://img.shields.io/jenkins/s/https/jenkins.baptiste-wicht.ch/etl.svg
.. |license| image:: https://img.shields.io/github/license/mashape/apistatus.svg
.. |doc| image:: https://codedocs.xyz/wichtounet/etl.svg
   :target: https://codedocs.xyz/wichtounet/etl/

ETL is a header only library for C++ that provides vector and matrix classes
with support for Expression Templates to perform very efficient operations on
them.

At this time, the library support compile-time sized matrix and vector and
runtime-sized matrix and vector with all element-wise operations implemented. It
also supports 1D and 2D convolution, matrix multiplication (naive algorithm and
Strassen) and FFT.

You can clone this repository directly to get all ETL features. I advice using
it as a submodule of your current project, but you can install it anywhere you
like. There are several branches you can chose from

* *master*: The main development branch
* *stable*: The last stable version

You can also access by tag to a fixed version such as the tag *1.0*.

Usage
-----

The `Reference Documentation <https://github.com/wichtounet/etl/wiki>`_ is always available on
the wiki. This document contains the most basic information that
should be enough to get you started.

The library is header-only and does not need to be built it at all,
you just have to include its header files.

Most of the headers are not meant to be included directly inside
a program. Here are the header that are made to be included:

* etl.hpp: Contains all the features of the library
* etl_light.hpp: Contains the basic features of the library (no matrix multiplication, no convolution, no FFT)

You should always include one of these headers in your program. You
should never include any other header from the library.

Data structures
***************

Several data structures are available:

* fast_matrix<T, Dim...>: A matrix of variadic size with elements of type T.
  This must be used when you know the size of the vector at compile-time. The
  number of dimensions can be anything. The data is stored is stored
  directly inside the matrix.
* fast_dyn_matrix<T, Dim...>: Variant of fast_matrix where the data
  is stored on the heap.
* dyn_matrix<T, D>: A matrix with element of type T. The size of the
  matrix can be set at runtime.  The matrix can have D dimensions.

There also exists typedefs for vectors:

* fast_vector<T, Rows>>
* fast_dyn_vector<T, Rows>>
* dyn_vector<T>

You have to keep in mind that fast_matrix directly store its values
inside it, therefore, it can be very large and should rarely be
stored on the stack. Moreover, that also makes it very expensive to
move and copy. This is why fast_dyn_matrix may be an interesting
alternative.

Element-wise operations
***********************

Classic element-wise operations can be done on vector and matrix as
if it was done on scalars. Matrices and vectors can also be
added,subtracted,divided, ... by scalars.

.. code:: cpp

    etl::dyn_vector<double> a{1.0,2.0,3.0};
    etl::dyn_vector<double> b{3.0,2.0,1.0};
    etl::dyn_vector<double> c;

    c = 1.4 * (a + b) / b + b + a / 1.2;


All the operations are only executed once the expression is
evaluated to be assigned to a data structure.

Unary operators
***************

Several unary operators are available. Each operation is performed
on every element of the vector or the matrix.

Available operators:

* log
* abs
* sign
* max/min
* sigmoid
* noise: Add standard normal noise to each element
* logistic_noise: Add normal noise of mean zero and variance sigmoid(x) to each
  element
* exp
* softplus
* bernoulli

Several transformations are also available:

* hflip: Flip the vector or the matrix horizontally
* vflip: Flip the vector or the matrix vertically
* fflip: Flip the vector or the matrix horizontally and vertically. It is the
  equivalent of hflip(vflip(x))
* sub: Return a sub part of the matrix. The first dimension is forced to a
  special value. It works with matrices of any dimension.
* dim/row/col: Return a vector representing a sub part of a matrix (a row or a
  col)
* reshape: Interpret a vector as a matrix

Reduction
*********

Several reduction functions are available:

* sum: Return the sum of a vector or matrix
* mean: Return the sum of a vector or matrix
* dot: Return the dot product of two vector or matrices

Functions
*********

The header *convolution.hpp* provides several convolution operations
both in 1D (vector) and 2D (matrix).

The header *mutiplication.hpp* provides the matrix multiplication
operation. mmul is the naive algorithm (ijk), which strassen_mmul implements
Strassen algorithm.

It is possible to pass an expression rather than an data structure
to functions. Keep in mind that expression are lazy, therefore if
you pass a + b to a matrix multiplication, an addition will be run
each time an element is accessed, therefore, it is not often
efficient.

Generators
**********

It is also possible to generate sequences of data and perform
operations on them.

For now, two generators are available:

* normal_generator: Generates real numbers distributed on a normal
  distribution
* sequence_generator(c=0): Generates numbers in sequence from c

All sequences are considered to have infinite size, therefore, they
can be used to initialize or modify any containers or expressions.

Building
--------

This library is completely header-only, there is no need to build it.

However, this library makes extensive use of C++11 and C++14,
therefore, a recent compiler is necessary to use it. This library is
tested on the following compilers:
 * GCC 6.3.0 and greater
 * CLang 3.9 and greater

If compilation does not work on one of these compilers, or produces warnings,
please open an issue on Github and I'll do my best to fix the issue.

The library has never been tested on Windows.

The folder **include** must be included with the **-I** option.

There are no link-time dependencies.

If you have problems compiling this library, I'd be glad to help,
but I do not guarantee that this will work on every compiler. I
strongly expect it to not build under Visual Studio.

License
-------

This library is distributed under the terms of the MIT license, see `LICENSE`
file for details.
