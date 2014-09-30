Expression Templates Library (ETL)
==================================

ETL is a header only library for cC+ that provides vector and
matrix classes with support for Expression Templates to perform very
efficient operations on them. 

At this time, the library support compile-time sized matrix and vector
and runtime-sized matrix and vector with all element-wise operations 
implemented. It also supports 1D and 2D convolution and matrix 
multiplication (naive algorithm). 

Usage
-----

The library is header-only and does not need to be built it at all,
you just have to include its header files. 

Data structures
***************

Several structures are available: 

* fast_matrix<T, Dim...>: A matrix of variadic size with elements of type T.
  This must be used when you know the size of the vector at compile-time. The
  number of dimensions can be anything. 
* dyn_matrix<T, D>: A matrix with element of type T. The size of the
  matrix can be set at runtime.  The matrix can have D dimensions.

There also exists typedefs for vectors: 

* fast_vector<T, Rows>
* dyn_vector<T>

You have to keep in mind that fast_matrix directly store its values inside it,
therefore, it can be very large and should rarely be stored on the stack. 

Element-wise operations
***********************

Classic element-wise operations can be done on vector and matrix as
if it was done on scalars. Matrices and vectors can also be
added,subtracted,divided, ... by scalars. 

.. code:: cpp

    etl::dyn_vector<double> a({1.0,2.0,3.0});
    etl::dyn_vector<double> b({3.0,2.0,1.0});

    etl::dyn_vector<double> c(1.4 * (a + b) / b + b + a / 1.2);


All the operations are only executed once the expression is
evaluated to construct the dyn_vector. 

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
* fflip: Flip the vector or the matrix horizontally and verticaly. It is the
  equivalent of hflip(vflip(x))
* sub: Return a sub part of the matrix. The first dimension is forced to a
  special value. It works with matrices of any dimension. 
* dim/row/col: Return a vector representing a sub part of a matrix (a row or a
  col)
* reshape: Interpet a vector as a matrix

Lazy evaluation
***************

All binary and unary operations are applied lazily, only when they are assigned
to a concrete vector or matrix class. 

The expression can be evaluated using the :code:`s(x)` function that returns a
concrete class (fast_matrix or dyn_matrix) based on the expression.

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
operation. 

It is possible to pass an expression rather than an data structure
to functions. Keep in mind that expression are lazy, therefore if
you pass a + b to a matrix multiplication, an addition will be run
each time an element is accessed, therefore, it is not often
efficient. 

Building
--------

This library is completely header-only, there is no need to build it.

The folder **include** must be included with the **-I** option. 

However, this library makes extensive use of C++11 and C++14,
therefore, a recent compiler is necessary to use it.  This library
has only been tested on CLang 3.4 and g++ 4.9.1. Moreover, this has
never been tested on Windows. 

If you have problems compiling this library, I'd be glad to help,
but I do not guarantee that this will work on every compiler. I
strongly expect it to not build under Visual Studio.

License
-------

This library is distributed under the terms of the MIT license, see `LICENSE`
file for details.
