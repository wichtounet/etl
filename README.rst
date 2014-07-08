Expression Templates Library (ETL)
==================================

ETL is a small header only library for c++ that provides vector and
matrix classes with support for Expression Templates to perform very
efficient operations on them. 

At this time, the library support statically sized matrix and vector
with all element-wise operations implemented. It also supports 1D
and 2D convolution and matrix multiplication (naive algorithm). 

Usage
-----

The library is header-only and does not need to be built it at all,
you just have to include its header files. 

Data structures
***************

Several structures are available: 

* fast_vector<T, Rows>: A vector of size Rows with elements of type
  T. This must be used when you know the size of the vector at
  compile-time. 
* dyn_vector<T>: A vector with element of type T. The size of the
  vector can be set at runtime. 
* fast_matrix<T, Rows,Columns>: A matrix of size Rows x Columns with
  elements of type T. This must be used when you know the size of
  the matrix at compile-time. 
* dyn_vector<T>: A matrix with element of type T. The size of the
  matrix can be set at runtime. 

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

Reduction
*********

Several reduction functions are available:

* sum: Return the sum of a vector or matrix
* mean: Return the sum of a vector or matrix

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

License
-------

This library is distributed under the terms of the MIT license, see `LICENSE` file for details.
