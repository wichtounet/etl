Implementation notes
====================

Code use C++14 extensively.

For now code is made to be compiled with:

 * >=clang 3.4
 * >=g++-4.9.1

Due to the limitations in g++, these features cannot be used:

 * Relaxed constexpr functions
 * Variable templates

Due to a bug in CLang, what I call universal enable_if cannot be used and the
dummy value must be used instead.

Compile-Time
------------

The time to compile expressions it currently not great. It is
starting to take quite long.

There are several possible solutions

 * Type erasure of some sort. For instance, some algorithms only
   need some sizes and data pointer, this would save some
   instantiations, but could mean major changes
 * Simplify the interface template by removing some enable_if and
   making some things not template.

However, some things are harder than expected. For instance, it is
not possible to remove SFINAE on the operator+ funtion because
otherwise it would match etl::complex or iterators from dyn matrix.
This because forwarding is "too perfect" and because ADL is used.

Notes
-----

There are too many corner cases in evaluation of expressions:
 * direct_evaluate
 * compound expressions
 * forced expressions
 * reductions
