Implementation notes
====================

Code use C++17 extensively.

For now code is made to be compiled with:

 * >=g++-9.3.0

Normally, everything should work fine with recent versions of clang. Due to
modern features that are being used, it is unlikely that everything works on
Windows and icc.

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

Coverage
--------

Coverage is quite low (less than 50%). The biggest problem now is that it is not
possible to merge the coverage statistics of several runs correctly. I do
believe that the real coverage is in fact higher than this. The merge process
only takes the maximum coverage from each profile, but does not merge individual
functions meaning that a lot of information is lost. One other problems is that
some files are polluting the results since they are not meant to be covered, for
instance SFINAE selection is never meant to be "executed" and no_vectorization
as well.

How to improve coverage:
 * Improve the merge of multiple coverage profile
 * Remove some files from the coverage analysis
 * Remove some lines from the coverage analysis (asserts) (not
   possible in sonar unfortunately)
 * Test more things, obviously

Notes
-----

There are too many corner cases in evaluation of expressions:
 * direct_evaluate
 * compound expressions
 * forced expressions
 * reductions

This should be improved
