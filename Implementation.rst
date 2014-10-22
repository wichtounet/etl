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