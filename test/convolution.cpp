#include "catch.hpp"

#include "etl/fast_vector.hpp"
#include "etl/fast_matrix.hpp"
#include "etl/convolution.hpp"

//{{{ convolution_1d_full

TEST_CASE( "convolution_1d/full_1", "convolution_1d_full" ) {
    //>>> numpy.convolve([1,2,3],[0,1,0.5],'full')
    //array([ 0. ,  1. ,  2.5,  4. ,  1.5])

    etl::fast_vector<double, 3> a = {1.0, 2.0, 3.0};
    etl::fast_vector<double, 3> b = {0.0, 1.0, 0.5};
    etl::fast_vector<double, 5> c;

    etl::convolve_1d_full(a, b, c);

    REQUIRE(c[0] == Approx(0.0));
    REQUIRE(c[1] == Approx(1.0));
    REQUIRE(c[2] == Approx(2.5));
    REQUIRE(c[3] == Approx(4.0));
    REQUIRE(c[4] == Approx(1.5));
}

TEST_CASE( "convolution_1d/full_2", "convolution_1d_full" ) {
    //>>> numpy.convolve([1,2,3,4,5],[0.5,1,1.5],'full')
    //array([  0.5,   2. ,   5. ,   8. ,  11. ,  11. ,   7.5])

    etl::fast_vector<double, 5> a = {1.0, 2.0, 3.0, 4.0, 5.0};
    etl::fast_vector<double, 3> b = {0.5, 1.0, 1.5};
    etl::fast_vector<double, 7> c;

    etl::convolve_1d_full(a, b, c);

    REQUIRE(c[0] == Approx(0.5));
    REQUIRE(c[1] == Approx(2.0));
    REQUIRE(c[2] == Approx(5.0));
    REQUIRE(c[3] == Approx(8.0));
    REQUIRE(c[4] == Approx(11.0));
    REQUIRE(c[5] == Approx(11.0));
    REQUIRE(c[6] == Approx(7.5));
}

//}}}

//{{{ convolution_1d_same

TEST_CASE( "convolution_1d/same_1", "convolution_1d_same" ) {
    //>>> numpy.convolve([1,2,3],[0,1,0.5],'same')
    //array([ 1. ,  2.5,  4. ])

    etl::fast_vector<double, 3> a = {1.0, 2.0, 3.0};
    etl::fast_vector<double, 3> b = {0.0, 1.0, 0.5};
    etl::fast_vector<double, 3> c;

    etl::convolve_1d_same(a, b, c);

    REQUIRE(c[0] == Approx(1.0));
    REQUIRE(c[1] == Approx(2.5));
    REQUIRE(c[2] == Approx(4.0));
}

TEST_CASE( "convolution_1d/same_2", "convolution_1d_same" ) {
    //octave> conv([1.0,2.0,3.0,0.0,0.5,2.0],[0.0,0.5,1.0,0.0],"same")
    //2.00000   3.50000   3.00000   0.25000   1.50000   2.00000

    etl::fast_vector<double, 6> a = {1.0, 2.0, 3.0, 0.0, 0.5, 2.0};
    etl::fast_vector<double, 4> b = {0.0, 0.5, 1.0, 0.0};
    etl::fast_vector<double, 6> c;

    etl::convolve_1d_same(a, b, c);

    REQUIRE(c[0] == Approx(2.0));
    REQUIRE(c[1] == Approx(3.5));
    REQUIRE(c[2] == Approx(3.0));
    REQUIRE(c[3] == Approx(0.25));
    REQUIRE(c[4] == Approx(1.5));
    REQUIRE(c[5] == Approx(2.0));
}

//}}}

//{{{ convolution_1d_valid

TEST_CASE( "convolution_1d/valid_1", "convolution_1d_valid" ) {
    //>>> numpy.convolve([1,2,3],[0,1,0.5],'valid')
    //array([ 2.5])

    etl::fast_vector<double, 3> a = {1.0, 2.0, 3.0};
    etl::fast_vector<double, 3> b = {0.0, 1.0, 0.5};
    etl::fast_vector<double, 1> c;

    etl::convolve_1d_valid(a, b, c);

    REQUIRE(c[0] == 2.5);
}

TEST_CASE( "convolution_1d/valid_2", "convolution_1d_valid" ) {
    //>>> numpy.convolve([1,2,3,4,5],[0.5,1,1.5],'valid')
    //array([  5.,   8.,  11.])

    etl::fast_vector<double, 5> a = {1.0, 2.0, 3.0, 4.0, 5.0};
    etl::fast_vector<double, 3> b = {0.5, 1.0, 1.5};
    etl::fast_vector<double, 3> c;

    etl::convolve_1d_valid(a, b, c);

    REQUIRE(c[0] == 5.0);
    REQUIRE(c[1] == 8.0);
    REQUIRE(c[2] == 11);
}

//}}}

//{{{ convolution_2d_full

TEST_CASE( "convolution_2d/full_1", "convolution_2d_full" ) {
    //>>> scipy.signal.convolve2d([[1,2,3],[0,1,1],[3,2,1]],[[2,0],[0.5,0.5]])
    //array([[ 2. ,  4. ,  6. ,  0. ],
    //       [ 0.5,  3.5,  4.5,  1.5],
    //       [ 6. ,  4.5,  3. ,  0.5],
    //       [ 1.5,  2.5,  1.5,  0.5]])

    etl::fast_matrix<double, 3, 3> a = {1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0};
    etl::fast_matrix<double, 2, 2> b = {2.0, 0.0, 0.5, 0.5};
    etl::fast_matrix<double, 4, 4> c;

    etl::convolve_2d_full(a, b, c);

    REQUIRE(c(0,0) == 2.0);
    REQUIRE(c(0,1) == 4.0);
    REQUIRE(c(0,2) == 6.0);
    REQUIRE(c(0,3) == 0.0);

    REQUIRE(c(1,0) == 0.5);
    REQUIRE(c(1,1) == 3.5);
    REQUIRE(c(1,2) == 4.5);
    REQUIRE(c(1,3) == 1.5);

    REQUIRE(c(2,0) == 6.0);
    REQUIRE(c(2,1) == 4.5);
    REQUIRE(c(2,2) == 3.0);
    REQUIRE(c(2,3) == 0.5);

    REQUIRE(c(3,0) == 1.5);
    REQUIRE(c(3,1) == 2.5);
    REQUIRE(c(3,2) == 1.5);
    REQUIRE(c(3,3) == 0.5);
}

TEST_CASE( "convolution_2d/full_2", "convolution_2d_full" ) {
    //>>> scipy.signal.convolve2d([[1,2],[0,1],[3,2]],[[2,0],[0.5,0.5]])
    //array([[ 2. ,  4. ,  0. ],
    //       [ 0.5,  3.5,  1. ],
    //       [ 6. ,  4.5,  0.5],
    //       [ 1.5,  2.5,  1. ]])

    etl::fast_matrix<double, 3, 2> a = {1.0, 2.0, 0.0, 1.0, 3.0, 2.0};
    etl::fast_matrix<double, 2, 2> b = {2.0, 0.0, 0.5, 0.5};
    etl::fast_matrix<double, 4, 3> c;

    etl::convolve_2d_full(a, b, c);

    REQUIRE(c(0,0) == 2.0);
    REQUIRE(c(0,1) == 4.0);
    REQUIRE(c(0,2) == 0.0);

    REQUIRE(c(1,0) == 0.5);
    REQUIRE(c(1,1) == 3.5);
    REQUIRE(c(1,2) == 1.0);

    REQUIRE(c(2,0) == 6.0);
    REQUIRE(c(2,1) == 4.5);
    REQUIRE(c(2,2) == 0.5);

    REQUIRE(c(3,0) == 1.5);
    REQUIRE(c(3,1) == 2.5);
    REQUIRE(c(3,2) == 1.0);
}

TEST_CASE( "convolution_2d/full_3", "convolution_2d_full" ) {
    //>>> scipy.signal.convolve2d([[1,2],[3,2]],[[2,1.0],[0.5,0.5]])
    //array([[ 2. ,  5. ,  2. ],
    //       [ 6.5,  8.5,  3. ],
    //       [ 1.5,  2.5,  1. ]])

    etl::fast_matrix<double, 2, 2> a = {1.0, 2.0, 3.0, 2.0};
    etl::fast_matrix<double, 2, 2> b = {2.0, 1.0, 0.5, 0.5};
    etl::fast_matrix<double, 3, 3> c;

    etl::convolve_2d_full(a, b, c);

    REQUIRE(c(0,0) == 2.0);
    REQUIRE(c(0,1) == 5.0);
    REQUIRE(c(0,2) == 2.0);

    REQUIRE(c(1,0) == 6.5);
    REQUIRE(c(1,1) == 8.5);
    REQUIRE(c(1,2) == 3.0);

    REQUIRE(c(2,0) == 1.5);
    REQUIRE(c(2,1) == 2.5);
    REQUIRE(c(2,2) == 1.0);
}

//}}}

//{{{ convolution_2d_same

TEST_CASE( "convolution_2d/same_1", "convolution_2d_same" ) {
    //octave:7> conv2([1,2,3;0,1,1;3,2,1],[2,0;0.5,0.5], 'same')
    //   3.50000   4.50000   1.50000
    //   4.50000   3.00000   0.50000
    //   2.50000   1.50000   0.50000

    etl::fast_matrix<double, 3, 3> a = {1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0};
    etl::fast_matrix<double, 2, 2> b = {2.0, 0.0, 0.5, 0.5};
    etl::fast_matrix<double, 3, 3> c;

    etl::convolve_2d_same(a, b, c);

    REQUIRE(c(0,0) == 3.5);
    REQUIRE(c(0,1) == 4.5);
    REQUIRE(c(0,2) == 1.5);

    REQUIRE(c(1,0) == 4.5);
    REQUIRE(c(1,1) == 3.0);
    REQUIRE(c(1,2) == 0.5);

    REQUIRE(c(2,0) == 2.5);
    REQUIRE(c(2,1) == 1.5);
    REQUIRE(c(2,2) == 0.5);
}

TEST_CASE( "convolution_2d/same_2", "convolution_2d_same" ) {
    //octave:8> conv2([1,2;0,1;3,2],[2,0;0.5,0.5], 'same')
    //   3.50000   1.00000
    //   4.50000   0.50000
    //   2.50000   1.00000

    etl::fast_matrix<double, 3, 2> a = {1.0, 2.0, 0.0, 1.0, 3.0, 2.0};
    etl::fast_matrix<double, 2, 2> b = {2.0, 0.0, 0.5, 0.5};
    etl::fast_matrix<double, 3, 2> c;

    etl::convolve_2d_same(a, b, c);

    REQUIRE(c(0,0) == 3.5);
    REQUIRE(c(0,1) == 1.0);

    REQUIRE(c(1,0) == 4.5);
    REQUIRE(c(1,1) == 0.5);

    REQUIRE(c(2,0) == 2.5);
    REQUIRE(c(2,1) == 1.0);
}

TEST_CASE( "convolution_2d/same_3", "convolution_2d_same" ) {
    //>>> scipy.signal.convolve2d([[1,2],[3,2]],[[2,1.0],[0.5,0.5]],'same')
    //array([[ 2. ,  5. ],
    //       [ 6.5,  8.5]])

    etl::fast_matrix<double, 2, 2> a = {1.0, 2.0, 3.0, 2.0};
    etl::fast_matrix<double, 2, 2> b = {2.0, 1.0, 0.5, 0.5};
    etl::fast_matrix<double, 2, 2> c;

    etl::convolve_2d_same(a, b, c);

    REQUIRE(c(0,0) == 8.5);
    REQUIRE(c(0,1) == 3.0);

    REQUIRE(c(1,0) == 2.5);
    REQUIRE(c(1,1) == 1.0);
}

//}}}

//{{{ convolution_2d_valid

TEST_CASE( "convolution_2d/valid_1", "convolution_2d_valid" ) {
    //>>> scipy.signal.convolve2d([[1,2,3],[0,1,1],[3,2,1]],[[2,0],[0.5,0.5]],'valid')
    //array([[ 3.5,  4.5],
    //       [ 4.5,  3. ]])

    etl::fast_matrix<double, 3, 3> a = {1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0};
    etl::fast_matrix<double, 2, 2> b = {2.0, 0.0, 0.5, 0.5};
    etl::fast_matrix<double, 2, 2> c;

    etl::convolve_2d_valid(a, b, c);

    REQUIRE(c(0,0) == 3.5);
    REQUIRE(c(0,1) == 4.5);

    REQUIRE(c(1,0) == 4.5);
    REQUIRE(c(1,1) == 3.0);
}

TEST_CASE( "convolution_2d/valid_2", "convolution_2d_valid" ) {
    //>>> scipy.signal.convolve2d([[1,2],[0,1],[3,2]],[[2,0],[0.5,0.5]],'valid')
    //array([[ 3.5],
    //       [ 4.5]])

    etl::fast_matrix<double, 3, 2> a = {1.0, 2.0, 0.0, 1.0, 3.0, 2.0};
    etl::fast_matrix<double, 2, 2> b = {2.0, 0.0, 0.5, 0.5};
    etl::fast_matrix<double, 2, 1> c;

    etl::convolve_2d_valid(a, b, c);

    REQUIRE(c(0,0) == 3.5);
    REQUIRE(c(1,0) == 4.5);
}

TEST_CASE( "convolution_2d/valid_3", "convolution_2d_valid" ) {
    //scipy.signal.convolve2d([[1,2],[3,2]],[[2,1.0],[0.5,0.5]],'valid')
    //array([[ 8.5]])

    etl::fast_matrix<double, 2, 2> a = {1.0, 2.0, 3.0, 2.0};
    etl::fast_matrix<double, 2, 2> b = {2.0, 1.0, 0.5, 0.5};
    etl::fast_matrix<double, 1, 1> c;

    etl::convolve_2d_valid(a, b, c);

    REQUIRE(c(0,0) == 8.5);
}

//}}}