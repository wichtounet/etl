//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

using outer_policy = NARY_POLICY(VALUES_POLICY(10, 50, 100, 500, 1000, 2000, 3000),
                                 VALUES_POLICY(10, 50, 100, 500, 1000, 2000, 3000));

using bias_add_policy = NARY_POLICY(VALUES_POLICY(10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
                                    VALUES_POLICY(128, 128, 128, 128, 128, 128, 128, 128, 128, 128),
                                    VALUES_POLICY(100, 100, 100, 100, 100, 100, 100, 100, 100, 100),
                                    VALUES_POLICY(100, 100, 100, 100, 100, 100, 100, 100, 100, 100));

using bias_add_2d_policy = NARY_POLICY(VALUES_POLICY(128, 128, 128, 128, 128, 128, 128, 128, 128, 128),
                                       VALUES_POLICY(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000));

using square_policy = NARY_POLICY(VALUES_POLICY(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000),
                                  VALUES_POLICY(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000));

using small_square_policy = NARY_POLICY(VALUES_POLICY(100, 150, 200, 250, 300, 350, 400, 450, 500),
                                        VALUES_POLICY(100, 150, 200, 250, 300, 350, 400, 450, 500));

using gemv_policy = NARY_POLICY(VALUES_POLICY(50, 100, 250, 500, 750, 1000, 2000, 3000, 4000, 5000, 6000),
                                VALUES_POLICY(50, 100, 250, 500, 750, 1000, 2000, 3000, 4000, 5000, 6000));
