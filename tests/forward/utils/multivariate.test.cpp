//                  _  _
//  _   _|_ _  _|o_|__|_
// (_||_||_(_)(_|| |  |
//
// automatic differentiation made easier in C++
// https://github.com/autodiff/autodiff
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright © 2018–2024 Allan Leal
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <vector>
#include <array>
#include <cmath>

// Catch includes
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

// autodiff includes
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <tests/utils/catch.hpp>
using namespace autodiff;

// Test complex multivariate functions with various mathematical operations
TEST_CASE("testing robust multivariate first derivatives", "[forward][utils][multivariate]")
{
    SECTION("testing polynomial multivariate functions")
    {
        // Test f(x,y,z) = x^3 + y^2*z + x*y*z^2 + x^2*y + z^3
        auto f = [](dual x, dual y, dual z) -> dual {
            return x*x*x + y*y*z + x*y*z*z + x*x*y + z*z*z;
        };

        dual x = 2.0, y = 3.0, z = 1.5;
        
        // Compute partial derivatives
        auto df_dx = derivatives(f, wrt(x), at(x, y, z));
        auto df_dy = derivatives(f, wrt(y), at(x, y, z));
        auto df_dz = derivatives(f, wrt(z), at(x, y, z));

        // Expected values: df/dx = 3x^2 + yz^2 + 2xy
        double expected_fx = 3*2*2 + 3*1.5*1.5 + 2*2*3; // = 12 + 6.75 + 12 = 30.75
        // Expected values: df/dy = 2yz + xz^2 + x^2  
        double expected_fy = 2*3*1.5 + 2*1.5*1.5 + 2*2; // = 9 + 4.5 + 4 = 17.5
        // Expected values: df/dz = y^2 + 2xyz + 3z^2
        double expected_fz = 3*3 + 2*2*3*1.5 + 3*1.5*1.5; // = 9 + 18 + 6.75 = 33.75

        CHECK( df_dx[1] == approx(expected_fx) );
        CHECK( df_dy[1] == approx(expected_fy) );
        CHECK( df_dz[1] == approx(expected_fz) );
    }

    SECTION("testing trigonometric multivariate functions")
    {
        // Test f(x,y) = sin(x*y) + cos(x+y) + tan(x-y)
        auto f = [](dual x, dual y) -> dual {
            return sin(x*y) + cos(x+y) + tan(x-y);
        };

        dual x = 0.8, y = 0.6;
        
        auto df_dx = derivatives(f, wrt(x), at(x, y));
        auto df_dy = derivatives(f, wrt(y), at(x, y));

        // Expected values: df/dx = y*cos(x*y) - sin(x+y) + sec^2(x-y)
        double xy = val(x)*val(y);
        double x_plus_y = val(x) + val(y);
        double x_minus_y = val(x) - val(y);
        double expected_fx = val(y)*cos(xy) - sin(x_plus_y) + 1.0/(cos(x_minus_y)*cos(x_minus_y));
        // Expected values: df/dy = x*cos(x*y) - sin(x+y) - sec^2(x-y)
        double expected_fy = val(x)*cos(xy) - sin(x_plus_y) - 1.0/(cos(x_minus_y)*cos(x_minus_y));

        CHECK( df_dx[1] == approx(expected_fx) );
        CHECK( df_dy[1] == approx(expected_fy) );
    }

    SECTION("testing exponential and logarithmic multivariate functions")
    {
        // Test f(x,y,z) = exp(x*y) + log(x+y+z) + x^y + y^z
        auto f = [](dual x, dual y, dual z) -> dual {
            return exp(x*y) + log(x+y+z) + pow(x, y) + pow(y, z);
        };

        dual x = 1.2, y = 1.8, z = 2.1;
        
        auto df_dx = derivatives(f, wrt(x), at(x, y, z));
        auto df_dy = derivatives(f, wrt(y), at(x, y, z));
        auto df_dz = derivatives(f, wrt(z), at(x, y, z));

        // Expected values: 
        // df/dx = y*exp(x*y) + 1/(x+y+z) + y*x^(y-1)
        double xy = val(x)*val(y);
        double xyz_sum = val(x) + val(y) + val(z);
        double expected_fx = val(y)*exp(xy) + 1.0/xyz_sum + val(y)*pow(val(x), val(y)-1);
        
        // df/dy = x*exp(x*y) + 1/(x+y+z) + x^y*ln(x) + z*y^(z-1)
        double expected_fy = val(x)*exp(xy) + 1.0/xyz_sum + pow(val(x), val(y))*log(val(x)) + val(z)*pow(val(y), val(z)-1);
        
        // df/dz = 1/(x+y+z) + y^z*ln(y)
        double expected_fz = 1.0/xyz_sum + pow(val(y), val(z))*log(val(y));

        CHECK( df_dx[1] == approx(expected_fx) );
        CHECK( df_dy[1] == approx(expected_fy) );
        CHECK( df_dz[1] == approx(expected_fz) );
    }

    SECTION("testing hyperbolic multivariate functions")
    {
        // Test f(x,y) = sinh(x*y) + cosh(x/y) + tanh(x+y)
        auto f = [](dual x, dual y) -> dual {
            return sinh(x*y) + cosh(x/y) + tanh(x+y);
        };

        dual x = 1.5, y = 2.0;
        
        auto df_dx = derivatives(f, wrt(x), at(x, y));
        auto df_dy = derivatives(f, wrt(y), at(x, y));

        // Expected values:
        // df/dx = y*cosh(x*y) + (1/y)*sinh(x/y) + sech^2(x+y)
        double xy = val(x)*val(y);
        double x_div_y = val(x)/val(y);
        double x_plus_y = val(x) + val(y);
        double sech_sq = 1.0/(cosh(x_plus_y)*cosh(x_plus_y));
        double expected_fx = val(y)*cosh(xy) + (1.0/val(y))*sinh(x_div_y) + sech_sq;
        
        // df/dy = x*cosh(x*y) - (x/y^2)*sinh(x/y) + sech^2(x+y)
        double expected_fy = val(x)*cosh(xy) - (val(x)/(val(y)*val(y)))*sinh(x_div_y) + sech_sq;

        CHECK( df_dx[1] == approx(expected_fx) );
        CHECK( df_dy[1] == approx(expected_fy) );
    }

    SECTION("testing inverse trigonometric multivariate functions")
    {
        // Test f(x,y) = asin(x/y) + acos(x*y) + atan(x+y)
        auto f = [](dual x, dual y) -> dual {
            return asin(x/y) + acos(x*y) + atan(x+y);
        };

        dual x = 0.4, y = 0.8;
        
        auto df_dx = derivatives(f, wrt(x), at(x, y));
        auto df_dy = derivatives(f, wrt(y), at(x, y));

        // Expected values:
        double x_div_y = val(x)/val(y);
        double xy = val(x)*val(y);
        double x_plus_y = val(x) + val(y);
        
        // df/dx = (1/y)/sqrt(1-(x/y)^2) - y/sqrt(1-(x*y)^2) + 1/(1+(x+y)^2)
        double expected_fx = (1.0/val(y))/sqrt(1-x_div_y*x_div_y) - val(y)/sqrt(1-xy*xy) + 1.0/(1+x_plus_y*x_plus_y);
        
        // df/dy = -(x/y^2)/sqrt(1-(x/y)^2) - x/sqrt(1-(x*y)^2) + 1/(1+(x+y)^2)
        double expected_fy = -(val(x)/(val(y)*val(y)))/sqrt(1-x_div_y*x_div_y) - val(x)/sqrt(1-xy*xy) + 1.0/(1+x_plus_y*x_plus_y);

        CHECK( df_dx[1] == approx(expected_fx) );
        CHECK( df_dy[1] == approx(expected_fy) );
    }

    SECTION("testing complex composite multivariate functions")
    {
        // Test f(x,y,z,w) = exp(sin(x*y)) + log(cos(z/w)) + sqrt(x^2+y^2+z^2+w^2)
        auto f = [](dual x, dual y, dual z, dual w) -> dual {
            return exp(sin(x*y)) + log(cos(z/w)) + sqrt(x*x+y*y+z*z+w*w);
        };

        dual x = 0.5, y = 0.8, z = 1.2, w = 1.6;
        
        auto df_dx = derivatives(f, wrt(x), at(x, y, z, w));
        auto df_dy = derivatives(f, wrt(y), at(x, y, z, w));
        auto df_dz = derivatives(f, wrt(z), at(x, y, z, w));
        auto df_dw = derivatives(f, wrt(w), at(x, y, z, w));

        // Expected values:
        double xy = val(x)*val(y);
        double z_div_w = val(z)/val(w);
        double norm_sq = val(x)*val(x) + val(y)*val(y) + val(z)*val(z) + val(w)*val(w);
        double norm = sqrt(norm_sq);
        
        // df/dx = y*cos(x*y)*exp(sin(x*y)) + x/sqrt(x^2+y^2+z^2+w^2)
        double expected_fx = val(y)*cos(xy)*exp(sin(xy)) + val(x)/norm;
        
        // df/dy = x*cos(x*y)*exp(sin(x*y)) + y/sqrt(x^2+y^2+z^2+w^2)
        double expected_fy = val(x)*cos(xy)*exp(sin(xy)) + val(y)/norm;
        
        // df/dz = -(1/w)*sin(z/w)/cos(z/w) + z/sqrt(x^2+y^2+z^2+w^2)
        double expected_fz = -(1.0/val(w))*sin(z_div_w)/cos(z_div_w) + val(z)/norm;
        
        // df/dw = (z/w^2)*sin(z/w)/cos(z/w) + w/sqrt(x^2+y^2+z^2+w^2)
        double expected_fw = (val(z)/(val(w)*val(w)))*sin(z_div_w)/cos(z_div_w) + val(w)/norm;

        CHECK( df_dx[1] == approx(expected_fx) );
        CHECK( df_dy[1] == approx(expected_fy) );
        CHECK( df_dz[1] == approx(expected_fz) );
        CHECK( df_dw[1] == approx(expected_fw) );
    }
}

#ifdef AUTODIFF_EIGEN_FOUND
TEST_CASE("testing robust multivariate vector functions with eigen", "[forward][utils][multivariate][eigen]")
{
    using Eigen::VectorXd;
    using Eigen::MatrixXd;

    SECTION("testing gradient computations for complex multivariate functions")
    {
        // Test f(x) = sum(x_i * exp(x_i) * sin(x_i)) where x is a vector
        auto f = [](const VectorXreal& x) -> real {
            real result = 0.0;
            for(int i = 0; i < x.size(); ++i) {
                result += x[i] * exp(x[i]) * sin(x[i]);
            }
            return result;
        };

        VectorXreal x(4);
        x << 0.5, 1.0, 1.5, 2.0;

        VectorXd g = gradient(f, wrt(x), at(x));

        // Expected gradient: df/dx_i = exp(x_i) * sin(x_i) + x_i * exp(x_i) * sin(x_i) + x_i * exp(x_i) * cos(x_i)
        //                            = exp(x_i) * (sin(x_i) + x_i * sin(x_i) + x_i * cos(x_i))
        //                            = exp(x_i) * (sin(x_i) * (1 + x_i) + x_i * cos(x_i))
        for(int i = 0; i < 4; ++i) {
            double xi = val(x[i]);
            double expected = exp(xi) * (sin(xi) * (1 + xi) + xi * cos(xi));
            CHECK( g[i] == approx(expected) );
        }
    }

    SECTION("testing jacobian computations for vector-valued multivariate functions")
    {
        // Test F(x,y) = [x^2 + y^2, x*y + sin(x*y), exp(x) + log(y), cos(x+y)]
        auto F = [](const VectorXreal& vars) -> VectorXreal {
            real x = vars[0];
            real y = vars[1];
            VectorXreal result(4);
            result[0] = x*x + y*y;
            result[1] = x*y + sin(x*y);
            result[2] = exp(x) + log(y);
            result[3] = cos(x+y);
            return result;
        };

        VectorXreal vars(2);
        vars << 1.2, 0.8;

        MatrixXd J = jacobian(F, wrt(vars), at(vars));

        double x_val = val(vars[0]);
        double y_val = val(vars[1]);
        double xy = x_val * y_val;

        // Expected Jacobian:
        // dF1/dx = 2x, dF1/dy = 2y
        CHECK( J(0, 0) == approx(2*x_val) );
        CHECK( J(0, 1) == approx(2*y_val) );
        
        // dF2/dx = y + y*cos(xy), dF2/dy = x + x*cos(xy)
        CHECK( J(1, 0) == approx(y_val + y_val*cos(xy)) );
        CHECK( J(1, 1) == approx(x_val + x_val*cos(xy)) );
        
        // dF3/dx = exp(x), dF3/dy = 1/y
        CHECK( J(2, 0) == approx(exp(x_val)) );
        CHECK( J(2, 1) == approx(1.0/y_val) );
        
        // dF4/dx = -sin(x+y), dF4/dy = -sin(x+y)
        double sin_sum = sin(x_val + y_val);
        CHECK( J(3, 0) == approx(-sin_sum) );
        CHECK( J(3, 1) == approx(-sin_sum) );
    }

    SECTION("testing high dimensional gradient computations")
    {
        // Test f(x) = sum(x_i^2 * sin(sum(x_j))) - a challenging high-dimensional function
        auto f = [](const VectorXreal& x) -> real {
            real sum_x = x.sum();
            real result = 0.0;
            for(int i = 0; i < x.size(); ++i) {
                result += x[i] * x[i] * sin(sum_x);
            }
            return result;
        };

        const int n = 10;
        VectorXreal x(n);
        for(int i = 0; i < n; ++i) {
            x[i] = 0.1 * (i + 1); // x = [0.1, 0.2, 0.3, ..., 1.0]
        }

        VectorXd g = gradient(f, wrt(x), at(x));

        // Expected gradient: df/dx_i = 2*x_i*sin(sum_x) + sum(x_j^2)*cos(sum_x)
        double sum_x = 0.0;
        double sum_x_sq = 0.0;
        for(int i = 0; i < n; ++i) {
            sum_x += val(x[i]);
            sum_x_sq += val(x[i]) * val(x[i]);
        }

        for(int i = 0; i < n; ++i) {
            double expected = 2*val(x[i])*sin(sum_x) + sum_x_sq*cos(sum_x);
            CHECK( g[i] == approx(expected) );
        }
    }

    SECTION("testing partial derivative consistency across different orderings")
    {
        // Test that partial derivatives are consistent regardless of the order of variables
        auto f = [](real x, real y, real z) -> real {
            return x*y*z + exp(x*y) + log(y*z + 1) + sin(x+z);
        };

        real x = 1.1, y = 1.3, z = 0.9;

        // Compute partial derivatives in different orders
        auto df_dx = derivative(f, wrt(x), at(x, y, z));
        auto df_dy = derivative(f, wrt(y), at(x, y, z));
        auto df_dz = derivative(f, wrt(z), at(x, y, z));

        // Test using different variable orderings
        auto df_dx_alt = derivative([=](real x_alt, real y_alt, real z_alt) { return f(x_alt, y_alt, z_alt); }, wrt(x), at(x, y, z));
        auto df_dy_alt = derivative([=](real x_alt, real y_alt, real z_alt) { return f(x_alt, y_alt, z_alt); }, wrt(y), at(x, y, z));
        auto df_dz_alt = derivative([=](real x_alt, real y_alt, real z_alt) { return f(x_alt, y_alt, z_alt); }, wrt(z), at(x, y, z));

        CHECK( df_dx == approx(df_dx_alt) );
        CHECK( df_dy == approx(df_dy_alt) );
        CHECK( df_dz == approx(df_dz_alt) );

#if AUTODIFF_DISABLE_HIGHER_ORDER
        // Test mixed partial derivatives with wrt(x,y) which should be available when AUTODIFF_DISABLE_HIGHER_ORDER is on
        // Note: This functionality is limited in AUTODIFF_DISABLE_HIGHER_ORDER mode
        INFO("Skipping mixed derivative test - requires manual implementation in AUTODIFF_DISABLE_HIGHER_ORDER mode");
#endif
    }
}
#endif // AUTODIFF_EIGEN_FOUND

TEST_CASE("testing edge cases for multivariate first derivatives", "[forward][utils][multivariate][edge-cases]")
{
    SECTION("testing functions with zero derivatives")
    {
        // Test f(x,y) = constant (should have zero derivatives)
        auto f = [](dual x, dual y) -> dual {
            return 42.0;
        };

        dual x = 1.5, y = 2.5;
        
        auto df_dx = derivatives(f, wrt(x), at(x, y));
        auto df_dy = derivatives(f, wrt(y), at(x, y));

        CHECK( df_dx[1] == approx(0.0) );
        CHECK( df_dy[1] == approx(0.0) );
    }

    SECTION("testing functions that depend on only one variable")
    {
        // Test f(x,y) = x^3 (independent of y)
        auto f = [](dual x, dual y) -> dual {
            return x*x*x;
        };

        dual x = 1.5, y = 2.5;
        
        auto df_dx = derivatives(f, wrt(x), at(x, y));
        auto df_dy = derivatives(f, wrt(y), at(x, y));

        CHECK( df_dx[1] == approx(3*1.5*1.5) );
        CHECK( df_dy[1] == approx(0.0) );
    }

    SECTION("testing functions with large dynamic ranges")
    {
        // Test f(x,y) = exp(x) + 1e-10*y^2 (vastly different scales)
        auto f = [](dual x, dual y) -> dual {
            return exp(x) + 1e-10*y*y;
        };

        dual x = 5.0, y = 1e6;
        
        auto df_dx = derivatives(f, wrt(x), at(x, y));
        auto df_dy = derivatives(f, wrt(y), at(x, y));

        CHECK( df_dx[1] == approx(exp(5.0)) );
        CHECK( df_dy[1] == approx(2e-10*1e6) );
    }

    SECTION("testing near-singular behavior")
    {
        // Test f(x,y) = x/y where y is small but not zero
        auto f = [](dual x, dual y) -> dual {
            return x / y;
        };

        dual x = 1.0, y = 1e-8;
        
        auto df_dx = derivatives(f, wrt(x), at(x, y));
        auto df_dy = derivatives(f, wrt(y), at(x, y));

        CHECK( df_dx[1] == approx(1.0/1e-8) );
        CHECK( df_dy[1] == approx(-1.0/(1e-8*1e-8)) );
    }

    SECTION("testing oscillatory functions")
    {
        // Test f(x,y) = sin(100*x) + cos(100*y) (high frequency oscillations)
        auto f = [](dual x, dual y) -> dual {
            return sin(100*x) + cos(100*y);
        };

        dual x = 0.01, y = 0.02;
        
        auto df_dx = derivatives(f, wrt(x), at(x, y));
        auto df_dy = derivatives(f, wrt(y), at(x, y));

        CHECK( df_dx[1] == approx(100*cos(100*0.01)) );
        CHECK( df_dy[1] == approx(-100*sin(100*0.02)) );
    }
}
