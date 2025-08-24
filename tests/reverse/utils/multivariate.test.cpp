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
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
#include <tests/utils/catch.hpp>
using namespace autodiff;

/// Convenient function used in the tests to calculate the derivative of a variable y with respect to a variable x.
inline auto grad(const var& y, var& x)
{
    auto g = derivatives(y, wrt(x));
    return val(g[0]);
}

// Test complex multivariate functions with various mathematical operations in reverse mode
TEST_CASE("testing robust multivariate first derivatives - reverse mode", "[reverse][utils][multivariate]")
{
    SECTION("testing polynomial multivariate functions")
    {
        // Test f(x,y,z) = x^3 + y^2*z + x*y*z^2 + x^2*y + z^3
        var x = 2.0, y = 3.0, z = 1.5;
        var f = x*x*x + y*y*z + x*y*z*z + x*x*y + z*z*z;
        
        // Compute partial derivatives
        auto df_dx = grad(f, x);
        auto df_dy = grad(f, y);
        auto df_dz = grad(f, z);

        // Expected values: df/dx = 3x^2 + yz^2 + 2xy
        double expected_fx = 3*2*2 + 3*1.5*1.5 + 2*2*3; // = 12 + 6.75 + 12 = 30.75
        // Expected values: df/dy = 2yz + xz^2 + x^2  
        double expected_fy = 2*3*1.5 + 2*1.5*1.5 + 2*2; // = 9 + 4.5 + 4 = 17.5
        // Expected values: df/dz = y^2 + 2xyz + 3z^2
        double expected_fz = 3*3 + 2*2*3*1.5 + 3*1.5*1.5; // = 9 + 18 + 6.75 = 33.75

        CHECK( df_dx == approx(expected_fx) );
        CHECK( df_dy == approx(expected_fy) );
        CHECK( df_dz == approx(expected_fz) );
    }

    SECTION("testing trigonometric multivariate functions")
    {
        // Test f(x,y) = sin(x*y) + cos(x+y) + tan(x-y)
        var x = 0.8, y = 0.6;
        var f = sin(x*y) + cos(x+y) + tan(x-y);
        
        auto df_dx = grad(f, x);
        auto df_dy = grad(f, y);

        // Expected values: df/dx = y*cos(x*y) - sin(x+y) + sec^2(x-y)
        double xy = val(x)*val(y);
        double x_plus_y = val(x) + val(y);
        double x_minus_y = val(x) - val(y);
        double expected_fx = val(y)*cos(xy) - sin(x_plus_y) + 1.0/(cos(x_minus_y)*cos(x_minus_y));
        // Expected values: df/dy = x*cos(x*y) - sin(x+y) - sec^2(x-y)
        double expected_fy = val(x)*cos(xy) - sin(x_plus_y) - 1.0/(cos(x_minus_y)*cos(x_minus_y));

        CHECK( df_dx == approx(expected_fx) );
        CHECK( df_dy == approx(expected_fy) );
    }

    SECTION("testing exponential and logarithmic multivariate functions")
    {
        // Test f(x,y,z) = exp(x*y) + log(x+y+z) + x^y + y^z
        var x = 1.2, y = 1.8, z = 2.1;
        var f = exp(x*y) + log(x+y+z) + pow(x, y) + pow(y, z);
        
        auto df_dx = grad(f, x);
        auto df_dy = grad(f, y);
        auto df_dz = grad(f, z);

        // Expected values: 
        // df/dx = y*exp(x*y) + 1/(x+y+z) + y*x^(y-1)
        double xy = val(x)*val(y);
        double xyz_sum = val(x) + val(y) + val(z);
        double expected_fx = val(y)*exp(xy) + 1.0/xyz_sum + val(y)*pow(val(x), val(y)-1);
        
        // df/dy = x*exp(x*y) + 1/(x+y+z) + x^y*ln(x) + z*y^(z-1)
        double expected_fy = val(x)*exp(xy) + 1.0/xyz_sum + pow(val(x), val(y))*log(val(x)) + val(z)*pow(val(y), val(z)-1);
        
        // df/dz = 1/(x+y+z) + y^z*ln(y)
        double expected_fz = 1.0/xyz_sum + pow(val(y), val(z))*log(val(y));

        CHECK( df_dx == approx(expected_fx) );
        CHECK( df_dy == approx(expected_fy) );
        CHECK( df_dz == approx(expected_fz) );
    }

    SECTION("testing complex composite multivariate functions")
    {
        // Test f(x,y,z,w) = exp(sin(x*y)) + log(cos(z/w)) + sqrt(x^2+y^2+z^2+w^2)
        var x = 0.5, y = 0.8, z = 1.2, w = 1.6;
        var f = exp(sin(x*y)) + log(cos(z/w)) + sqrt(x*x+y*y+z*z+w*w);
        
        auto df_dx = grad(f, x);
        auto df_dy = grad(f, y);
        auto df_dz = grad(f, z);
        auto df_dw = grad(f, w);

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

        CHECK( df_dx == approx(expected_fx) );
        CHECK( df_dy == approx(expected_fy) );
        CHECK( df_dz == approx(expected_fz) );
        CHECK( df_dw == approx(expected_fw) );
    }

    SECTION("testing optimization-relevant multivariate functions")
    {
        // Test Rosenbrock function: f(x,y) = (a-x)^2 + b*(y-x^2)^2 with a=1, b=100
        var x = 0.5, y = 0.8;
        var a = 1.0, b = 100.0;
        var f = (a-x)*(a-x) + b*(y-x*x)*(y-x*x);
        
        auto df_dx = grad(f, x);
        auto df_dy = grad(f, y);

        // Expected values:
        // df/dx = -2*(a-x) + b*2*(y-x^2)*(-2*x) = -2*(a-x) - 4*b*x*(y-x^2)
        double expected_fx = -2*(val(a)-val(x)) - 4*val(b)*val(x)*(val(y)-val(x)*val(x));
        // df/dy = b*2*(y-x^2)
        double expected_fy = val(b)*2*(val(y)-val(x)*val(x));

        CHECK( df_dx == approx(expected_fx) );
        CHECK( df_dy == approx(expected_fy) );
    }

    SECTION("testing constrained optimization scenarios")
    {
        // Test Lagrangian-like function: f(x,y,λ) = x^2 + y^2 + λ*(x+y-1)
        var x = 0.3, y = 0.4, lambda = 2.5;
        var f = x*x + y*y + lambda*(x+y-1);
        
        auto df_dx = grad(f, x);
        auto df_dy = grad(f, y);
        auto df_dlambda = grad(f, lambda);

        // Expected values:
        // df/dx = 2*x + λ
        double expected_fx = 2*val(x) + val(lambda);
        // df/dy = 2*y + λ  
        double expected_fy = 2*val(y) + val(lambda);
        // df/dλ = x + y - 1
        double expected_flambda = val(x) + val(y) - 1;

        CHECK( df_dx == approx(expected_fx) );
        CHECK( df_dy == approx(expected_fy) );
        CHECK( df_dlambda == approx(expected_flambda) );
    }
}

#ifdef AUTODIFF_EIGEN_FOUND
TEST_CASE("testing robust multivariate vector functions with eigen - reverse mode", "[reverse][utils][multivariate][eigen]")
{
    using Eigen::VectorXd;
    using Eigen::MatrixXd;

    SECTION("testing gradient computations for complex multivariate functions")
    {
        // Test f(x) = sum(x_i * exp(x_i) * sin(x_i)) where x is a vector
        VectorXvar x(4);
        x << 0.5, 1.0, 1.5, 2.0;

        var result = 0.0;
        for(int i = 0; i < x.size(); ++i) {
            result += x[i] * exp(x[i]) * sin(x[i]);
        }

        VectorXd g = gradient(result, x);

        // Expected gradient: df/dx_i = exp(x_i) * (sin(x_i) * (1 + x_i) + x_i * cos(x_i))
        for(int i = 0; i < 4; ++i) {
            double xi = val(x[i]);
            double expected = exp(xi) * (sin(xi) * (1 + xi) + xi * cos(xi));
            CHECK( g[i] == approx(expected) );
        }
    }

    SECTION("testing high dimensional gradient computations")
    {
        // Test f(x) = sum(x_i^2 * sin(sum(x_j))) - a challenging high-dimensional function
        const int n = 8;
        VectorXvar x(n);
        for(int i = 0; i < n; ++i) {
            x[i] = 0.1 * (i + 1); // x = [0.1, 0.2, 0.3, ..., 0.8]
        }

        var sum_x = x.sum();
        var result = 0.0;
        for(int i = 0; i < n; ++i) {
            result += x[i] * x[i] * sin(sum_x);
        }

        VectorXd g = gradient(result, x);

        // Expected gradient: df/dx_i = 2*x_i*sin(sum_x) + sum(x_j^2)*cos(sum_x)
        double sum_x_val = 0.0;
        double sum_x_sq = 0.0;
        for(int i = 0; i < n; ++i) {
            sum_x_val += val(x[i]);
            sum_x_sq += val(x[i]) * val(x[i]);
        }

        for(int i = 0; i < n; ++i) {
            double expected = 2*val(x[i])*sin(sum_x_val) + sum_x_sq*cos(sum_x_val);
            CHECK( g[i] == approx(expected) );
        }
    }

    SECTION("testing neural network-like functions")
    {
        // Test a simple neural network layer: f(x,W,b) = sum(tanh(W*x + b))
        const int input_dim = 3;
        const int hidden_dim = 4;
        
        VectorXvar x(input_dim);
        x << 0.5, -0.3, 0.8;
        
        MatrixXvar W(hidden_dim, input_dim);
        W << 0.1, 0.2, -0.1,
             0.3, -0.2, 0.4,
             -0.1, 0.5, 0.2,
             0.2, -0.3, -0.1;
        
        VectorXvar b(hidden_dim);
        b << 0.1, -0.2, 0.3, -0.1;

        // Forward pass: z = W*x + b, a = tanh(z), f = sum(a)
        VectorXvar z = W * x + b;
        var result = 0.0;
        for(int i = 0; i < hidden_dim; ++i) {
            result += tanh(z[i]);
        }

        // Test gradients with respect to input x
        VectorXd grad_x = gradient(result, x);
        
        // Expected gradient computation is complex but can be verified numerically
        // df/dx_j = sum_i (sech^2(z_i) * W_ij)
        for(int j = 0; j < input_dim; ++j) {
            double expected = 0.0;
            for(int i = 0; i < hidden_dim; ++i) {
                double z_val = val(z[i]);
                double sech_sq = 1.0 / (cosh(z_val) * cosh(z_val));
                expected += sech_sq * val(W(i, j));
            }
            CHECK( grad_x[j] == approx(expected) );
        }
    }

    SECTION("testing chain rule with multiple intermediate variables")
    {
        // Test f(x,y) = sin(exp(x*y)) + cos(log(x^2 + y^2))
        var x = 1.2, y = 0.8;
        
        // Intermediate variables
        var xy = x * y;
        var exp_xy = exp(xy);
        var sin_exp_xy = sin(exp_xy);
        
        var x_sq_plus_y_sq = x*x + y*y;
        var log_sum = log(x_sq_plus_y_sq);
        var cos_log_sum = cos(log_sum);
        
        var f = sin_exp_xy + cos_log_sum;
        
        auto df_dx = grad(f, x);
        auto df_dy = grad(f, y);

        // Expected values using chain rule:
        // df/dx = cos(exp(x*y)) * exp(x*y) * y + (-sin(log(x^2+y^2))) * (1/(x^2+y^2)) * 2*x
        double xy_val = val(x) * val(y);
        double exp_xy_val = exp(xy_val);
        double x_sq_plus_y_sq_val = val(x)*val(x) + val(y)*val(y);
        double log_sum_val = log(x_sq_plus_y_sq_val);
        
        double expected_fx = cos(exp_xy_val) * exp_xy_val * val(y) + 
                            (-sin(log_sum_val)) * (1.0/x_sq_plus_y_sq_val) * 2*val(x);
        double expected_fy = cos(exp_xy_val) * exp_xy_val * val(x) + 
                            (-sin(log_sum_val)) * (1.0/x_sq_plus_y_sq_val) * 2*val(y);

        CHECK( df_dx == approx(expected_fx) );
        CHECK( df_dy == approx(expected_fy) );
    }
}
#endif // AUTODIFF_EIGEN_FOUND

TEST_CASE("testing edge cases for multivariate first derivatives - reverse mode", "[reverse][utils][multivariate][edge-cases]")
{
    SECTION("testing functions with zero derivatives")
    {
        // Test f(x,y) = constant (should have zero derivatives)
        var x = 1.5, y = 2.5;
        var f = 42.0;
        
        auto df_dx = grad(f, x);
        auto df_dy = grad(f, y);

        CHECK( df_dx == approx(0.0) );
        CHECK( df_dy == approx(0.0) );
    }

    SECTION("testing functions that depend on only one variable")
    {
        // Test f(x,y) = x^3 (independent of y)
        var x = 1.5, y = 2.5;
        var f = x*x*x;
        
        auto df_dx = grad(f, x);
        auto df_dy = grad(f, y);

        CHECK( df_dx == approx(3*1.5*1.5) );
        CHECK( df_dy == approx(0.0) );
    }

    SECTION("testing precision with small numbers")
    {
        // Test f(x,y) = 1e-10*x^2 + 1e-12*y^3
        var x = 1e3, y = 1e4;
        var f = 1e-10*x*x + 1e-12*y*y*y;
        
        auto df_dx = grad(f, x);
        auto df_dy = grad(f, y);

        CHECK( df_dx == approx(2e-10*1e3) );
        CHECK( df_dy == approx(3e-12*1e4*1e4) );
    }

    SECTION("testing oscillatory functions with high frequency")
    {
        // Test f(x,y) = sin(1000*x) + cos(1000*y)
        var x = 0.001, y = 0.002;
        var f = sin(1000*x) + cos(1000*y);
        
        auto df_dx = grad(f, x);
        auto df_dy = grad(f, y);

        CHECK( df_dx == approx(1000*cos(1000*0.001)) );
        CHECK( df_dy == approx(-1000*sin(1000*0.002)) );
    }

    SECTION("testing performance with many variables")
    {
        // Test f(x) = sum(x_i^2) where x has many components
        const int n = 100;
        std::vector<var> x(n);
        for(int i = 0; i < n; ++i) {
            x[i] = 0.01 * i;
        }

        var f = 0.0;
        for(int i = 0; i < n; ++i) {
            f += x[i] * x[i];
        }

        // Test a few random derivatives
        auto df_dx0 = grad(f, x[0]);
        auto df_dx50 = grad(f, x[50]);
        auto df_dx99 = grad(f, x[99]);

        CHECK( df_dx0 == approx(2*0.01*0) );
        CHECK( df_dx50 == approx(2*0.01*50) );
        CHECK( df_dx99 == approx(2*0.01*99) );
    }
}
