//                  _  _
//  _   _|_ _  _|o_|__|_
// (_||_||_(_)(_|| |  automatic differentiation made easier in C++
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

#include <cmath>

// Catch includes
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

// autodiff includes
#include <autodiff/reverse/var.hpp>
#include <tests/utils/catch.hpp>
using namespace autodiff;

// Comprehensive test suite for reverse mode first-order derivatives
// This ensures that reverse mode functionality is properly tested when AUTODIFF_DISABLE_HIGHER_ORDER is enabled

TEST_CASE("testing reverse mode first-order mathematical functions", "[reverse][var][first-order-mathematical]")
{
    SECTION("testing basic arithmetic operations - reverse mode")
    {
        var x = 2.5, y = 1.5;
        
        // Addition
        var f1 = x + y;
        auto grad_x1 = derivatives(f1, autodiff::reverse::detail::wrt(x));
        auto grad_y1 = derivatives(f1, autodiff::reverse::detail::wrt(y));
        CHECK( val(f1) == approx(4.0) );
        CHECK( val(grad_x1[0]) == approx(1.0) );
        CHECK( val(grad_y1[0]) == approx(1.0) );
        
        // Subtraction
        var f2 = x - y;
        auto grad_x2 = derivatives(f2, autodiff::reverse::detail::wrt(x));
        auto grad_y2 = derivatives(f2, autodiff::reverse::detail::wrt(y));
        CHECK( val(f2) == approx(1.0) );
        CHECK( val(grad_x2[0]) == approx(1.0) );
        CHECK( val(grad_y2[0]) == approx(-1.0) );
        
        // Multiplication
        var f3 = x * y;
        auto grad_x3 = derivatives(f3, autodiff::reverse::detail::wrt(x));
        auto grad_y3 = derivatives(f3, autodiff::reverse::detail::wrt(y));
        CHECK( val(f3) == approx(3.75) );
        CHECK( val(grad_x3[0]) == approx(1.5) );  // df/dx = y
        CHECK( val(grad_y3[0]) == approx(2.5) );  // df/dy = x
        
        // Division
        var f4 = x / y;
        auto grad_x4 = derivatives(f4, autodiff::reverse::detail::wrt(x));
        auto grad_y4 = derivatives(f4, autodiff::reverse::detail::wrt(y));
        CHECK( val(f4) == approx(2.5/1.5) );
        CHECK( val(grad_x4[0]) == approx(1.0/1.5) );     // df/dx = 1/y
        CHECK( val(grad_y4[0]) == approx(-2.5/(1.5*1.5)) ); // df/dy = -x/y²
    }

    SECTION("testing trigonometric functions - reverse mode")
    {
        var x = 0.6;
        
        // sin function
        var f_sin = sin(x);
        auto grad_sin = derivatives(f_sin, autodiff::reverse::detail::wrt(x));
        CHECK( val(f_sin) == approx(sin(0.6)) );
        CHECK( val(grad_sin[0]) == approx(cos(0.6)) );
        
        // cos function
        var f_cos = cos(x);
        auto grad_cos = derivatives(f_cos, autodiff::reverse::detail::wrt(x));
        CHECK( val(f_cos) == approx(cos(0.6)) );
        CHECK( val(grad_cos[0]) == approx(-sin(0.6)) );
        
        // tan function
        var f_tan = tan(x);
        auto grad_tan = derivatives(f_tan, autodiff::reverse::detail::wrt(x));
        CHECK( val(f_tan) == approx(tan(0.6)) );
        CHECK( val(grad_tan[0]) == approx(1.0/(cos(0.6)*cos(0.6))) );
    }

    SECTION("testing inverse trigonometric functions - reverse mode")
    {
        var x = 0.4;  // Ensure |x| < 1 for asin and acos
        
        // asin function
        var f_asin = asin(x);
        auto grad_asin = derivatives(f_asin, autodiff::reverse::detail::wrt(x));
        CHECK( val(f_asin) == approx(asin(0.4)) );
        CHECK( val(grad_asin[0]) == approx(1.0/sqrt(1.0 - 0.4*0.4)) );
        
        // acos function
        var f_acos = acos(x);
        auto grad_acos = derivatives(f_acos, autodiff::reverse::detail::wrt(x));
        CHECK( val(f_acos) == approx(acos(0.4)) );
        CHECK( val(grad_acos[0]) == approx(-1.0/sqrt(1.0 - 0.4*0.4)) );
        
        // atan function
        var x_atan = 1.2;
        var f_atan = atan(x_atan);
        auto grad_atan = derivatives(f_atan, autodiff::reverse::detail::wrt(x_atan));
        CHECK( val(f_atan) == approx(atan(1.2)) );
        CHECK( val(grad_atan[0]) == approx(1.0/(1.0 + 1.2*1.2)) );
    }

    SECTION("testing hyperbolic functions - reverse mode")
    {
        var x = 0.7;
        
        // sinh function
        var f_sinh = sinh(x);
        auto grad_sinh = derivatives(f_sinh, autodiff::reverse::detail::wrt(x));
        CHECK( val(f_sinh) == approx(sinh(0.7)) );
        CHECK( val(grad_sinh[0]) == approx(cosh(0.7)) );
        
        // cosh function
        var f_cosh = cosh(x);
        auto grad_cosh = derivatives(f_cosh, autodiff::reverse::detail::wrt(x));
        CHECK( val(f_cosh) == approx(cosh(0.7)) );
        CHECK( val(grad_cosh[0]) == approx(sinh(0.7)) );
        
        // tanh function
        var f_tanh = tanh(x);
        auto grad_tanh = derivatives(f_tanh, autodiff::reverse::detail::wrt(x));
        CHECK( val(f_tanh) == approx(tanh(0.7)) );
        CHECK( val(grad_tanh[0]) == approx(1.0/(cosh(0.7)*cosh(0.7))) );
    }

    SECTION("testing exponential and logarithmic functions - reverse mode")
    {
        var x = 1.8;
        
        // exp function
        var f_exp = exp(x);
        auto grad_exp = derivatives(f_exp, autodiff::reverse::detail::wrt(x));
        CHECK( val(f_exp) == approx(exp(1.8)) );
        CHECK( val(grad_exp[0]) == approx(exp(1.8)) );
        
        // log function
        var f_log = log(x);
        auto grad_log = derivatives(f_log, autodiff::reverse::detail::wrt(x));
        CHECK( val(f_log) == approx(log(1.8)) );
        CHECK( val(grad_log[0]) == approx(1.0/1.8) );
        
        // log10 function
        var f_log10 = log10(x);
        auto grad_log10 = derivatives(f_log10, autodiff::reverse::detail::wrt(x));
        CHECK( val(f_log10) == approx(log10(1.8)) );
        CHECK( val(grad_log10[0]) == approx(1.0/(1.8*log(10.0))) );
    }

    SECTION("testing power and root functions - reverse mode")
    {
        var x = 3.0;
        
        // sqrt function
        var f_sqrt = sqrt(x);
        auto grad_sqrt = derivatives(f_sqrt, autodiff::reverse::detail::wrt(x));
        CHECK( val(f_sqrt) == approx(sqrt(3.0)) );
        CHECK( val(grad_sqrt[0]) == approx(1.0/(2.0*sqrt(3.0))) );
        
        // pow function with constant exponent
        var f_pow = pow(x, 2.5);
        auto grad_pow = derivatives(f_pow, autodiff::reverse::detail::wrt(x));
        CHECK( val(f_pow) == approx(pow(3.0, 2.5)) );
        CHECK( val(grad_pow[0]) == approx(2.5*pow(3.0, 1.5)) );
        
        // pow function with variable exponent (x^x)
        var f_pow_var = pow(x, x);
        auto grad_pow_var = derivatives(f_pow_var, autodiff::reverse::detail::wrt(x));
        CHECK( val(f_pow_var) == approx(pow(3.0, 3.0)) );
        CHECK( val(grad_pow_var[0]) == approx(pow(3.0, 3.0) * (log(3.0) + 1.0)) );
    }

    SECTION("testing special functions - reverse mode")
    {
        // abs function for positive values
        var x_pos = 2.3;
        var f_abs_pos = abs(x_pos);
        auto grad_abs_pos = derivatives(f_abs_pos, autodiff::reverse::detail::wrt(x_pos));
        CHECK( val(f_abs_pos) == approx(std::fabs(2.3)) );
        CHECK( val(grad_abs_pos[0]) == approx(1.0) );
        
        // abs function for negative values
        var x_neg = -2.3;
        var f_abs_neg = abs(x_neg);
        auto grad_abs_neg = derivatives(f_abs_neg, autodiff::reverse::detail::wrt(x_neg));
        CHECK( val(f_abs_neg) == approx(std::fabs(-2.3)) );
        CHECK( val(grad_abs_neg[0]) == approx(-1.0) );
        
        // erf function
        var x_erf = 1.1;
        var f_erf = erf(x_erf);
        auto grad_erf = derivatives(f_erf, autodiff::reverse::detail::wrt(x_erf));
        CHECK( val(f_erf) == approx(erf(1.1)) );
        CHECK( std::isfinite(val(grad_erf[0])) );
        CHECK( val(grad_erf[0]) > 0.0 ); // erf derivative is always positive
    }

    SECTION("testing binary functions - reverse mode")
    {
        var x = 3.5, y = 2.8;
        
        // atan2 function
        var f_atan2 = atan2(y, x);
        auto grad_atan2_x = derivatives(f_atan2, autodiff::reverse::detail::wrt(x));
        auto grad_atan2_y = derivatives(f_atan2, autodiff::reverse::detail::wrt(y));
        CHECK( val(f_atan2) == approx(atan2(2.8, 3.5)) );
        CHECK( val(grad_atan2_x[0]) == approx(-2.8/(3.5*3.5 + 2.8*2.8)) );
        CHECK( val(grad_atan2_y[0]) == approx(3.5/(3.5*3.5 + 2.8*2.8)) );
        
        // hypot function
        var f_hypot = hypot(x, y);
        auto grad_hypot_x = derivatives(f_hypot, autodiff::reverse::detail::wrt(x));
        auto grad_hypot_y = derivatives(f_hypot, autodiff::reverse::detail::wrt(y));
        CHECK( val(f_hypot) == approx(hypot(3.5, 2.8)) );
        CHECK( val(grad_hypot_x[0]) == approx(3.5/hypot(3.5, 2.8)) );
        CHECK( val(grad_hypot_y[0]) == approx(2.8/hypot(3.5, 2.8)) );
    }

    SECTION("testing complex expressions - reverse mode")
    {
        var x = 1.3, y = 0.9;
        
        // Complex function: f(x,y) = sin(x*y) + exp(x-y) + log(x+y)
        var f = sin(x*y) + exp(x-y) + log(x+y);
        auto grad_x = derivatives(f, autodiff::reverse::detail::wrt(x));
        auto grad_y = derivatives(f, autodiff::reverse::detail::wrt(y));
        
        double expected_fx = sin(1.3*0.9) + exp(1.3-0.9) + log(1.3+0.9);
        double expected_dx = cos(1.3*0.9)*0.9 + exp(1.3-0.9)*1.0 + 1.0/(1.3+0.9);
        double expected_dy = cos(1.3*0.9)*1.3 + exp(1.3-0.9)*(-1.0) + 1.0/(1.3+0.9);
        
        CHECK( val(f) == approx(expected_fx) );
        CHECK( val(grad_x[0]) == approx(expected_dx) );
        CHECK( val(grad_y[0]) == approx(expected_dy) );
    }

    SECTION("testing assignment operators - reverse mode")
    {
        var x = 4.0;
        
        // Test += operator
        var f1 = x;
        f1 += 2.0;
        auto grad1 = derivatives(f1, autodiff::reverse::detail::wrt(x));
        CHECK( val(f1) == approx(6.0) );
        CHECK( val(grad1[0]) == approx(1.0) );
        
        // Test -= operator
        var f2 = x;
        f2 -= 1.5;
        auto grad2 = derivatives(f2, autodiff::reverse::detail::wrt(x));
        CHECK( val(f2) == approx(2.5) );
        CHECK( val(grad2[0]) == approx(1.0) );
        
        // Test *= operator
        var f3 = x;
        f3 *= 1.5;
        auto grad3 = derivatives(f3, autodiff::reverse::detail::wrt(x));
        CHECK( val(f3) == approx(6.0) );
        CHECK( val(grad3[0]) == approx(1.5) );
        
        // Test /= operator
        var f4 = x;
        f4 /= 2.0;
        auto grad4 = derivatives(f4, autodiff::reverse::detail::wrt(x));
        CHECK( val(f4) == approx(2.0) );
        CHECK( val(grad4[0]) == approx(0.5) );
    }

    SECTION("testing min and max functions - reverse mode")
    {
        var x = 2.1, y = 3.4;
        
        // min function when x < y
        var f_min = min(x, y);
        auto grad_min_x = derivatives(f_min, autodiff::reverse::detail::wrt(x));
        auto grad_min_y = derivatives(f_min, autodiff::reverse::detail::wrt(y));
        CHECK( val(f_min) == approx(2.1) );
        CHECK( val(grad_min_x[0]) == approx(1.0) );  // x is selected
        CHECK( val(grad_min_y[0]) == approx(0.0) );  // y is not selected
        
        // max function when x < y
        var f_max = max(x, y);
        auto grad_max_x = derivatives(f_max, autodiff::reverse::detail::wrt(x));
        auto grad_max_y = derivatives(f_max, autodiff::reverse::detail::wrt(y));
        CHECK( val(f_max) == approx(3.4) );
        CHECK( val(grad_max_x[0]) == approx(0.0) );  // x is not selected
        CHECK( val(grad_max_y[0]) == approx(1.0) );  // y is selected
        
        // Test when x > y
        var x2 = 4.5, y2 = 2.1;
        var f_min2 = min(x2, y2);
        auto grad_min2_x = derivatives(f_min2, autodiff::reverse::detail::wrt(x2));
        CHECK( val(f_min2) == approx(2.1) );
        CHECK( val(grad_min2_x[0]) == approx(0.0) );  // x is not selected
        
        var f_max2 = max(x2, y2);
        auto grad_max2_x = derivatives(f_max2, autodiff::reverse::detail::wrt(x2));
        CHECK( val(f_max2) == approx(4.5) );
        CHECK( val(grad_max2_x[0]) == approx(1.0) );  // x is selected
    }

    SECTION("testing chain rule applications - reverse mode")
    {
        var x = 0.8;
        
        // f(x) = sin(cos(x))
        var f1 = sin(cos(x));
        auto grad1 = derivatives(f1, autodiff::reverse::detail::wrt(x));
        double expected_grad1 = cos(cos(0.8)) * (-sin(0.8));
        CHECK( val(f1) == approx(sin(cos(0.8))) );
        CHECK( val(grad1[0]) == approx(expected_grad1) );
        
        // f(x) = exp(x²)
        var f2 = exp(x*x);
        auto grad2 = derivatives(f2, autodiff::reverse::detail::wrt(x));
        double expected_grad2 = exp(0.8*0.8) * 2.0*0.8;
        CHECK( val(f2) == approx(exp(0.8*0.8)) );
        CHECK( val(grad2[0]) == approx(expected_grad2) );
        
        // f(x) = log(sqrt(x))
        var f3 = log(sqrt(x));
        auto grad3 = derivatives(f3, autodiff::reverse::detail::wrt(x));
        double expected_grad3 = 1.0/(2.0*0.8);
        CHECK( val(f3) == approx(log(sqrt(0.8))) );
        CHECK( val(grad3[0]) == approx(expected_grad3) );
    }

    SECTION("testing optimization functions - reverse mode")
    {
        var x = -0.5, y = 1.5;
        
        // Rosenbrock function: f(x,y) = (1-x)² + 100*(y-x²)²
        var f_ros = (1-x)*(1-x) + 100*(y-x*x)*(y-x*x);
        auto grad_ros_x = derivatives(f_ros, autodiff::reverse::detail::wrt(x));
        auto grad_ros_y = derivatives(f_ros, autodiff::reverse::detail::wrt(y));
        
        double expected_val = (1-(-0.5))*(1-(-0.5)) + 100*(1.5-(-0.5)*(-0.5))*(1.5-(-0.5)*(-0.5));
        double expected_dx = -2*(1-(-0.5)) - 400*(-0.5)*(1.5-(-0.5)*(-0.5));
        double expected_dy = 200*(1.5-(-0.5)*(-0.5));
        
        CHECK( val(f_ros) == approx(expected_val) );
        CHECK( val(grad_ros_x[0]) == approx(expected_dx) );
        CHECK( val(grad_ros_y[0]) == approx(expected_dy) );
        
        // Sphere function: f(x,y) = x² + y²
        var f_sphere = x*x + y*y;
        auto grad_sphere_x = derivatives(f_sphere, autodiff::reverse::detail::wrt(x));
        auto grad_sphere_y = derivatives(f_sphere, autodiff::reverse::detail::wrt(y));
        
        CHECK( val(f_sphere) == approx((-0.5)*(-0.5) + 1.5*1.5) );
        CHECK( val(grad_sphere_x[0]) == approx(2*(-0.5)) );
        CHECK( val(grad_sphere_y[0]) == approx(2*1.5) );
        
        // Himmelblau's function: f(x,y) = (x²+y-11)² + (x+y²-7)²
        var f_himmel = (x*x + y - 11)*(x*x + y - 11) + (x + y*y - 7)*(x + y*y - 7);
        auto grad_himmel_x = derivatives(f_himmel, autodiff::reverse::detail::wrt(x));
        auto grad_himmel_y = derivatives(f_himmel, autodiff::reverse::detail::wrt(y));
        
        CHECK( std::isfinite(val(f_himmel)) );
        CHECK( std::isfinite(val(grad_himmel_x[0])) );
        CHECK( std::isfinite(val(grad_himmel_y[0])) );
    }

    SECTION("testing numerical stability - reverse mode")
    {
        // Test with very small numbers
        var x_small = 1e-8;
        var f_small = x_small*x_small + x_small;
        auto grad_small = derivatives(f_small, autodiff::reverse::detail::wrt(x_small));
        CHECK( std::isfinite(val(f_small)) );
        CHECK( std::isfinite(val(grad_small[0])) );
        CHECK( val(grad_small[0]) == approx(2*1e-8 + 1.0) );
        
        // Test with large numbers
        var x_large = 1e6;
        var f_large = log(x_large);
        auto grad_large = derivatives(f_large, autodiff::reverse::detail::wrt(x_large));
        CHECK( std::isfinite(val(f_large)) );
        CHECK( std::isfinite(val(grad_large[0])) );
        CHECK( val(grad_large[0]) == approx(1.0/1e6) );
        
        // Test with values near zero
        var x_zero = 1e-15;
        var f_zero = sqrt(x_zero);
        auto grad_zero = derivatives(f_zero, autodiff::reverse::detail::wrt(x_zero));
        CHECK( std::isfinite(val(f_zero)) );
        CHECK( std::isfinite(val(grad_zero[0])) );
    }

    SECTION("testing comparison operators - reverse mode")
    {
        var x = 3.0, y = 2.0;
        
        // These should work for value comparison
        CHECK( x == 3.0 );
        CHECK( 3.0 == x );
        CHECK( x != 2.0 );
        CHECK( 2.0 != x );
        CHECK( x > y );
        CHECK( x > 2.0 );
        CHECK( 3.0 > y );
        CHECK( x >= y );
        CHECK( x >= 3.0 );
        CHECK( y < x );
        CHECK( 2.0 < x );
        CHECK( y <= x );
        CHECK( 2.0 <= x );
    }
}
