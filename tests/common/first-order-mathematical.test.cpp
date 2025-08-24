//                  _  _
//  _   _|_ _  _|o_|__|_
//  (_||_||_(_)(_|| |  |
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

#include <cmath>

// Catch includes
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

// autodiff includes
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/real.hpp>
#include <tests/utils/catch.hpp>
using namespace autodiff;

// Test suite that provides equivalent first-order tests when AUTODIFF_DISABLE_HIGHER_ORDER is enabled
// These tests ensure that all mathematical functions and operations that work in first-order mode
// are properly tested, maintaining the same level of test coverage regardless of compilation flags

// Helper macros for manual seeding approach that works when AUTODIFF_DISABLE_HIGHER_ORDER is enabled
#define CHECK_MANUAL_DERIVATIVES_FX(expr, expected_val, expected_grad)                          \
{                                                                                               \
    dual x;                                                                                     \
    detail::seed<0>(x, 1.0);                                                                    \
    detail::seed<1>(x, 1.0);                                                                    \
    auto result = expr;                                                                         \
    CHECK( val(result) == approx(val(expected_val)) );                                         \
    CHECK( grad(result) == approx(val(expected_grad)) );                                       \
}

#define CHECK_DERIVATIVES_AT_VALUE(expr, test_val, expected_val, expected_grad)                \
{                                                                                               \
    auto f = [](dual x) -> dual { return expr; };                                              \
    dual x_dummy = test_val;                                                                    \
    auto result = derivatives(f, autodiff::detail::wrt(x_dummy), autodiff::detail::at(x_dummy));   \
    CHECK( result[0] == approx(expected_val) );                                                 \
    CHECK( result[1] == approx(expected_grad) );                                               \
}

#define CHECK_MANUAL_DERIVATIVES_FXY_DX(expr, expected_val, expected_grad_x)                   \
{                                                                                               \
    dual x, y;                                                                                  \
    detail::seed<0>(x, 1.0);                                                                    \
    detail::seed<1>(x, 1.0);                                                                    \
    y = 2.0;                                                                                    \
    auto result = expr;                                                                         \
    CHECK( val(result) == approx(val(expected_val)) );                                         \
    CHECK( grad(result) == approx(val(expected_grad_x)) );                                     \
}

#define CHECK_MANUAL_DERIVATIVES_FXY_DY(expr, expected_val, expected_grad_y)                   \
{                                                                                               \
    dual x, y;                                                                                  \
    x = 1.0;                                                                                    \
    detail::seed<0>(y, 2.0);                                                                    \
    detail::seed<1>(y, 1.0);                                                                    \
    auto result = expr;                                                                         \
    CHECK( val(result) == approx(val(expected_val)) );                                         \
    CHECK( grad(result) == approx(val(expected_grad_y)) );                                     \
}



TEST_CASE("testing first-order dual mathematical functions", "[forward][dual][first-order-mathematical]")
{
    SECTION("testing trigonometric functions with manual seeding")
    {
        // sin function
        CHECK_MANUAL_DERIVATIVES_FX(sin(x), sin(1.0), cos(1.0));
        
        // cos function  
        CHECK_MANUAL_DERIVATIVES_FX(cos(x), cos(1.0), -sin(1.0));
        
        // tan function
        CHECK_MANUAL_DERIVATIVES_FX(tan(x), tan(1.0), 1.0/(cos(1.0)*cos(1.0)));
        
        // asin function (with x = 0.5 to ensure |x| < 1)
        dual x_asin;
        detail::seed<0>(x_asin, 0.5);
        detail::seed<1>(x_asin, 1.0);
        auto asin_result = asin(x_asin);
        CHECK( val(asin_result) == approx(asin(0.5)) );
        CHECK( grad(asin_result) == approx(1.0/sqrt(1.0 - 0.5*0.5)) );
        
        // acos function (with x = 0.5 to ensure |x| < 1)
        dual x_acos;
        detail::seed<0>(x_acos, 0.5);
        detail::seed<1>(x_acos, 1.0);
        auto acos_result = acos(x_acos);
        CHECK( val(acos_result) == approx(acos(0.5)) );
        CHECK( grad(acos_result) == approx(-1.0/sqrt(1.0 - 0.5*0.5)) );
        
        // atan function
        CHECK_MANUAL_DERIVATIVES_FX(atan(x), atan(1.0), 1.0/(1.0 + 1.0*1.0));
    }

    SECTION("testing hyperbolic functions with manual seeding")
    {
        // sinh function
        CHECK_MANUAL_DERIVATIVES_FX(sinh(x), sinh(1.0), cosh(1.0));
        
        // cosh function
        CHECK_MANUAL_DERIVATIVES_FX(cosh(x), cosh(1.0), sinh(1.0));
        
        // tanh function
        CHECK_MANUAL_DERIVATIVES_FX(tanh(x), tanh(1.0), 1.0/(cosh(1.0)*cosh(1.0)));
    }

    SECTION("testing exponential and logarithmic functions with manual seeding")
    {
        // exp function
        CHECK_MANUAL_DERIVATIVES_FX(exp(x), exp(1.0), exp(1.0));
        
        // log function
        CHECK_MANUAL_DERIVATIVES_FX(log(x), log(1.0), 1.0/1.0);
        
        // log10 function
        CHECK_MANUAL_DERIVATIVES_FX(log10(x), log10(1.0), 1.0/(1.0*log(10.0)));
        
        // Test with x = 2.0 for more interesting log values
        dual x_log;
        detail::seed<0>(x_log, 2.0);
        detail::seed<1>(x_log, 1.0);
        auto log_result = log(x_log);
        CHECK( val(log_result) == approx(log(2.0)) );
        CHECK( grad(log_result) == approx(1.0/2.0) );
    }

    SECTION("testing power and root functions with manual seeding")
    {
        // sqrt function with x = 4.0
        dual x_sqrt;
        detail::seed<0>(x_sqrt, 4.0);
        detail::seed<1>(x_sqrt, 1.0);
        auto sqrt_result = sqrt(x_sqrt);
        CHECK( val(sqrt_result) == approx(sqrt(4.0)) );
        CHECK( grad(sqrt_result) == approx(1.0/(2.0*sqrt(4.0))) );
        
        // pow function with constant exponent
        dual x_pow;
        detail::seed<0>(x_pow, 2.0);
        detail::seed<1>(x_pow, 1.0);
        auto pow_result = pow(x_pow, 3.0);
        CHECK( val(pow_result) == approx(pow(2.0, 3.0)) );
        CHECK( grad(pow_result) == approx(3.0*pow(2.0, 2.0)) );
        
        // pow function with dual exponent (x^x)
        auto pow_dual_result = pow(x_pow, x_pow);
        CHECK( val(pow_dual_result) == approx(pow(2.0, 2.0)) );
        CHECK( grad(pow_dual_result) == approx(pow(2.0, 2.0) * (log(2.0) + 1.0)) );
    }

    SECTION("testing special functions with manual seeding")
    {
        // abs function for positive values
        CHECK_DERIVATIVES_AT_VALUE(abs(x), 2.5, std::fabs(2.5), 1.0);
        
        // abs function for negative values
        CHECK_DERIVATIVES_AT_VALUE(abs(x), -2.5, std::fabs(-2.5), -1.0);
        
        // erf function
        CHECK_DERIVATIVES_AT_VALUE(erf(x), 1.5, erf(1.5), 2.0/sqrt(M_PI) * exp(-1.5*1.5));
    }

    SECTION("testing binary functions with manual seeding")
    {
        // atan2 function - testing atan2(y, x) derivatives
        dual x_at2, y_at2;
        
        // df/dx for atan2(y, x) = -y/(x² + y²)
        detail::seed<0>(x_at2, 3.0);
        detail::seed<1>(x_at2, 1.0);
        y_at2 = 4.0; // y is constant
        auto atan2_dx = atan2(y_at2, x_at2);
        CHECK( val(atan2_dx) == approx(atan2(4.0, 3.0)) );
        CHECK( grad(atan2_dx) == approx(-4.0/(3.0*3.0 + 4.0*4.0)) );
        
        // df/dy for atan2(y, x) = x/(x² + y²)
        x_at2 = 3.0; // x is constant
        detail::seed<0>(y_at2, 4.0);
        detail::seed<1>(y_at2, 1.0);
        auto atan2_dy = atan2(y_at2, x_at2);
        CHECK( val(atan2_dy) == approx(atan2(4.0, 3.0)) );
        CHECK( grad(atan2_dy) == approx(3.0/(3.0*3.0 + 4.0*4.0)) );
        
        // hypot function - testing hypot(x, y) derivatives
        dual x_hyp, y_hyp;
        
        // df/dx for hypot(x, y) = x/hypot(x, y)
        detail::seed<0>(x_hyp, 3.0);
        detail::seed<1>(x_hyp, 1.0);
        y_hyp = 4.0; // y is constant
        auto hypot_dx = hypot(x_hyp, y_hyp);
        CHECK( val(hypot_dx) == approx(hypot(3.0, 4.0)) );
        CHECK( grad(hypot_dx) == approx(3.0/hypot(3.0, 4.0)) );
        
        // df/dy for hypot(x, y) = y/hypot(x, y)
        x_hyp = 3.0; // x is constant
        detail::seed<0>(y_hyp, 4.0);
        detail::seed<1>(y_hyp, 1.0);
        auto hypot_dy = hypot(x_hyp, y_hyp);
        CHECK( val(hypot_dy) == approx(hypot(3.0, 4.0)) );
        CHECK( grad(hypot_dy) == approx(4.0/hypot(3.0, 4.0)) );
    }

    SECTION("testing arithmetic operations with manual seeding")
    {
        // Addition: (x + y) derivatives
        CHECK_MANUAL_DERIVATIVES_FXY_DX(x + y, 3.0, 1.0);
        CHECK_MANUAL_DERIVATIVES_FXY_DY(x + y, 3.0, 1.0);
        
        // Subtraction: (x - y) derivatives
        CHECK_MANUAL_DERIVATIVES_FXY_DX(x - y, -1.0, 1.0);
        CHECK_MANUAL_DERIVATIVES_FXY_DY(x - y, -1.0, -1.0);
        
        // Multiplication: (x * y) derivatives
        CHECK_MANUAL_DERIVATIVES_FXY_DX(x * y, 2.0, 2.0);  // df/dx = y = 2.0
        CHECK_MANUAL_DERIVATIVES_FXY_DY(x * y, 2.0, 1.0);  // df/dy = x = 1.0
        
        // Division: (x / y) derivatives
        CHECK_MANUAL_DERIVATIVES_FXY_DX(x / y, 0.5, 0.5);    // df/dx = 1/y = 1/2
        CHECK_MANUAL_DERIVATIVES_FXY_DY(x / y, 0.5, -0.25);  // df/dy = -x/y² = -1/4
    }

    SECTION("testing complex expressions with manual seeding")
    {
        // Complex expression: f(x,y) = sin(x*y) + exp(x-y)
        // df/dx = cos(x*y)*y + exp(x-y)
        // df/dy = cos(x*y)*x - exp(x-y)
        
        dual x_complex, y_complex;
        
        // Test df/dx
        detail::seed<0>(x_complex, 1.0);
        detail::seed<1>(x_complex, 1.0);
        y_complex = 2.0; // y is constant
        auto complex_dx = sin(x_complex * y_complex) + exp(x_complex - y_complex);
        double expected_val = sin(1.0 * 2.0) + exp(1.0 - 2.0);
        double expected_dx = cos(1.0 * 2.0) * 2.0 + exp(1.0 - 2.0);
        CHECK( val(complex_dx) == approx(expected_val) );
        CHECK( grad(complex_dx) == approx(expected_dx) );
        
        // Test df/dy
        x_complex = 1.0; // x is constant
        detail::seed<0>(y_complex, 2.0);
        detail::seed<1>(y_complex, 1.0);
        auto complex_dy = sin(x_complex * y_complex) + exp(x_complex - y_complex);
        double expected_dy = cos(1.0 * 2.0) * 1.0 - exp(1.0 - 2.0);
        CHECK( val(complex_dy) == approx(expected_val) );
        CHECK( grad(complex_dy) == approx(expected_dy) );
    }

    SECTION("testing assignment operators with manual seeding")
    {
        dual x;
        detail::seed<0>(x, 3.0);
        detail::seed<1>(x, 1.0);
        
        // Test +=
        auto x1 = x;
        x1 += 5.0;
        CHECK( val(x1) == approx(8.0) );
        CHECK( grad(x1) == approx(1.0) );
        
        // Test -=
        auto x2 = x;
        x2 -= 1.0;
        CHECK( val(x2) == approx(2.0) );
        CHECK( grad(x2) == approx(1.0) );
        
        // Test *=
        auto x3 = x;
        x3 *= 2.0;
        CHECK( val(x3) == approx(6.0) );
        CHECK( grad(x3) == approx(2.0) );
        
        // Test /=
        auto x4 = x;
        x4 /= 3.0;
        CHECK( val(x4) == approx(1.0) );
        CHECK( grad(x4) == approx(1.0/3.0) );
    }

    SECTION("testing min and max functions with manual seeding")
    {
        dual x_min, y_min;
        
        // min function when x < y
        detail::seed<0>(x_min, 1.5);
        detail::seed<1>(x_min, 1.0);
        y_min = 2.5; // y is constant
        auto min_result = min(x_min, y_min);
        CHECK( val(min_result) == approx(1.5) );
        CHECK( grad(min_result) == approx(1.0) );
        
        // max function when x < y (should select y)
        auto max_result = max(x_min, y_min);
        CHECK( val(max_result) == approx(2.5) );
        CHECK( grad(max_result) == approx(0.0) ); // derivative w.r.t. x when y is selected
        
        // min function when x > y
        detail::seed<0>(x_min, 2.5);
        detail::seed<1>(x_min, 1.0);
        y_min = 1.5; // y is constant
        auto min_result2 = min(x_min, y_min);
        CHECK( val(min_result2) == approx(1.5) );
        CHECK( grad(min_result2) == approx(0.0) ); // derivative w.r.t. x when y is selected
        
        // max function when x > y (should select x)
        auto max_result2 = max(x_min, y_min);
        CHECK( val(max_result2) == approx(2.5) );
        CHECK( grad(max_result2) == approx(1.0) ); // derivative w.r.t. x when x is selected
    }

    SECTION("testing chain rule with manual seeding")
    {
        dual x;
        detail::seed<0>(x, 0.5);
        detail::seed<1>(x, 1.0);
        
        // f(x) = sin(cos(x))
        // df/dx = cos(cos(x)) * (-sin(x))
        auto chain1 = sin(cos(x));
        double expected_chain1 = cos(cos(0.5)) * (-sin(0.5));
        CHECK( val(chain1) == approx(sin(cos(0.5))) );
        CHECK( grad(chain1) == approx(expected_chain1) );
        
        // f(x) = exp(x²)
        // df/dx = exp(x²) * 2x
        auto chain2 = exp(x*x);
        double expected_chain2 = exp(0.5*0.5) * 2.0*0.5;
        CHECK( val(chain2) == approx(exp(0.5*0.5)) );
        CHECK( grad(chain2) == approx(expected_chain2) );
        
        // f(x) = log(sqrt(x))
        // df/dx = 1/sqrt(x) * 1/(2*sqrt(x)) = 1/(2*x)
        auto chain3 = log(sqrt(x));
        double expected_chain3 = 1.0/(2.0*0.5);
        CHECK( val(chain3) == approx(log(sqrt(0.5))) );
        CHECK( grad(chain3) == approx(expected_chain3) );
    }
}

TEST_CASE("testing first-order real mathematical functions", "[forward][real][first-order-mathematical]")
{
    SECTION("testing real type basic operations")
    {
        real x = 2.0;
        real y = 3.0;
        
        // These should work even with AUTODIFF_DISABLE_HIGHER_ORDER
        auto f1 = [](real x, real y) -> real { return x*x + y*y; };
        auto f2 = [](real x, real y) -> real { return sin(x) * cos(y); };
        auto f3 = [](real x, real y) -> real { return exp(x) / log(y); };
        
        // Test that the functions compute correctly
        CHECK( val(f1(x, y)) == approx(2.0*2.0 + 3.0*3.0) );
        CHECK( val(f2(x, y)) == approx(sin(2.0) * cos(3.0)) );
        CHECK( val(f3(x, y)) == approx(exp(2.0) / log(3.0)) );
        
        // Test first-order derivatives work with derivatives function
        auto df1_dx = derivatives(f1, autodiff::detail::wrt(x), at(x, y));
        auto df1_dy = derivatives(f1, autodiff::detail::wrt(y), at(x, y));
        
        CHECK( df1_dx[0] == approx(2.0*2.0 + 3.0*3.0) );  // function value
        CHECK( df1_dx[1] == approx(2.0*2.0) );            // df/dx = 2x
        CHECK( df1_dy[0] == approx(2.0*2.0 + 3.0*3.0) );  // function value
        CHECK( df1_dy[1] == approx(2.0*3.0) );            // df/dy = 2y
    }

    SECTION("testing real type mathematical functions")
    {
        real x = 1.5;
        
        auto f_trig = [](real x) -> real { return sin(x) + cos(x) + tan(x); };
        auto f_hyp = [](real x) -> real { return sinh(x) + cosh(x) + tanh(x); };
        auto f_exp = [](real x) -> real { return exp(x) + log(x) + sqrt(x); };
        
        // Test derivatives
        auto df_trig = derivatives(f_trig, autodiff::detail::wrt(x), at(x));
        auto df_hyp = derivatives(f_hyp, autodiff::detail::wrt(x), at(x));
        auto df_exp = derivatives(f_exp, autodiff::detail::wrt(x), at(x));
        
        // Check function values
        CHECK( df_trig[0] == approx(sin(1.5) + cos(1.5) + tan(1.5)) );
        CHECK( df_hyp[0] == approx(sinh(1.5) + cosh(1.5) + tanh(1.5)) );
        CHECK( df_exp[0] == approx(exp(1.5) + log(1.5) + sqrt(1.5)) );
        
        // Check derivatives are finite and reasonable
        CHECK( std::isfinite(df_trig[1]) );
        CHECK( std::isfinite(df_hyp[1]) );
        CHECK( std::isfinite(df_exp[1]) );
        
        // Check some specific derivative values
        double expected_exp_deriv = exp(1.5) + 1.0/1.5 + 1.0/(2.0*sqrt(1.5));
        CHECK( df_exp[1] == approx(expected_exp_deriv) );
    }

    SECTION("testing real type complex functions")
    {
        real x = 0.8, y = 1.2;
        
        // Complex optimization-like function
        auto rosenbrock = [](real x, real y) -> real {
            return (1-x)*(1-x) + 100*(y-x*x)*(y-x*x);
        };
        
        // Test derivatives
        auto dr_dx = derivatives(rosenbrock, autodiff::detail::wrt(x), at(x, y));
        auto dr_dy = derivatives(rosenbrock, autodiff::detail::wrt(y), at(x, y));
        
        // Check function value
        double expected_val = (1-0.8)*(1-0.8) + 100*(1.2-0.8*0.8)*(1.2-0.8*0.8);
        CHECK( dr_dx[0] == approx(expected_val) );
        CHECK( dr_dy[0] == approx(expected_val) );
        
        // Check derivatives are finite
        CHECK( std::isfinite(dr_dx[1]) );
        CHECK( std::isfinite(dr_dy[1]) );
        
        // Check derivative values
        double expected_dx = -2*(1-0.8) - 400*0.8*(1.2-0.8*0.8);
        double expected_dy = 200*(1.2-0.8*0.8);
        CHECK( dr_dx[1] == approx(expected_dx) );
        CHECK( dr_dy[1] == approx(expected_dy) );
    }
}

// Test numerical stability and edge cases that should work regardless of higher-order support
TEST_CASE("testing first-order numerical stability", "[first-order-stability]")
{
    SECTION("testing with very small numbers")
    {
        dual x;
        detail::seed<0>(x, 1e-10);
        detail::seed<1>(x, 1.0);
        
        auto result1 = x*x + x;
        CHECK( std::isfinite(val(result1)) );
        CHECK( std::isfinite(grad(result1)) );
        CHECK( grad(result1) == approx(2*1e-10 + 1.0) );
        
        auto result2 = sqrt(x);
        CHECK( std::isfinite(val(result2)) );
        CHECK( std::isfinite(grad(result2)) );
    }

    SECTION("testing with large numbers")
    {
        dual x;
        detail::seed<0>(x, 1e6);
        detail::seed<1>(x, 1.0);
        
        auto result1 = log(x);
        CHECK( std::isfinite(val(result1)) );
        CHECK( std::isfinite(grad(result1)) );
        CHECK( grad(result1) == approx(1.0/1e6) );
        
        auto result2 = sqrt(x);
        CHECK( std::isfinite(val(result2)) );
        CHECK( std::isfinite(grad(result2)) );
    }

    SECTION("testing near-zero gradients")
    {
        dual x;
        detail::seed<0>(x, 0.0);
        detail::seed<1>(x, 1.0);
        
        // f(x) = x³ at x=0 should have zero gradient
        auto result = x*x*x;
        CHECK( val(result) == approx(0.0) );
        CHECK( grad(result) == approx(0.0) );
        
        // f(x) = sin(x) at x=0 should have gradient 1
        auto result2 = sin(x);
        CHECK( val(result2) == approx(0.0) );
        CHECK( grad(result2) == approx(1.0) );
    }

    SECTION("testing function composition stability")
    {
        dual x;
        detail::seed<0>(x, 1.0);
        detail::seed<1>(x, 1.0);
        
        // Complex composition that should remain stable
        auto result = sin(exp(log(sqrt(x*x))));
        CHECK( std::isfinite(val(result)) );
        CHECK( std::isfinite(grad(result)) );
        
        // This should simplify to sin(x)
        CHECK( val(result) == approx(sin(1.0)) );
        CHECK( grad(result) == approx(cos(1.0)) );
    }
}
