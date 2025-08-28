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

#include <cmath>

// Catch includes
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

// autodiff includes
#include <autodiff/reverse/unified_expr.hpp>
#include <tests/utils/catch.hpp>

using namespace autodiff::reverse::unified;

// Test suite that provides equivalent first-order tests for autodiff::reverse::unified implementation
// These tests mirror the functionality tested in first-order-mathematical.test.cpp but use the unified reverse mode

// Helper macros for unified reverse mode testing
#define CHECK_UNIFIED_DERIVATIVES_FX(expr, expected_val, expected_grad)                         \
{                                                                                               \
    auto arena = std::make_shared<ExpressionArena<double>>();                                  \
    UnifiedVariable<double> x(arena, 1.0);                                                     \
    auto result = expr;                                                                         \
    auto grads = derivatives(result, wrt(x));                                                   \
    CHECK( result.value() == approx(expected_val) );                                           \
    CHECK( grads[0] == approx(expected_grad) );                                                \
}

#define CHECK_UNIFIED_DERIVATIVES_AT_VALUE(expr, test_val, expected_val, expected_grad)        \
{                                                                                               \
    auto arena = std::make_shared<ExpressionArena<double>>();                                  \
    UnifiedVariable<double> x(arena, test_val);                                                \
    auto result = expr;                                                                         \
    auto grads = derivatives(result, wrt(x));                                                   \
    CHECK( result.value() == approx(expected_val) );                                           \
    CHECK( grads[0] == approx(expected_grad) );                                                \
}

#define CHECK_UNIFIED_DERIVATIVES_FXY_DX(expr, expected_val, expected_grad_x)                  \
{                                                                                               \
    auto arena = std::make_shared<ExpressionArena<double>>();                                  \
    UnifiedVariable<double> x(arena, 1.0);                                                     \
    UnifiedVariable<double> y(arena, 2.0);                                                     \
    auto result = expr;                                                                         \
    auto grads = derivatives(result, wrt(x));                                                   \
    CHECK( result.value() == approx(expected_val) );                                           \
    CHECK( grads[0] == approx(expected_grad_x) );                                              \
}

#define CHECK_UNIFIED_DERIVATIVES_FXY_DY(expr, expected_val, expected_grad_y)                  \
{                                                                                               \
    auto arena = std::make_shared<ExpressionArena<double>>();                                  \
    UnifiedVariable<double> x(arena, 1.0);                                                     \
    UnifiedVariable<double> y(arena, 2.0);                                                     \
    auto result = expr;                                                                         \
    auto grads = derivatives(result, wrt(y));                                                   \
    CHECK( result.value() == approx(expected_val) );                                           \
    CHECK( grads[0] == approx(expected_grad_y) );                                              \
}

#define CHECK_UNIFIED_DERIVATIVES_FXY_BOTH(expr, expected_val, expected_grad_x, expected_grad_y) \
{                                                                                               \
    auto arena = std::make_shared<ExpressionArena<double>>();                                  \
    UnifiedVariable<double> x(arena, 1.0);                                                     \
    UnifiedVariable<double> y(arena, 2.0);                                                     \
    auto result = expr;                                                                         \
    auto grads = derivatives(result, wrt(x, y));                                               \
    CHECK( result.value() == approx(expected_val) );                                           \
    CHECK( grads[0] == approx(expected_grad_x) );                                              \
    CHECK( grads[1] == approx(expected_grad_y) );                                              \
}

TEST_CASE("testing first-order unified mathematical functions", "[reverse][unified][first-order-mathematical]")
{
    SECTION("testing trigonometric functions with unified")
    {
        // sin function
        CHECK_UNIFIED_DERIVATIVES_FX(sin(x), sin(1.0), cos(1.0));
        
        // cos function  
        CHECK_UNIFIED_DERIVATIVES_FX(cos(x), cos(1.0), -sin(1.0));
        
        // tan function
        CHECK_UNIFIED_DERIVATIVES_FX(tan(x), tan(1.0), 1.0/(cos(1.0)*cos(1.0)));
        
        // asin function (with x = 0.5 to ensure |x| < 1)
        CHECK_UNIFIED_DERIVATIVES_AT_VALUE(asin(x), 0.5, asin(0.5), 1.0/sqrt(1.0 - 0.5*0.5));
        
        // acos function (with x = 0.5 to ensure |x| < 1)
        CHECK_UNIFIED_DERIVATIVES_AT_VALUE(acos(x), 0.5, acos(0.5), -1.0/sqrt(1.0 - 0.5*0.5));
        
        // atan function
        CHECK_UNIFIED_DERIVATIVES_FX(atan(x), atan(1.0), 1.0/(1.0 + 1.0*1.0));
    }

    SECTION("testing hyperbolic functions with unified")
    {
        // sinh function
        CHECK_UNIFIED_DERIVATIVES_FX(sinh(x), sinh(1.0), cosh(1.0));
        
        // cosh function
        CHECK_UNIFIED_DERIVATIVES_FX(cosh(x), cosh(1.0), sinh(1.0));
        
        // tanh function
        CHECK_UNIFIED_DERIVATIVES_FX(tanh(x), tanh(1.0), 1.0/(cosh(1.0)*cosh(1.0)));
    }

    SECTION("testing exponential and logarithmic functions with unified")
    {
        // exp function
        CHECK_UNIFIED_DERIVATIVES_FX(exp(x), exp(1.0), exp(1.0));
        
        // log function
        CHECK_UNIFIED_DERIVATIVES_FX(log(x), log(1.0), 1.0/1.0);
        
        // log10 function
        CHECK_UNIFIED_DERIVATIVES_FX(log10(x), log10(1.0), 1.0/(1.0*log(10.0)));
        
        // Test with x = 2.0 for more interesting log values
        CHECK_UNIFIED_DERIVATIVES_AT_VALUE(log(x), 2.0, log(2.0), 1.0/2.0);
    }

    SECTION("testing power and root functions with unified")
    {
        // sqrt function with x = 4.0
        CHECK_UNIFIED_DERIVATIVES_AT_VALUE(sqrt(x), 4.0, sqrt(4.0), 1.0/(2.0*sqrt(4.0)));
        
        // pow function with constant exponent
        CHECK_UNIFIED_DERIVATIVES_AT_VALUE(pow(x, 3.0), 2.0, pow(2.0, 3.0), 3.0*pow(2.0, 2.0));
        
        // pow function with dual exponent (x^x)
        CHECK_UNIFIED_DERIVATIVES_AT_VALUE(pow(x, x), 2.0, pow(2.0, 2.0), pow(2.0, 2.0) * (log(2.0) + 1.0));
    }

    SECTION("testing special functions with unified")
    {
        // abs function for positive values
        CHECK_UNIFIED_DERIVATIVES_AT_VALUE(abs(x), 2.5, std::fabs(2.5), 1.0);
        
        // abs function for negative values
        CHECK_UNIFIED_DERIVATIVES_AT_VALUE(abs(x), -2.5, std::fabs(-2.5), -1.0);
        
        // erf function
        CHECK_UNIFIED_DERIVATIVES_AT_VALUE(erf(x), 1.5, erf(1.5), 2.0/sqrt(M_PI) * exp(-1.5*1.5));
    }

    SECTION("testing binary functions with unified")
    {
        auto arena = std::make_shared<ExpressionArena<double>>();
        
        // atan2 function - testing atan2(y, x) derivatives
        UnifiedVariable<double> x_at2(arena, 3.0);
        UnifiedVariable<double> y_at2(arena, 4.0);
        
        // df/dx for atan2(y, x) = -y/(x² + y²)
        auto atan2_result = atan2(y_at2, x_at2);
        auto atan2_grads_x = derivatives(atan2_result, wrt(x_at2));
        CHECK( atan2_result.value() == approx(atan2(4.0, 3.0)) );
        CHECK( atan2_grads_x[0] == approx(-4.0/(3.0*3.0 + 4.0*4.0)) );
        
        // df/dy for atan2(y, x) = x/(x² + y²)
        auto atan2_grads_y = derivatives(atan2_result, wrt(y_at2));
        CHECK( atan2_grads_y[0] == approx(3.0/(3.0*3.0 + 4.0*4.0)) );
        
        // hypot function - testing hypot(x, y) derivatives
        auto hypot_result = hypot(x_at2, y_at2);
        
        // df/dx for hypot(x, y) = x/hypot(x, y)
        auto hypot_grads_x = derivatives(hypot_result, wrt(x_at2));
        CHECK( hypot_result.value() == approx(hypot(3.0, 4.0)) );
        CHECK( hypot_grads_x[0] == approx(3.0/hypot(3.0, 4.0)) );
        
        // df/dy for hypot(x, y) = y/hypot(x, y)
        auto hypot_grads_y = derivatives(hypot_result, wrt(y_at2));
        CHECK( hypot_grads_y[0] == approx(4.0/hypot(3.0, 4.0)) );
    }

    SECTION("testing arithmetic operations with unified")
    {
        // Addition: (x + y) derivatives
        CHECK_UNIFIED_DERIVATIVES_FXY_DX(x + y, 3.0, 1.0);
        CHECK_UNIFIED_DERIVATIVES_FXY_DY(x + y, 3.0, 1.0);
        
        // Subtraction: (x - y) derivatives
        CHECK_UNIFIED_DERIVATIVES_FXY_DX(x - y, -1.0, 1.0);
        CHECK_UNIFIED_DERIVATIVES_FXY_DY(x - y, -1.0, -1.0);
        
        // Multiplication: (x * y) derivatives
        CHECK_UNIFIED_DERIVATIVES_FXY_DX(x * y, 2.0, 2.0);  // df/dx = y = 2.0
        CHECK_UNIFIED_DERIVATIVES_FXY_DY(x * y, 2.0, 1.0);  // df/dy = x = 1.0
        
        // Division: (x / y) derivatives
        CHECK_UNIFIED_DERIVATIVES_FXY_DX(x / y, 0.5, 0.5);    // df/dx = 1/y = 1/2
        CHECK_UNIFIED_DERIVATIVES_FXY_DY(x / y, 0.5, -0.25);  // df/dy = -x/y² = -1/4
    }

    SECTION("testing complex expressions with unified")
    {
        // Complex expression: f(x,y) = sin(x*y) + exp(x-y)
        // df/dx = cos(x*y)*y + exp(x-y)
        // df/dy = cos(x*y)*x - exp(x-y)
        
        auto arena = std::make_shared<ExpressionArena<double>>();
        UnifiedVariable<double> x_complex(arena, 1.0);
        UnifiedVariable<double> y_complex(arena, 2.0);
        
        auto complex_expr = sin(x_complex * y_complex) + exp(x_complex - y_complex);
        auto complex_grads = derivatives(complex_expr, wrt(x_complex, y_complex));
        
        double expected_val = sin(1.0 * 2.0) + exp(1.0 - 2.0);
        double expected_dx = cos(1.0 * 2.0) * 2.0 + exp(1.0 - 2.0);
        double expected_dy = cos(1.0 * 2.0) * 1.0 - exp(1.0 - 2.0);
        
        CHECK( complex_expr.value() == approx(expected_val) );
        CHECK( complex_grads[0] == approx(expected_dx) );
        CHECK( complex_grads[1] == approx(expected_dy) );
    }

    SECTION("testing assignment operators with unified")
    {
        auto arena = std::make_shared<ExpressionArena<double>>();
        UnifiedVariable<double> x(arena, 3.0);
        
        // Test += (note: unified expressions are immutable, so we test equivalent operations)
        auto x1 = x + 5.0;
        auto grads1 = derivatives(x1, wrt(x));
        CHECK( x1.value() == approx(8.0) );
        CHECK( grads1[0] == approx(1.0) );
        
        // Test -= equivalent
        auto x2 = x - 1.0;
        auto grads2 = derivatives(x2, wrt(x));
        CHECK( x2.value() == approx(2.0) );
        CHECK( grads2[0] == approx(1.0) );
        
        // Test *= equivalent
        auto x3 = x * 2.0;
        auto grads3 = derivatives(x3, wrt(x));
        CHECK( x3.value() == approx(6.0) );
        CHECK( grads3[0] == approx(2.0) );
        
        // Test /= equivalent
        auto x4 = x / 3.0;
        auto grads4 = derivatives(x4, wrt(x));
        CHECK( x4.value() == approx(1.0) );
        CHECK( grads4[0] == approx(1.0/3.0) );
    }

    SECTION("testing min and max functions with unified")
    {
        auto arena = std::make_shared<ExpressionArena<double>>();
        
        // min function when x < y
        UnifiedVariable<double> x_min1(arena, 1.5);
        UnifiedVariable<double> y_min1(arena, 2.5);
        auto min_result1 = min(x_min1, y_min1);
        auto min_grads1_x = derivatives(min_result1, wrt(x_min1));
        auto min_grads1_y = derivatives(min_result1, wrt(y_min1));
        CHECK( min_result1.value() == approx(1.5) );
        CHECK( min_grads1_x[0] == approx(1.0) );
        CHECK( min_grads1_y[0] == approx(0.0) );
        
        // max function when x < y (should select y)
        auto max_result1 = max(x_min1, y_min1);
        auto max_grads1_x = derivatives(max_result1, wrt(x_min1));
        auto max_grads1_y = derivatives(max_result1, wrt(y_min1));
        CHECK( max_result1.value() == approx(2.5) );
        CHECK( max_grads1_x[0] == approx(0.0) ); // derivative w.r.t. x when y is selected
        CHECK( max_grads1_y[0] == approx(1.0) );
        
        // min function when x > y
        UnifiedVariable<double> x_min2(arena, 2.5);
        UnifiedVariable<double> y_min2(arena, 1.5);
        auto min_result2 = min(x_min2, y_min2);
        auto min_grads2_x = derivatives(min_result2, wrt(x_min2));
        auto min_grads2_y = derivatives(min_result2, wrt(y_min2));
        CHECK( min_result2.value() == approx(1.5) );
        CHECK( min_grads2_x[0] == approx(0.0) ); // derivative w.r.t. x when y is selected
        CHECK( min_grads2_y[0] == approx(1.0) );
        
        // max function when x > y (should select x)
        auto max_result2 = max(x_min2, y_min2);
        auto max_grads2_x = derivatives(max_result2, wrt(x_min2));
        auto max_grads2_y = derivatives(max_result2, wrt(y_min2));
        CHECK( max_result2.value() == approx(2.5) );
        CHECK( max_grads2_x[0] == approx(1.0) ); // derivative w.r.t. x when x is selected
        CHECK( max_grads2_y[0] == approx(0.0) );
    }

    SECTION("testing chain rule with unified")
    {
        auto arena = std::make_shared<ExpressionArena<double>>();
        UnifiedVariable<double> x(arena, 0.5);
        
        // f(x) = sin(cos(x))
        // df/dx = cos(cos(x)) * (-sin(x))
        auto chain1 = sin(cos(x));
        auto chain1_grads = derivatives(chain1, wrt(x));
        double expected_chain1 = cos(cos(0.5)) * (-sin(0.5));
        CHECK( chain1.value() == approx(sin(cos(0.5))) );
        CHECK( chain1_grads[0] == approx(expected_chain1) );
        
        // f(x) = exp(x²)
        // df/dx = exp(x²) * 2x
        auto chain2 = exp(x*x);
        auto chain2_grads = derivatives(chain2, wrt(x));
        double expected_chain2 = exp(0.5*0.5) * 2.0*0.5;
        CHECK( chain2.value() == approx(exp(0.5*0.5)) );
        CHECK( chain2_grads[0] == approx(expected_chain2) );
        
        // f(x) = log(sqrt(x))
        // df/dx = 1/sqrt(x) * 1/(2*sqrt(x)) = 1/(2*x)
        auto chain3 = log(sqrt(x));
        auto chain3_grads = derivatives(chain3, wrt(x));
        double expected_chain3 = 1.0/(2.0*0.5);
        CHECK( chain3.value() == approx(log(sqrt(0.5))) );
        CHECK( chain3_grads[0] == approx(expected_chain3) );
    }
}

TEST_CASE("testing first-order unified var mathematical functions", "[reverse][unified][first-order-mathematical]")
{
    SECTION("testing unified type basic operations")
    {
        auto arena = std::make_shared<ExpressionArena<double>>();
        UnifiedVariable<double> x(arena, 2.0);
        UnifiedVariable<double> y(arena, 3.0);
        
        // These should work with unified expressions
        auto f1 = x*x + y*y;
        auto f2 = sin(x) * cos(y);
        auto f3 = exp(x) / log(y);
        
        // Test that the functions compute correctly
        CHECK( f1.value() == approx(2.0*2.0 + 3.0*3.0) );
        CHECK( f2.value() == approx(sin(2.0) * cos(3.0)) );
        CHECK( f3.value() == approx(exp(2.0) / log(3.0)) );
        
        // Test first-order derivatives work with derivatives function
        auto df1_dx = derivatives(f1, wrt(x));
        auto df1_dy = derivatives(f1, wrt(y));
        
        CHECK( df1_dx[0] == approx(2.0*2.0) );            // df/dx = 2x
        CHECK( df1_dy[0] == approx(2.0*3.0) );            // df/dy = 2y
    }

    SECTION("testing unified type mathematical functions")
    {
        auto arena = std::make_shared<ExpressionArena<double>>();
        UnifiedVariable<double> x(arena, 1.5);
        
        auto f_trig = sin(x) + cos(x) + tan(x);
        auto f_hyp = sinh(x) + cosh(x) + tanh(x);
        auto f_exp = exp(x) + log(x) + sqrt(x);
        
        // Test derivatives
        auto df_trig = derivatives(f_trig, wrt(x));
        auto df_hyp = derivatives(f_hyp, wrt(x));
        auto df_exp = derivatives(f_exp, wrt(x));
        
        // Check function values
        CHECK( f_trig.value() == approx(sin(1.5) + cos(1.5) + tan(1.5)) );
        CHECK( f_hyp.value() == approx(sinh(1.5) + cosh(1.5) + tanh(1.5)) );
        CHECK( f_exp.value() == approx(exp(1.5) + log(1.5) + sqrt(1.5)) );
        
        // Check derivatives are finite and reasonable
        CHECK( std::isfinite(df_trig[0]) );
        CHECK( std::isfinite(df_hyp[0]) );
        CHECK( std::isfinite(df_exp[0]) );
        
        // Check some specific derivative values
        double expected_exp_deriv = exp(1.5) + 1.0/1.5 + 1.0/(2.0*sqrt(1.5));
        CHECK( df_exp[0] == approx(expected_exp_deriv) );
    }

    SECTION("testing unified type complex functions")
    {
        auto arena = std::make_shared<ExpressionArena<double>>();
        UnifiedVariable<double> x(arena, 0.8);
        UnifiedVariable<double> y(arena, 1.2);
        
        // Complex optimization-like function
        auto rosenbrock = (1.0-x)*(1.0-x) + 100.0*(y-x*x)*(y-x*x);
        
        // Test derivatives
        auto dr_dx = derivatives(rosenbrock, wrt(x));
        auto dr_dy = derivatives(rosenbrock, wrt(y));
        
        // Check function value
        double expected_val = (1-0.8)*(1-0.8) + 100*(1.2-0.8*0.8)*(1.2-0.8*0.8);
        CHECK( rosenbrock.value() == approx(expected_val) );
        
        // Check derivatives are finite
        CHECK( std::isfinite(dr_dx[0]) );
        CHECK( std::isfinite(dr_dy[0]) );
        
        // Check derivative values
        double expected_dx = -2*(1-0.8) - 400*0.8*(1.2-0.8*0.8);
        double expected_dy = 200*(1.2-0.8*0.8);
        CHECK( dr_dx[0] == approx(expected_dx) );
        CHECK( dr_dy[0] == approx(expected_dy) );
    }
}

// Test numerical stability and edge cases that should work regardless of higher-order support
TEST_CASE("testing first-order unified numerical stability", "[reverse][unified][first-order-stability]")
{
    SECTION("testing with very small numbers")
    {
        auto arena = std::make_shared<ExpressionArena<double>>();
        UnifiedVariable<double> x(arena, 1e-10);
        
        auto result1 = x*x + x;
        auto grads1 = derivatives(result1, wrt(x));
        CHECK( std::isfinite(result1.value()) );
        CHECK( std::isfinite(grads1[0]) );
        CHECK( grads1[0] == approx(2*1e-10 + 1.0) );
        
        auto result2 = sqrt(x);
        auto grads2 = derivatives(result2, wrt(x));
        CHECK( std::isfinite(result2.value()) );
        CHECK( std::isfinite(grads2[0]) );
    }

    SECTION("testing with large numbers")
    {
        auto arena = std::make_shared<ExpressionArena<double>>();
        UnifiedVariable<double> x(arena, 1e6);
        
        auto result1 = log(x);
        auto grads1 = derivatives(result1, wrt(x));
        CHECK( std::isfinite(result1.value()) );
        CHECK( std::isfinite(grads1[0]) );
        CHECK( grads1[0] == approx(1.0/1e6) );
        
        auto result2 = sqrt(x);
        auto grads2 = derivatives(result2, wrt(x));
        CHECK( std::isfinite(result2.value()) );
        CHECK( std::isfinite(grads2[0]) );
    }

    SECTION("testing near-zero gradients")
    {
        auto arena = std::make_shared<ExpressionArena<double>>();
        UnifiedVariable<double> x(arena, 0.0);
        
        // f(x) = x³ at x=0 should have zero gradient
        auto result = x*x*x;
        auto grads = derivatives(result, wrt(x));
        CHECK( result.value() == approx(0.0) );
        CHECK( grads[0] == approx(0.0) );
        
        // f(x) = sin(x) at x=0 should have gradient 1
        auto result2 = sin(x);
        auto grads2 = derivatives(result2, wrt(x));
        CHECK( result2.value() == approx(0.0) );
        CHECK( grads2[0] == approx(1.0) );
    }

    SECTION("testing function composition stability")
    {
        auto arena = std::make_shared<ExpressionArena<double>>();
        UnifiedVariable<double> x(arena, 1.0);
        
        // Complex composition that should remain stable
        auto result = sin(exp(log(sqrt(x*x))));
        auto grads = derivatives(result, wrt(x));
        CHECK( std::isfinite(result.value()) );
        CHECK( std::isfinite(grads[0]) );
        
        // This should simplify to sin(x)
        CHECK( result.value() == approx(sin(1.0)) );
        CHECK( grads[0] == approx(cos(1.0)) );
    }
}
