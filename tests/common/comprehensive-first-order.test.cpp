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

// C++ includes
#include <cstddef>
#include <chrono>
#include <cmath>

// Catch includes
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

// autodiff includes
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
#include <tests/utils/catch.hpp>
using namespace autodiff;

// Helper macros for first-order testing that work regardless of AUTODIFF_DISABLE_HIGHER_ORDER
#define CHECK_FIRST_ORDER_FX(expr, u, ux)                                             \
{                                                                                      \
    dual x = 1.0;                                                                      \
    detail::seed<0>(x, 1.0);                                                           \
    detail::seed<1>(x, 1.0);                                                           \
    auto result = expr;                                                                \
    CHECK( val(result) == approx(val(u)) );                                           \
    CHECK( grad(result) == approx(val(ux)) );                                         \
}

#define CHECK_FIRST_ORDER_FY(expr, u, uy)                                             \
{                                                                                      \
    dual y = 2.0;                                                                      \
    detail::seed<0>(y, 2.0);                                                           \
    detail::seed<1>(y, 1.0);                                                           \
    auto result = expr;                                                                \
    CHECK( val(result) == approx(val(u)) );                                           \
    CHECK( grad(result) == approx(val(uy)) );                                         \
}

#define CHECK_FIRST_ORDER_FXY_DX(expr, u, ux)                                         \
{                                                                                      \
    dual x = 1.0, y = 2.0;                                                             \
    detail::seed<0>(x, 1.0);                                                           \
    detail::seed<1>(x, 1.0);                                                           \
    auto result = expr;                                                                \
    CHECK( val(result) == approx(val(u)) );                                           \
    CHECK( grad(result) == approx(val(ux)) );                                         \
}

#define CHECK_FIRST_ORDER_FXY_DY(expr, u, uy)                                         \
{                                                                                      \
    dual x = 1.0, y = 2.0;                                                             \
    detail::seed<0>(y, 2.0);                                                           \
    detail::seed<1>(y, 1.0);                                                           \
    auto result = expr;                                                                \
    CHECK( val(result) == approx(val(u)) );                                           \
    CHECK( grad(result) == approx(val(uy)) );                                         \
}

// Comprehensive test suite for first-order derivatives that works regardless of AUTODIFF_DISABLE_HIGHER_ORDER
TEST_CASE("comprehensive testing first-order derivatives", "[first-order-comprehensive]")
{
    SECTION("testing basic arithmetic operations - first order")
    {
        // Addition operations
        CHECK_FIRST_ORDER_FX(x + 5.0, 6.0, 1.0);
        CHECK_FIRST_ORDER_FX(5.0 + x, 6.0, 1.0);
        CHECK_FIRST_ORDER_FXY_DX(x + y, 3.0, 1.0);
        CHECK_FIRST_ORDER_FXY_DY(x + y, 3.0, 1.0);
        
        // Subtraction operations
        CHECK_FIRST_ORDER_FX(x - 3.0, -2.0, 1.0);
        CHECK_FIRST_ORDER_FX(3.0 - x, 2.0, -1.0);
        CHECK_FIRST_ORDER_FXY_DX(x - y, -1.0, 1.0);
        CHECK_FIRST_ORDER_FXY_DY(x - y, -1.0, -1.0);
        
        // Multiplication operations
        CHECK_FIRST_ORDER_FX(x * 3.0, 3.0, 3.0);
        CHECK_FIRST_ORDER_FX(3.0 * x, 3.0, 3.0);
        CHECK_FIRST_ORDER_FXY_DX(x * y, 2.0, 2.0);
        CHECK_FIRST_ORDER_FXY_DY(x * y, 2.0, 1.0);
        
        // Division operations
        CHECK_FIRST_ORDER_FX(x / 2.0, 0.5, 0.5);
        CHECK_FIRST_ORDER_FX(2.0 / x, 2.0, -2.0);
        CHECK_FIRST_ORDER_FXY_DX(x / y, 0.5, 0.5);
        CHECK_FIRST_ORDER_FXY_DY(x / y, 0.5, -0.25);
    }

    SECTION("testing trigonometric functions - first order")
    {
        dual x = 0.5;
        detail::seed<0>(x, 0.5);
        detail::seed<1>(x, 1.0);
        
        // sin function: d/dx[sin(x)] = cos(x)
        auto sin_result = sin(x);
        CHECK( val(sin_result) == approx(sin(0.5)) );
        CHECK( grad(sin_result) == approx(cos(0.5)) );
        
        // cos function: d/dx[cos(x)] = -sin(x)
        auto cos_result = cos(x);
        CHECK( val(cos_result) == approx(cos(0.5)) );
        CHECK( grad(cos_result) == approx(-sin(0.5)) );
        
        // tan function: d/dx[tan(x)] = sec²(x) = 1/cos²(x)
        auto tan_result = tan(x);
        CHECK( val(tan_result) == approx(tan(0.5)) );
        CHECK( grad(tan_result) == approx(1.0/(cos(0.5)*cos(0.5))) );
    }

    SECTION("testing inverse trigonometric functions - first order")
    {
        dual x = 0.3;
        detail::seed<0>(x, 0.3);
        detail::seed<1>(x, 1.0);
        
        // asin function: d/dx[asin(x)] = 1/sqrt(1-x²)
        auto asin_result = asin(x);
        CHECK( val(asin_result) == approx(asin(0.3)) );
        CHECK( grad(asin_result) == approx(1.0/sqrt(1.0 - 0.3*0.3)) );
        
        // acos function: d/dx[acos(x)] = -1/sqrt(1-x²)
        auto acos_result = acos(x);
        CHECK( val(acos_result) == approx(acos(0.3)) );
        CHECK( grad(acos_result) == approx(-1.0/sqrt(1.0 - 0.3*0.3)) );
        
        // atan function: d/dx[atan(x)] = 1/(1+x²)
        auto atan_result = atan(x);
        CHECK( val(atan_result) == approx(atan(0.3)) );
        CHECK( grad(atan_result) == approx(1.0/(1.0 + 0.3*0.3)) );
    }

    SECTION("testing hyperbolic functions - first order")
    {
        dual x = 0.8;
        detail::seed<0>(x, 0.8);
        detail::seed<1>(x, 1.0);
        
        // sinh function: d/dx[sinh(x)] = cosh(x)
        auto sinh_result = sinh(x);
        CHECK( val(sinh_result) == approx(sinh(0.8)) );
        CHECK( grad(sinh_result) == approx(cosh(0.8)) );
        
        // cosh function: d/dx[cosh(x)] = sinh(x)
        auto cosh_result = cosh(x);
        CHECK( val(cosh_result) == approx(cosh(0.8)) );
        CHECK( grad(cosh_result) == approx(sinh(0.8)) );
        
        // tanh function: d/dx[tanh(x)] = sech²(x) = 1/cosh²(x)
        auto tanh_result = tanh(x);
        CHECK( val(tanh_result) == approx(tanh(0.8)) );
        CHECK( grad(tanh_result) == approx(1.0/(cosh(0.8)*cosh(0.8))) );
    }

    SECTION("testing exponential and logarithmic functions - first order")
    {
        dual x = 1.5;
        detail::seed<0>(x, 1.5);
        detail::seed<1>(x, 1.0);
        
        // exp function: d/dx[exp(x)] = exp(x)
        auto exp_result = exp(x);
        CHECK( val(exp_result) == approx(exp(1.5)) );
        CHECK( grad(exp_result) == approx(exp(1.5)) );
        
        // log function: d/dx[log(x)] = 1/x
        auto log_result = log(x);
        CHECK( val(log_result) == approx(log(1.5)) );
        CHECK( grad(log_result) == approx(1.0/1.5) );
        
        // log10 function: d/dx[log10(x)] = 1/(x*ln(10))
        auto log10_result = log10(x);
        CHECK( val(log10_result) == approx(log10(1.5)) );
        CHECK( grad(log10_result) == approx(1.0/(1.5*log(10.0))) );
    }

    SECTION("testing power and root functions - first order")
    {
        dual x = 2.0;
        detail::seed<0>(x, 2.0);
        detail::seed<1>(x, 1.0);
        
        // sqrt function: d/dx[sqrt(x)] = 1/(2*sqrt(x))
        auto sqrt_result = sqrt(x);
        CHECK( val(sqrt_result) == approx(sqrt(2.0)) );
        CHECK( grad(sqrt_result) == approx(1.0/(2.0*sqrt(2.0))) );
        
        // pow function with constant exponent: d/dx[x^n] = n*x^(n-1)
        auto pow_result = pow(x, 3.0);
        CHECK( val(pow_result) == approx(pow(2.0, 3.0)) );
        CHECK( grad(pow_result) == approx(3.0*pow(2.0, 2.0)) );
        
        // pow function with dual exponent: d/dx[x^x] = x^x * (ln(x) + 1)
        auto pow_dual_result = pow(x, x);
        CHECK( val(pow_dual_result) == approx(pow(2.0, 2.0)) );
        CHECK( grad(pow_dual_result) == approx(pow(2.0, 2.0) * (log(2.0) + 1.0)) );
    }

    SECTION("testing special mathematical functions - first order")
    {
        dual x = 1.2;
        detail::seed<0>(x, 1.2);
        detail::seed<1>(x, 1.0);
        
        // abs function: d/dx[abs(x)] = sign(x) (for x > 0)
        auto abs_result = abs(x);
        CHECK( val(abs_result) == approx(std::fabs(1.2)) );
        CHECK( grad(abs_result) == approx(1.0) );
        
        // erf function
        auto erf_result = erf(x);
        CHECK( val(erf_result) == approx(erf(1.2)) );
        CHECK( std::isfinite(grad(erf_result)) );
        CHECK( grad(erf_result) > 0.0 ); // erf derivative is always positive
        
        // Test with negative value for abs
        dual x_neg = -1.2;
        detail::seed<0>(x_neg, -1.2);
        detail::seed<1>(x_neg, 1.0);
        auto abs_neg_result = abs(x_neg);
        CHECK( val(abs_neg_result) == approx(std::fabs(-1.2)) );
        CHECK( grad(abs_neg_result) == approx(-1.0) );
    }

    SECTION("testing atan2 and hypot functions - first order")
    {
        dual x = 1.5, y = 2.5;
        
        // Test atan2(y, x) with respect to x
        detail::seed<0>(x, 1.5);
        detail::seed<1>(x, 1.0);
        y = 2.5; // y is constant
        auto atan2_dx = atan2(y, x);
        CHECK( val(atan2_dx) == approx(atan2(2.5, 1.5)) );
        CHECK( grad(atan2_dx) == approx(-2.5/(1.5*1.5 + 2.5*2.5)) );
        
        // Test atan2(y, x) with respect to y
        x = 1.5; // x is constant
        detail::seed<0>(y, 2.5);
        detail::seed<1>(y, 1.0);
        auto atan2_dy = atan2(y, x);
        CHECK( val(atan2_dy) == approx(atan2(2.5, 1.5)) );
        CHECK( grad(atan2_dy) == approx(1.5/(1.5*1.5 + 2.5*2.5)) );
        
        // Test hypot(x, y) with respect to x
        detail::seed<0>(x, 1.5);
        detail::seed<1>(x, 1.0);
        y = 2.5; // y is constant
        auto hypot_dx = hypot(x, y);
        CHECK( val(hypot_dx) == approx(hypot(1.5, 2.5)) );
        CHECK( grad(hypot_dx) == approx(1.5/hypot(1.5, 2.5)) );
        
        // Test hypot(x, y) with respect to y
        x = 1.5; // x is constant
        detail::seed<0>(y, 2.5);
        detail::seed<1>(y, 1.0);
        auto hypot_dy = hypot(x, y);
        CHECK( val(hypot_dy) == approx(hypot(1.5, 2.5)) );
        CHECK( grad(hypot_dy) == approx(2.5/hypot(1.5, 2.5)) );
    }

    SECTION("testing complex expressions - first order")
    {
        dual x = 0.7, y = 1.3;
        
        // Complex expression: f(x,y) = sin(x)*exp(y) + log(x+y)*sqrt(x*y)
        // Test df/dx
        detail::seed<0>(x, 0.7);
        detail::seed<1>(x, 1.0);
        y = 1.3; // y is constant
        auto complex_dx = sin(x)*exp(y) + log(x+y)*sqrt(x*y);
        double expected_dx = cos(0.7)*exp(1.3) + (1.0/(0.7+1.3))*sqrt(0.7*1.3) + log(0.7+1.3)*(1.3/(2.0*sqrt(0.7*1.3)));
        CHECK( val(complex_dx) == approx(sin(0.7)*exp(1.3) + log(0.7+1.3)*sqrt(0.7*1.3)) );
        CHECK( grad(complex_dx) == approx(expected_dx) );
        
        // Test df/dy
        x = 0.7; // x is constant
        detail::seed<0>(y, 1.3);
        detail::seed<1>(y, 1.0);
        auto complex_dy = sin(x)*exp(y) + log(x+y)*sqrt(x*y);
        double expected_dy = sin(0.7)*exp(1.3) + (1.0/(0.7+1.3))*sqrt(0.7*1.3) + log(0.7+1.3)*(0.7/(2.0*sqrt(0.7*1.3)));
        CHECK( val(complex_dy) == approx(sin(0.7)*exp(1.3) + log(0.7+1.3)*sqrt(0.7*1.3)) );
        CHECK( grad(complex_dy) == approx(expected_dy) );
    }

    SECTION("testing assignment operators - first order")
    {
        dual x = 3.0, y = 2.0;
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

    SECTION("testing min and max functions - first order")
    {
        dual x = 1.5, y = 2.5;
        
        // min function - when x < y, derivative w.r.t. x should be 1, w.r.t. y should be 0
        detail::seed<0>(x, 1.5);
        detail::seed<1>(x, 1.0);
        y = 2.5; // y is constant
        auto min_result = min(x, y);
        CHECK( val(min_result) == approx(1.5) );
        CHECK( grad(min_result) == approx(1.0) );
        
        // max function - when x < y, derivative w.r.t. x should be 0, w.r.t. y should be 1
        detail::seed<0>(y, 2.5);
        detail::seed<1>(y, 1.0);
        x = 1.5; // x is constant
        auto max_result = max(x, y);
        CHECK( val(max_result) == approx(2.5) );
        CHECK( grad(max_result) == approx(1.0) );
    }

    SECTION("testing chain rule applications - first order")
    {
        dual x = 1.0;
        detail::seed<0>(x, 1.0);
        detail::seed<1>(x, 1.0);
        
        // f(x) = sin(cos(x))
        // df/dx = cos(cos(x)) * (-sin(x))
        auto chain1 = sin(cos(x));
        double expected_chain1 = cos(cos(1.0)) * (-sin(1.0));
        CHECK( val(chain1) == approx(sin(cos(1.0))) );
        CHECK( grad(chain1) == approx(expected_chain1) );
        
        // f(x) = exp(x²)
        // df/dx = exp(x²) * 2x
        auto chain2 = exp(x*x);
        double expected_chain2 = exp(1.0*1.0) * 2.0*1.0;
        CHECK( val(chain2) == approx(exp(1.0*1.0)) );
        CHECK( grad(chain2) == approx(expected_chain2) );
        
        // f(x) = log(sqrt(x))
        // df/dx = 1/sqrt(x) * 1/(2*sqrt(x)) = 1/(2*x)
        auto chain3 = log(sqrt(x));
        double expected_chain3 = 1.0/(2.0*1.0);
        CHECK( val(chain3) == approx(log(sqrt(1.0))) );
        CHECK( grad(chain3) == approx(expected_chain3) );
    }

    SECTION("testing optimization-relevant functions - first order")
    {
        dual x = -1.0, y = 1.0;
        
        // Rosenbrock function: f(x,y) = (1-x)² + 100*(y-x²)²
        // df/dx = -2*(1-x) + 100*2*(y-x²)*(-2*x) = -2*(1-x) - 400*x*(y-x²)
        detail::seed<0>(x, -1.0);
        detail::seed<1>(x, 1.0);
        y = 1.0; // y is constant
        auto rosenbrock_dx = (1-x)*(1-x) + 100*(y-x*x)*(y-x*x);
        double expected_rosenbrock_dx = -2.0*(1.0-(-1.0)) - 400.0*(-1.0)*(1.0-(-1.0)*(-1.0));
        CHECK( val(rosenbrock_dx) == approx((1-(-1.0))*(1-(-1.0)) + 100*(1.0-(-1.0)*(-1.0))*(1.0-(-1.0)*(-1.0))) );
        CHECK( grad(rosenbrock_dx) == approx(expected_rosenbrock_dx) );
        
        // Sphere function: f(x,y) = x² + y²
        // df/dx = 2x, df/dy = 2y
        auto sphere_dx = x*x + y*y;
        CHECK( val(sphere_dx) == approx((-1.0)*(-1.0) + 1.0*1.0) );
        CHECK( grad(sphere_dx) == approx(2.0*(-1.0)) );
    }

#ifdef AUTODIFF_EIGEN_FOUND
    SECTION("testing Eigen vector operations - first order")
    {
        // Test gradient computation for vector-valued function
        VectorXreal x(3);
        x << 1.0, 2.0, 3.0;
        
        auto f = [](const VectorXreal& x) -> real {
            return x[0]*x[0] + 2*x[1]*x[1]*x[1] + 3*x[2]*x[2]*x[2]*x[2];
        };
        
        Eigen::VectorXd g = gradient(f, autodiff::detail::wrt(x), at(x));
        
        CHECK( g.size() == 3 );
        CHECK( g[0] == approx(2*1.0) );           // df/dx0 = 2*x0
        CHECK( g[1] == approx(6*2.0*2.0) );       // df/dx1 = 6*x1²
        CHECK( g[2] == approx(12*3.0*3.0*3.0) );  // df/dx2 = 12*x2³
    }

    SECTION("testing Eigen jacobian computations - first order")
    {
        // Test F(x) = [x0², x0*x1, x1²] where x = [x0, x1]
        auto F = [](const VectorXreal& x) -> VectorXreal {
            VectorXreal result(3);
            result[0] = x[0]*x[0];
            result[1] = x[0]*x[1];
            result[2] = x[1]*x[1];
            return result;
        };

        VectorXreal x(2);
        x << 2.0, 3.0;

        Eigen::MatrixXd J = jacobian(F, autodiff::detail::wrt(x), at(x));

        CHECK( J.rows() == 3 );
        CHECK( J.cols() == 2 );
        CHECK( J(0, 0) == approx(2*2.0) );     // ∂F0/∂x0 = 2*x0
        CHECK( J(0, 1) == approx(0.0) );       // ∂F0/∂x1 = 0
        CHECK( J(1, 0) == approx(3.0) );       // ∂F1/∂x0 = x1
        CHECK( J(1, 1) == approx(2.0) );       // ∂F1/∂x1 = x0
        CHECK( J(2, 0) == approx(0.0) );       // ∂F2/∂x0 = 0
        CHECK( J(2, 1) == approx(2*3.0) );     // ∂F2/∂x1 = 2*x1
    }
#endif // AUTODIFF_EIGEN_FOUND

    SECTION("testing performance and numerical stability - first order")
    {
        // Test with very small numbers
        dual x = 1e-8;
        detail::seed<0>(x, 1e-8);
        detail::seed<1>(x, 1.0);
        
        auto result_small = x*x + x;
        CHECK( std::isfinite(val(result_small)) );
        CHECK( std::isfinite(grad(result_small)) );
        CHECK( grad(result_small) == approx(2*1e-8 + 1.0) );
        
        // Test with large numbers
        dual x_large = 1e6;
        detail::seed<0>(x_large, 1e6);
        detail::seed<1>(x_large, 1.0);
        
        auto result_large = sqrt(x_large);
        CHECK( std::isfinite(val(result_large)) );
        CHECK( std::isfinite(grad(result_large)) );
        CHECK( grad(result_large) == approx(1.0/(2.0*sqrt(1e6))) );
    }
}

// Test reverse mode equivalent functionality
TEST_CASE("comprehensive testing first-order derivatives - reverse mode", "[first-order-comprehensive-reverse]")
{
    SECTION("testing basic arithmetic operations - reverse mode")
    {
        var x = 1.5, y = 2.5;
        
        // Test addition
        var f1 = x + y;
        auto grad_x1 = derivatives(f1, autodiff::reverse::detail::wrt(x));
        auto grad_y1 = derivatives(f1, autodiff::reverse::detail::wrt(y));
        CHECK( val(f1) == approx(4.0) );
        CHECK( val(grad_x1[0]) == approx(1.0) );
        CHECK( val(grad_y1[0]) == approx(1.0) );
        
        // Test multiplication
        var f2 = x * y;
        auto grad_x2 = derivatives(f2, autodiff::reverse::detail::wrt(x));
        auto grad_y2 = derivatives(f2, autodiff::reverse::detail::wrt(y));
        CHECK( val(f2) == approx(3.75) );
        CHECK( val(grad_x2[0]) == approx(2.5) );
        CHECK( val(grad_y2[0]) == approx(1.5) );
    }

    SECTION("testing mathematical functions - reverse mode")
    {
        var x = 0.5;
        
        // Test sin function
        var f_sin = sin(x);
        auto grad_sin = derivatives(f_sin, autodiff::reverse::detail::wrt(x));
        CHECK( val(f_sin) == approx(sin(0.5)) );
        CHECK( val(grad_sin[0]) == approx(cos(0.5)) );
        
        // Test exp function
        var f_exp = exp(x);
        auto grad_exp = derivatives(f_exp, autodiff::reverse::detail::wrt(x));
        CHECK( val(f_exp) == approx(exp(0.5)) );
        CHECK( val(grad_exp[0]) == approx(exp(0.5)) );
        
        // Test log function
        var x_log = 2.0;
        var f_log = log(x_log);
        auto grad_log = derivatives(f_log, autodiff::reverse::detail::wrt(x_log));
        CHECK( val(f_log) == approx(log(2.0)) );
        CHECK( val(grad_log[0]) == approx(1.0/2.0) );
    }

    SECTION("testing complex expressions - reverse mode")
    {
        var x = 1.2, y = 0.8;
        
        // Complex function: f(x,y) = sin(x*y) + exp(x-y)
        var f = sin(x*y) + exp(x-y);
        auto grad_x = derivatives(f, autodiff::reverse::detail::wrt(x));
        auto grad_y = derivatives(f, autodiff::reverse::detail::wrt(y));
        
        double expected_fx = sin(1.2*0.8) + exp(1.2-0.8);
        double expected_dx = cos(1.2*0.8)*0.8 + exp(1.2-0.8)*1.0;
        double expected_dy = cos(1.2*0.8)*1.2 + exp(1.2-0.8)*(-1.0);
        
        CHECK( val(f) == approx(expected_fx) );
        CHECK( val(grad_x[0]) == approx(expected_dx) );
        CHECK( val(grad_y[0]) == approx(expected_dy) );
    }

#ifdef AUTODIFF_EIGEN_FOUND
    SECTION("testing Eigen operations - reverse mode")
    {
        // Test gradient computation for vector function
        VectorXvar x(2);
        x << 1.5, 2.5;
        
        var f = x[0]*x[0] + x[1]*x[1]*x[1];
        
        auto grad_x0 = derivatives(f, autodiff::reverse::detail::wrt(x[0]));
        auto grad_x1 = derivatives(f, autodiff::reverse::detail::wrt(x[1]));
        
        CHECK( val(f) == approx(1.5*1.5 + 2.5*2.5*2.5) );
        CHECK( val(grad_x0[0]) == approx(2*1.5) );
        CHECK( val(grad_x1[0]) == approx(3*2.5*2.5) );
    }
#endif // AUTODIFF_EIGEN_FOUND
}
