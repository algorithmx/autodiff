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

// Catch includes
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

// autodiff includes
#include <autodiff/reverse/unified_expr.hpp>
#include <tests/utils/catch.hpp>

using namespace autodiff::reverse::unified;

TEST_CASE("testing autodiff::reverse::unified", "[reverse][unified]")
{
    SECTION("testing basic arithmetic operations")
    {
        // Create shared arena
        auto arena = std::make_shared<ExpressionArena<double>>();
        
        // Create variables
        UnifiedVariable<double> x(arena, 2.0);
        UnifiedVariable<double> y(arena, 3.0);
        
        // Test basic operations
        auto z1 = x + y;
        auto z2 = x - y;
        auto z3 = x * y;
        auto z4 = x / y;
        
        CHECK(z1.value() == approx(5.0));
        CHECK(z2.value() == approx(-1.0));
        CHECK(z3.value() == approx(6.0));
        CHECK(z4.value() == approx(2.0/3.0));
        
        // Test with scalars
        auto z5 = x + 1.0;
        auto z6 = 2.0 * x;
        auto z7 = 1.0 - x;
        auto z8 = 10.0 / x;
        
        CHECK(z5.value() == approx(3.0));
        CHECK(z6.value() == approx(4.0));
        CHECK(z7.value() == approx(-1.0));
        CHECK(z8.value() == approx(5.0));
        
        // Test unary minus
        auto z9 = -x;
        CHECK(z9.value() == approx(-2.0));
    }

    SECTION("testing mathematical functions")
    {
        auto arena = std::make_shared<ExpressionArena<double>>();
        
        UnifiedVariable<double> x(arena, 1.0);
        
        // Test trigonometric functions
        auto y1 = sin(x);
        auto y2 = cos(x);
        auto y3 = tan(x);
        
        CHECK(y1.value() == approx(std::sin(1.0)));
        CHECK(y2.value() == approx(std::cos(1.0)));
        CHECK(y3.value() == approx(std::tan(1.0)));
        
        // Test exponential and logarithmic functions
        auto y4 = exp(x);
        auto y5 = log(x);
        auto y6 = sqrt(x);
        auto y7 = abs(x);
        
        CHECK(y4.value() == approx(std::exp(1.0)));
        CHECK(y5.value() == approx(std::log(1.0)));
        CHECK(y6.value() == approx(std::sqrt(1.0)));
        CHECK(y7.value() == approx(std::abs(1.0)));
        
        // Test power functions
        auto y8 = pow(x, 2.0);
        auto y9 = pow(2.0, x);
        auto y10 = pow(x, x);
        
        CHECK(y8.value() == approx(1.0));
        CHECK(y9.value() == approx(2.0));
        CHECK(y10.value() == approx(1.0));
    }

    SECTION("testing first-order derivative computation")
    {
        auto arena = std::make_shared<ExpressionArena<double>>();
        
        // Test f(x) = x^2, f'(x) = 2x
        UnifiedVariable<double> x(arena, 3.0);
        auto f = x * x;
        
        auto grads = derivatives(f, wrt(x));
        double df_dx = grads[0];
        
        CHECK(f.value() == approx(9.0));
        CHECK(df_dx == approx(6.0));
        
        // Test multivariate function f(x,y) = x*y + sin(x)
        UnifiedVariable<double> y(arena, 2.0);
        auto g = x * y + sin(x);
        
        auto grads2 = derivatives(g, wrt(x, y));
        double dg_dx = grads2[0];
        double dg_dy = grads2[1];
        
        CHECK(g.value() == approx(6.0 + std::sin(3.0)));
        CHECK(dg_dx == approx(2.0 + std::cos(3.0)));
        CHECK(dg_dy == approx(3.0));
    }

    SECTION("testing complex expressions")
    {
        auto arena = std::make_shared<ExpressionArena<double>>();
        
        // Test complex expression: f(x,y,z) = exp(x*y) + log(z) - sin(x+y*z)
        UnifiedVariable<double> x(arena, 1.0);
        UnifiedVariable<double> y(arena, 2.0);
        UnifiedVariable<double> z(arena, 3.0);
        
        auto f = exp(x * y) + log(z) - sin(x + y * z);
        
        auto grads = derivatives(f, wrt(x, y, z));
        
        // Manually computed expected values
        double expected_f = std::exp(2.0) + std::log(3.0) - std::sin(7.0);
        double expected_df_dx = 2.0 * std::exp(2.0) - std::cos(7.0);
        double expected_df_dy = std::exp(2.0) - 3.0 * std::cos(7.0);
        double expected_df_dz = 1.0/3.0 - 2.0 * std::cos(7.0);
        
        CHECK(f.value() == approx(expected_f));
        CHECK(grads[0] == approx(expected_df_dx));
        CHECK(grads[1] == approx(expected_df_dy));
        CHECK(grads[2] == approx(expected_df_dz));
    }
    
    SECTION("testing polynomial functions")
    {
        auto arena = std::make_shared<ExpressionArena<double>>();
        
        // Test f(x,y,z) = x^3 + y^2*z + x*y*z^2 + x^2*y + z^3
        UnifiedVariable<double> x(arena, 2.0);
        UnifiedVariable<double> y(arena, 3.0);
        UnifiedVariable<double> z(arena, 1.5);
        
        auto f = pow(x, 3.0) + y*y*z + x*y*z*z + x*x*y + pow(z, 3.0);
        
        // Compute partial derivatives
        auto grads = derivatives(f, wrt(x, y, z));
        
        // Expected values: df/dx = 3x^2 + yz^2 + 2xy
        double expected_fx = 3*2*2 + 3*1.5*1.5 + 2*2*3; // = 12 + 6.75 + 12 = 30.75
        // Expected values: df/dy = 2yz + xz^2 + x^2  
        double expected_fy = 2*3*1.5 + 2*1.5*1.5 + 2*2; // = 9 + 4.5 + 4 = 17.5
        // Expected values: df/dz = y^2 + 2xyz + 3z^2
        double expected_fz = 3*3 + 2*2*3*1.5 + 3*1.5*1.5; // = 9 + 18 + 6.75 = 33.75

        CHECK(grads[0] == approx(expected_fx));
        CHECK(grads[1] == approx(expected_fy));
        CHECK(grads[2] == approx(expected_fz));
    }
    
    SECTION("testing trigonometric functions")
    {
        auto arena = std::make_shared<ExpressionArena<double>>();
        
        // Test f(x,y) = sin(x*y) + cos(x+y) + tan(x-y)
        UnifiedVariable<double> x(arena, 0.8);
        UnifiedVariable<double> y(arena, 0.6);
        
        auto f = sin(x*y) + cos(x+y) + tan(x-y);
        
        auto grads = derivatives(f, wrt(x, y));
        
        // Expected values: df/dx = y*cos(x*y) - sin(x+y) + sec^2(x-y)
        double xy = x.value()*y.value();
        double x_plus_y = x.value() + y.value();
        double x_minus_y = x.value() - y.value();
        double expected_fx = y.value()*std::cos(xy) - std::sin(x_plus_y) + 1.0/(std::cos(x_minus_y)*std::cos(x_minus_y));
        // Expected values: df/dy = x*cos(x*y) - sin(x+y) - sec^2(x-y)
        double expected_fy = x.value()*std::cos(xy) - std::sin(x_plus_y) - 1.0/(std::cos(x_minus_y)*std::cos(x_minus_y));

        CHECK(grads[0] == approx(expected_fx));
        CHECK(grads[1] == approx(expected_fy));
    }

    SECTION("testing arena efficiency")
    {
        auto arena = std::make_shared<ExpressionArena<double>>();
        
        size_t initial_size = arena->size();
        
        // Create variables
        UnifiedVariable<double> x(arena, 1.0);
        UnifiedVariable<double> y(arena, 2.0);
        
        size_t after_vars = arena->size();
        
        // Create complex expression
        auto z = x * y + sin(x) - cos(y) + exp(x / y);
        
        size_t after_expr = arena->size();
        
        // Arena should have grown with expressions
        CHECK(after_vars > initial_size);
        CHECK(after_expr > after_vars);
        
        // Compute derivatives
        auto grads = derivatives(z, wrt(x, y));
        
        // Should be able to compute valid derivatives
        CHECK(std::isfinite(grads[0]));
        CHECK(std::isfinite(grads[1]));
    }
}
