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


#define CHECK_UNIFIED_DERIVATIVES_FX(arena, x_val, expr, u, ux)         \
{                                                                        \
    UnifiedVariable<double> x(arena, x_val);                             \
    auto f = expr;                                                       \
    auto dfdx = derivatives(f, wrt(x));                                  \
    CHECK( f.value() == approx(u) );                                     \
    CHECK( dfdx[0] == approx(ux) );                                      \
}

#define CHECK_UNIFIED_DERIVATIVES_FXY(arena, x_val, y_val, expr, u, ux, uy)            \
{                                                                                       \
    UnifiedVariable<double> x(arena, x_val);                                            \
    UnifiedVariable<double> y(arena, y_val);                                            \
    auto f = expr;                                                                      \
    auto dfdx = derivatives(f, wrt(x));                                                 \
    CHECK( f.value() == approx(u) );                                                    \
    CHECK( dfdx[0] == approx(ux) );                                                     \
    auto dfdy = derivatives(f, wrt(y));                                                 \
    CHECK( dfdy[0] == approx(uy) );                                                     \
}

#define CHECK_UNIFIED_DERIVATIVES_FXYZ(arena, x_val, y_val, z_val, expr, u, ux, uy, uz)               \
{                                                                                                       \
    UnifiedVariable<double> x(arena, x_val);                                                            \
    UnifiedVariable<double> y(arena, y_val);                                                            \
    UnifiedVariable<double> z(arena, z_val);                                                            \
    auto f = expr;                                                                                      \
    auto dfdx = derivatives(f, wrt(x));                                                                 \
    CHECK( f.value() == approx(u) );                                                                    \
    CHECK( dfdx[0] == approx(ux) );                                                                     \
    auto dfdy = derivatives(f, wrt(y));                                                                 \
    CHECK( dfdy[0] == approx(uy) );                                                                     \
    auto dfdz = derivatives(f, wrt(z));                                                                 \
    CHECK( dfdz[0] == approx(uz) );                                                                     \
}

// This file provides equivalent first-order tests for functionality that would normally
// be tested in higher-order test sections that get disabled when AUTODIFF_DISABLE_HIGHER_ORDER is enabled.
// These tests ensure comprehensive coverage of first-order derivatives regardless of compilation flags.

TEST_CASE("testing unified first-order equivalents of higher-order patterns", "[reverse][unified][first-order-equivalents]")
{
    auto arena = std::make_shared<ExpressionArena<double>>();
    
    UnifiedVariable<double> x(arena, 1.0);
    UnifiedVariable<double> y(arena, 2.0);
    UnifiedVariable<double> z(arena, 3.0);

    SECTION("testing complex mathematical expressions - first order")
    {
        x = UnifiedVariable<double>(arena, 0.5);
        y = UnifiedVariable<double>(arena, 0.8);

        // Complex trigonometric expression
        CHECK_UNIFIED_DERIVATIVES_FXY(arena, 0.5, 0.8, sin(x + y) * cos(x / y) + tan(2.0 * x * y), 
            std::sin(0.5 + 0.8) * std::cos(0.5 / 0.8) + std::tan(2.0 * 0.5 * 0.8),
            std::cos(0.5 + 0.8) * std::cos(0.5 / 0.8) - std::sin(0.5 + 0.8) * std::sin(0.5 / 0.8) / 0.8 + 2.0 * 0.8 / (std::cos(2.0 * 0.5 * 0.8) * std::cos(2.0 * 0.5 * 0.8)),
            std::cos(0.5 + 0.8) * std::cos(0.5 / 0.8) + std::sin(0.5 + 0.8) * std::sin(0.5 / 0.8) * 0.5 / (0.8 * 0.8) + 2.0 * 0.5 / (std::cos(2.0 * 0.5 * 0.8) * std::cos(2.0 * 0.5 * 0.8)));

        // Complex exponential/logarithmic expression  
        x = UnifiedVariable<double>(arena, 1.5);
        y = UnifiedVariable<double>(arena, 2.5);
        CHECK_UNIFIED_DERIVATIVES_FXY(arena, 1.5, 2.5, log(x + y) * exp(x / y) + sqrt(2.0 * x * y),
            std::log(1.5 + 2.5) * std::exp(1.5 / 2.5) + std::sqrt(2.0 * 1.5 * 2.5),
            std::exp(1.5 / 2.5) / (1.5 + 2.5) + std::log(1.5 + 2.5) * std::exp(1.5 / 2.5) / 2.5 + 2.5 / std::sqrt(2.0 * 1.5 * 2.5),
            std::exp(1.5 / 2.5) / (1.5 + 2.5) - std::log(1.5 + 2.5) * std::exp(1.5 / 2.5) * 1.5 / (2.5 * 2.5) + 1.5 / std::sqrt(2.0 * 1.5 * 2.5));
    }

    SECTION("testing polynomial expressions - first order")
    {
        x = UnifiedVariable<double>(arena, 1.0);
        y = UnifiedVariable<double>(arena, 2.0);

        // Cubic polynomial: f(x,y) = (x+y)³
        CHECK_UNIFIED_DERIVATIVES_FXY(arena, 1.0, 2.0, (x + y) * (x + y) * (x + y),
            (1.0 + 2.0) * (1.0 + 2.0) * (1.0 + 2.0),
            3.0 * std::pow(1.0 + 2.0, 2.0),
            3.0 * std::pow(1.0 + 2.0, 2.0));

        // Quartic polynomial: f(x,y) = (x+y)⁴
        x = UnifiedVariable<double>(arena, 1.5);
        y = UnifiedVariable<double>(arena, 0.5);
        CHECK_UNIFIED_DERIVATIVES_FXY(arena, 1.5, 0.5, (x + y) * (x + y) * (x + y) * (x + y),
            (1.5 + 0.5) * (1.5 + 0.5) * (1.5 + 0.5) * (1.5 + 0.5),
            4.0 * std::pow(1.5 + 0.5, 3.0),
            4.0 * std::pow(1.5 + 0.5, 3.0));
    }

    SECTION("testing mathematical function compositions - first order")
    {
        x = UnifiedVariable<double>(arena, 0.5);

        // Composition: f(x) = sin(sqrt(x))
        CHECK_UNIFIED_DERIVATIVES_FX(arena, 0.5, sin(sqrt(x)),
            std::sin(std::sqrt(0.5)),
            std::cos(std::sqrt(0.5)) / (2.0 * std::sqrt(0.5)));

        // Another composition: f(x) = log(exp(x²))
        // This should simplify to x²
        CHECK_UNIFIED_DERIVATIVES_FX(arena, 0.5, log(exp(x * x)),
            0.5 * 0.5,
            2.0 * 0.5);

        // More complex composition: f(x) = exp(sin(log(x)))
        x = UnifiedVariable<double>(arena, 2.0);
        CHECK_UNIFIED_DERIVATIVES_FX(arena, 2.0, exp(sin(log(x))),
            std::exp(std::sin(std::log(2.0))),
            std::exp(std::sin(std::log(2.0))) * std::cos(std::log(2.0)) / 2.0);
    }

    SECTION("testing expression node handling - first order")
    {
        x = UnifiedVariable<double>(arena, 2.0);
        y = UnifiedVariable<double>(arena, 3.0);

        // Complex expression that creates multiple intermediate nodes
        CHECK_UNIFIED_DERIVATIVES_FXY(arena, 2.0, 3.0, ((x + y) * (x - y)) / (x * y) + sqrt(x + y), 
            ((2.0 + 3.0) * (2.0 - 3.0)) / (2.0 * 3.0) + std::sqrt(2.0 + 3.0),
            // df/dx for ((x+y)(x-y))/(xy) + sqrt(x+y)
            ((2.0 - 3.0 + 2.0 + 3.0) * (2.0 * 3.0) - (2.0 + 3.0) * (2.0 - 3.0) * 3.0) / (2.0 * 3.0 * 2.0 * 3.0) + 1.0 / (2.0 * std::sqrt(2.0 + 3.0)),
            // df/dy for ((x+y)(x-y))/(xy) + sqrt(x+y) 
            - 2.0 / (3.0 * 3.0) - 1.0 / 2.0 + 1.0 / (2.0 * std::sqrt(2.0 + 3.0)));
    }

    SECTION("testing equivalence with analytical derivatives - first order")
    {
        // Test various functions with known analytical derivatives
        x = UnifiedVariable<double>(arena, 2.0);

        // f(x) = x³, f'(x) = 3x²
        CHECK_UNIFIED_DERIVATIVES_FX(arena, 2.0, x * x * x, 2.0 * 2.0 * 2.0, 3.0 * 2.0 * 2.0);

        x = UnifiedVariable<double>(arena, 0.7);
        // f(x) = sin(x), f'(x) = cos(x)
        CHECK_UNIFIED_DERIVATIVES_FX(arena, 0.7, sin(x), std::sin(0.7), std::cos(0.7));

        x = UnifiedVariable<double>(arena, 1.5);
        // f(x) = exp(x), f'(x) = exp(x)
        CHECK_UNIFIED_DERIVATIVES_FX(arena, 1.5, exp(x), std::exp(1.5), std::exp(1.5));

        x = UnifiedVariable<double>(arena, 2.5);
        // f(x) = log(x), f'(x) = 1/x
        CHECK_UNIFIED_DERIVATIVES_FX(arena, 2.5, log(x), std::log(2.5), 1.0 / 2.5);

        x = UnifiedVariable<double>(arena, 4.0);
        // f(x) = sqrt(x), f'(x) = 1/(2*sqrt(x))
        CHECK_UNIFIED_DERIVATIVES_FX(arena, 4.0, sqrt(x), std::sqrt(4.0), 1.0 / (2.0 * std::sqrt(4.0)));

        x = UnifiedVariable<double>(arena, 3.0);
        // f(x) = 1/x, f'(x) = -1/x²
        CHECK_UNIFIED_DERIVATIVES_FX(arena, 3.0, 1.0 / x, 1.0 / 3.0, -1.0 / (3.0 * 3.0));
    }

    SECTION("testing optimization-type functions - first order")
    {
        // Extended Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²
        x = UnifiedVariable<double>(arena, 1.0);
        y = UnifiedVariable<double>(arena, 1.0);
        
        // At (1,1), this should be the global minimum with zero gradient
        CHECK_UNIFIED_DERIVATIVES_FXY(arena, 1.0, 1.0, (1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x), 0.0, 0.0, 0.0);

        // Test at a different point
        x = UnifiedVariable<double>(arena, 0.5);
        y = UnifiedVariable<double>(arena, 0.25);
        CHECK_UNIFIED_DERIVATIVES_FXY(arena, 0.5, 0.25, (1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x),
            (1 - 0.5) * (1 - 0.5) + 100 * (0.25 - 0.5 * 0.5) * (0.25 - 0.5 * 0.5),
            -2 * (1 - 0.5) - 400 * 0.5 * (0.25 - 0.5 * 0.5),
            200 * (0.25 - 0.5 * 0.5));
    }

    SECTION("testing neural network-like functions - first order")
    {
        // Simple neural network layer: tanh(w1*x1 + w2*x2 + w3*x3 + bias)
        UnifiedVariable<double> x1(arena, 0.5), x2(arena, -0.3), x3(arena, 0.8);
        double w1 = 1.2, w2 = -0.7, w3 = 0.4, bias = 0.1;
        
        // Test individual derivatives using simple functions
        auto layer1 = tanh(w1 * x1 + w2 * (-0.3) + w3 * 0.8 + bias);
        auto grad_x1 = derivatives(layer1, wrt(x1));
        CHECK( std::isfinite(layer1.value()) );
        CHECK( std::isfinite(grad_x1[0]) );

        // Multi-layer composition - using simpler approach
        auto layer1_full = tanh(w1 * x1 + w2 * x2 + w3 * x3 + bias);
        auto layer2 = sin(layer1_full * 2.0 + 0.5);
        auto output = exp(-layer2 * layer2); // Gaussian-like output
        
        // Test that derivatives are computed and are finite
        auto grad_out = derivatives(output, wrt(x1));
        
        CHECK( std::isfinite(output.value()) );
        CHECK( output.value() > 0.0 ); // exp is always positive
        CHECK( std::isfinite(grad_out[0]) );
    }
}

// Test unified reverse mode equivalents
TEST_CASE("testing unified reverse mode equivalents of higher-order patterns", "[reverse][unified][first-order-equivalents]")
{
    auto arena = std::make_shared<ExpressionArena<double>>();

    SECTION("testing complex expressions - reverse mode")
    {
        UnifiedVariable<double> x(arena, 1.2), y(arena, 0.8);
        
        // Complex expression that would normally be tested at higher orders
        auto f = sin(x*y)*cos(x/y) + exp(x-y)*log(x+y) + sqrt(x*x + y*y);
        
        auto grad_x = derivatives(f, wrt(x));
        auto grad_y = derivatives(f, wrt(y));
        
        // Verify results are finite
        CHECK( std::isfinite(f.value()) );
        CHECK( std::isfinite(grad_x[0]) );
        CHECK( std::isfinite(grad_y[0]) );
        
        // Test that derivatives have reasonable magnitudes
        CHECK( std::abs(grad_x[0]) < 1000.0 );
        CHECK( std::abs(grad_y[0]) < 1000.0 );
    }

    SECTION("testing polynomial expressions - reverse mode")
    {
        UnifiedVariable<double> x(arena, 1.5), y(arena, 2.5);
        
        // High-degree polynomial
        auto poly = x*x*x*x + 4*x*x*x*y + 6*x*x*y*y + 4*x*y*y*y + y*y*y*y;
        
        auto grad_x = derivatives(poly, wrt(x));
        auto grad_y = derivatives(poly, wrt(y));
        
        // This is (x+y)⁴, so:
        // df/dx = 4(x+y)³ = 4*4³ = 256
        // df/dy = 4(x+y)³ = 4*4³ = 256
        CHECK( poly.value() == approx(std::pow(1.5+2.5, 4)) );
        CHECK( grad_x[0] == approx(4*std::pow(1.5+2.5, 3)) );
        CHECK( grad_y[0] == approx(4*std::pow(1.5+2.5, 3)) );
    }

    SECTION("testing optimization functions - reverse mode")
    {
        // Functions commonly used in optimization that would benefit from testing
        
        UnifiedVariable<double> x(arena, 1.0), y(arena, 1.0);
        
        // Extended Rosenbrock function
        auto rosenbrock = (1-x)*(1-x) + 100*(y-x*x)*(y-x*x);
        
        auto grad_ros_x = derivatives(rosenbrock, wrt(x));
        auto grad_ros_y = derivatives(rosenbrock, wrt(y));
        
        // At (1,1), this should be the global minimum with zero gradient
        CHECK( rosenbrock.value() == approx(0.0) );
        CHECK( grad_ros_x[0] == approx(0.0).margin(1e-10) );
        CHECK( grad_ros_y[0] == approx(0.0).margin(1e-10) );
        
        // Beale function: f(x,y) = (1.5 - x + xy)² + (2.25 - x + xy²)² + (2.625 - x + xy³)²
        auto beale = (1.5 - x + x*y)*(1.5 - x + x*y) + 
                   (2.25 - x + x*y*y)*(2.25 - x + x*y*y) + 
                   (2.625 - x + x*y*y*y)*(2.625 - x + x*y*y*y);
        
        auto grad_beale_x = derivatives(beale, wrt(x));
        auto grad_beale_y = derivatives(beale, wrt(y));
        
        CHECK( std::isfinite(beale.value()) );
        CHECK( std::isfinite(grad_beale_x[0]) );
        CHECK( std::isfinite(grad_beale_y[0]) );
    }

    SECTION("testing neural network-like functions - reverse mode")
    {
        // Functions that resemble neural network computations
        
        UnifiedVariable<double> x1(arena, 0.5), x2(arena, -0.3), x3(arena, 0.8);
        
        // Simple neural network layer: tanh(w1*x1 + w2*x2 + w3*x3 + bias)
        double w1 = 1.2, w2 = -0.7, w3 = 0.4, bias = 0.1;
        auto layer1 = tanh(w1*x1 + w2*x2 + w3*x3 + bias);
        
        auto grad_l1_x1 = derivatives(layer1, wrt(x1));
        auto grad_l1_x2 = derivatives(layer1, wrt(x2));
        auto grad_l1_x3 = derivatives(layer1, wrt(x3));
        
        // Check that gradients have expected signs and magnitudes
        CHECK( std::isfinite(layer1.value()) );
        CHECK( (layer1.value() >= -1.0 && layer1.value() <= 1.0) ); // tanh range
        
        // Gradient w.r.t. x1 should have same sign as w1
        CHECK( grad_l1_x1[0] * w1 > 0 );
        // Gradient w.r.t. x2 should have same sign as w2
        CHECK( grad_l1_x2[0] * w2 > 0 );
        // Gradient w.r.t. x3 should have same sign as w3
        CHECK( grad_l1_x3[0] * w3 > 0 );
        
        // Multi-layer composition
        auto layer2 = sin(layer1 * 2.0 + 0.5);
        auto output = exp(-layer2 * layer2); // Gaussian-like output
        
        auto grad_out_x1 = derivatives(output, wrt(x1));
        
        CHECK( std::isfinite(output.value()) );
        CHECK( output.value() > 0.0 ); // exp is always positive
        CHECK( std::isfinite(grad_out_x1[0]) );
    }
}

// Test complex mathematical function compositions with unified implementation
