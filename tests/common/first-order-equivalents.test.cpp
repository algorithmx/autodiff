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
#include <autodiff/forward/dual.hpp>
#include <autodiff/reverse/var.hpp>
using namespace autodiff;

template<typename T>
auto approx(T&& expr) -> Catch::Approx
{
    return Catch::Approx(val(std::forward<T>(expr))).margin(1e-12);
}

#define CHECK_DERIVATIVES_FX(expr, u, ux)         \
{                                                 \
    auto f = [](dual x) -> dual { return expr; }; \
    auto dfdx = derivatives(f, autodiff::detail::wrt(x), autodiff::detail::at(x));    \
    CHECK( dfdx[0] == approx(val(u)) );           \
    CHECK( dfdx[1] == approx(val(ux)) );          \
}

#define CHECK_DERIVATIVES_FXY(expr, u, ux, uy)            \
{                                                         \
    auto f = [](dual x, dual y) -> dual { return expr; }; \
    auto dfdx = derivatives(f, autodiff::detail::wrt(x), autodiff::detail::at(x, y));         \
    CHECK( dfdx[0] == approx(val(u)) );                   \
    CHECK( dfdx[1] == approx(val(ux)) );                  \
    auto dfdy = derivatives(f, autodiff::detail::wrt(y), autodiff::detail::at(x, y));         \
    CHECK( dfdy[0] == approx(val(u)) );                   \
    CHECK( dfdy[1] == approx(val(uy)) );                  \
}

#define CHECK_DERIVATIVES_FXYZ(expr, u, ux, uy, uz)               \
{                                                                 \
    auto f = [](dual x, dual y, dual z) -> dual { return expr; }; \
    auto dfdx = derivatives(f, autodiff::detail::wrt(x), autodiff::detail::at(x, y, z));              \
    CHECK( dfdx[0] == approx(val(u)) );                           \
    CHECK( dfdx[1] == approx(val(ux)) );                          \
    auto dfdy = derivatives(f, autodiff::detail::wrt(y), autodiff::detail::at(x, y, z));              \
    CHECK( dfdy[0] == approx(val(u)) );                           \
    CHECK( dfdy[1] == approx(val(uy)) );                          \
    auto dfdz = derivatives(f, autodiff::detail::wrt(z), autodiff::detail::at(x, y, z));              \
    CHECK( dfdz[0] == approx(val(u)) );                           \
    CHECK( dfdz[1] == approx(val(uz)) );                          \
}

// This file provides equivalent first-order tests for functionality that would normally
// be tested in higher-order test sections that get disabled when AUTODIFF_DISABLE_HIGHER_ORDER is enabled.
// These tests ensure comprehensive coverage of first-order derivatives regardless of compilation flags.

TEST_CASE("testing first-order equivalents of higher-order patterns", "[first-order-equivalents]")
{
    dual x = 1.0;
    dual y = 2.0;
    dual z = 3.0;

    SECTION("testing complex mathematical expressions - first order")
    {
        x = 0.5;
        y = 0.8;

        // Complex trigonometric expression
        CHECK_DERIVATIVES_FXY(sin(x + y) * cos(x / y) + tan(2.0 * x * y), 
            sin(val(x) + val(y)) * cos(val(x) / val(y)) + tan(2.0 * val(x) * val(y)),
            cos(x + y) * cos(x / y) - sin(x + y) * sin(x / y) / y + 2.0 * y / (cos(2.0 * x * y) * cos(2.0 * x * y)),
            cos(x + y) * cos(x / y) + sin(x + y) * sin(x / y) * x / (y * y) + 2.0 * x / (cos(2.0 * x * y) * cos(2.0 * x * y)));

        // Complex exponential/logarithmic expression  
        x = 1.5;
        y = 2.5;
        CHECK_DERIVATIVES_FXY(log(x + y) * exp(x / y) + sqrt(2.0 * x * y),
            log(val(x) + val(y)) * exp(val(x) / val(y)) + sqrt(2.0 * val(x) * val(y)),
            exp(x / y) / (x + y) + log(x + y) * exp(x / y) / y + y / sqrt(2.0 * x * y),
            exp(x / y) / (x + y) - log(x + y) * exp(x / y) * x / (y * y) + x / sqrt(2.0 * x * y));
    }

    SECTION("testing polynomial expressions - first order")
    {
        x = 1.0;
        y = 2.0;

        // Cubic polynomial: f(x,y) = (x+y)³
        CHECK_DERIVATIVES_FXY((x + y) * (x + y) * (x + y),
            (val(x) + val(y)) * (val(x) + val(y)) * (val(x) + val(y)),
            3.0 * (x + y) * (x + y),
            3.0 * (x + y) * (x + y));

        // Quartic polynomial: f(x,y) = (x+y)⁴
        x = 1.5;
        y = 0.5;
        CHECK_DERIVATIVES_FXY((x + y) * (x + y) * (x + y) * (x + y),
            (val(x) + val(y)) * (val(x) + val(y)) * (val(x) + val(y)) * (val(x) + val(y)),
            4.0 * (x + y) * (x + y) * (x + y),
            4.0 * (x + y) * (x + y) * (x + y));
    }

    SECTION("testing mathematical function compositions - first order")
    {
        x = 0.5;

        // Composition: f(x) = sin(sqrt(x))
        CHECK_DERIVATIVES_FX(sin(sqrt(x)),
            sin(sqrt(val(x))),
            cos(sqrt(x)) / (2.0 * sqrt(x)));

        // Another composition: f(x) = log(exp(x²))
        // This should simplify to x²
        CHECK_DERIVATIVES_FX(log(exp(x * x)),
            val(x) * val(x),
            2.0 * x);

        // More complex composition: f(x) = exp(sin(log(x)))
        x = 2.0;
        CHECK_DERIVATIVES_FX(exp(sin(log(x))),
            exp(sin(log(val(x)))),
            exp(sin(log(x))) * cos(log(x)) / x);
    }

    SECTION("testing expression node handling - first order")
    {
        x = 2.0;
        y = 3.0;

        // Complex expression that creates multiple intermediate nodes
        CHECK_DERIVATIVES_FXY(((x + y) * (x - y)) / (x * y) + sqrt(x + y), 
            ((val(x) + val(y)) * (val(x) - val(y))) / (val(x) * val(y)) + sqrt(val(x) + val(y)),
            // df/dx for ((x+y)(x-y))/(xy) + sqrt(x+y)
            ((x - y + x + y) * (x * y) - (x + y) * (x - y) * y) / (x * y * x * y) + 1.0 / (2.0 * sqrt(x + y)),
            // df/dy for ((x+y)(x-y))/(xy) + sqrt(x+y) 
            - x / (y * y) - 1.0 / x + 1.0 / (2.0 * sqrt(x + y)));
    }

    SECTION("testing equivalence with analytical derivatives - first order")
    {
        // Test various functions with known analytical derivatives
        x = 2.0;

        // f(x) = x³, f'(x) = 3x²
        CHECK_DERIVATIVES_FX(x * x * x, val(x) * val(x) * val(x), 3.0 * x * x);

        x = 0.7;
        // f(x) = sin(x), f'(x) = cos(x)
        CHECK_DERIVATIVES_FX(sin(x), sin(val(x)), cos(x));

        x = 1.5;
        // f(x) = exp(x), f'(x) = exp(x)
        CHECK_DERIVATIVES_FX(exp(x), exp(val(x)), exp(x));

        x = 2.5;
        // f(x) = log(x), f'(x) = 1/x
        CHECK_DERIVATIVES_FX(log(x), log(val(x)), 1.0 / x);

        x = 4.0;
        // f(x) = sqrt(x), f'(x) = 1/(2*sqrt(x))
        CHECK_DERIVATIVES_FX(sqrt(x), sqrt(val(x)), 1.0 / (2.0 * sqrt(x)));

        x = 3.0;
        // f(x) = 1/x, f'(x) = -1/x²
        CHECK_DERIVATIVES_FX(1.0 / x, 1.0 / val(x), -1.0 / (x * x));
    }

    SECTION("testing optimization-type functions - first order")
    {
        // Extended Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²
        x = 1.0;
        y = 1.0;
        
        // At (1,1), this should be the global minimum with zero gradient
        CHECK_DERIVATIVES_FXY((1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x), 0.0, 0.0, 0.0);

        // Test at a different point
        x = 0.5;
        y = 0.25;
        CHECK_DERIVATIVES_FXY((1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x),
            (1 - val(x)) * (1 - val(x)) + 100 * (val(y) - val(x) * val(x)) * (val(y) - val(x) * val(x)),
            -2 * (1 - x) - 400 * x * (y - x * x),
            200 * (y - x * x));
    }

    SECTION("testing neural network-like functions - first order")
    {
                // Simple neural network layer: tanh(w1*x1 + w2*x2 + w3*x3 + bias)
        dual x1 = 0.5, x2 = -0.3, x3 = 0.8;
        double w1 = 1.2, w2 = -0.7, w3 = 0.4, bias = 0.1;
        
        // Test individual derivatives using simple functions
        auto f_x1 = [w1, w2, w3, bias](dual x1) -> dual { 
            return tanh(w1 * x1 + w2 * (-0.3) + w3 * 0.8 + bias);
        };
        auto grad_x1 = derivatives(f_x1, autodiff::detail::wrt(x1), autodiff::detail::at(x1));
        CHECK( std::isfinite(grad_x1[0]) );
        CHECK( std::isfinite(grad_x1[1]) );

        // Multi-layer composition - using simpler approach
        auto layer1 = tanh(w1 * x1 + w2 * x2 + w3 * x3 + bias);
        auto layer2 = sin(layer1 * 2.0 + 0.5);
        auto output = exp(-layer2 * layer2); // Gaussian-like output
        
        // Test that derivatives are computed and are finite
        auto f_output = [w1, w2, w3, bias](dual x1) -> dual { 
            auto l1 = tanh(w1 * x1 + w2 * (-0.3) + w3 * 0.8 + bias);
            auto l2 = sin(l1 * 2.0 + 0.5);
            return exp(-l2 * l2);
        };
        auto grad_out = derivatives(f_output, autodiff::detail::wrt(x1), autodiff::detail::at(x1));
        
        CHECK( std::isfinite(grad_out[0]) );
        CHECK( grad_out[0] > 0.0 ); // exp is always positive
        CHECK( std::isfinite(grad_out[1]) );
    }
}

// Test reverse mode equivalents
TEST_CASE("testing reverse mode equivalents of higher-order patterns", "[reverse][first-order-equivalents]")
{
    SECTION("testing complex expressions - reverse mode")
    {
        var x = 1.2, y = 0.8;
        
        // Complex expression that would normally be tested at higher orders
        var f = sin(x*y)*cos(x/y) + exp(x-y)*log(x+y) + sqrt(x*x + y*y);
        
        auto grad_x = derivatives(f, autodiff::reverse::detail::wrt(x));
        auto grad_y = derivatives(f, autodiff::reverse::detail::wrt(y));
        
        // Verify results are finite
        CHECK( std::isfinite(val(f)) );
        CHECK( std::isfinite(grad_x[0]) );
        CHECK( std::isfinite(grad_y[0]) );
        
        // Test that derivatives have reasonable magnitudes
        CHECK( abs(grad_x[0]) < 1000.0 );
        CHECK( abs(grad_y[0]) < 1000.0 );
    }

    SECTION("testing polynomial expressions - reverse mode")
    {
        var x = 1.5, y = 2.5;
        
        // High-degree polynomial
        var poly = x*x*x*x + 4*x*x*x*y + 6*x*x*y*y + 4*x*y*y*y + y*y*y*y;
        
        auto grad_x = derivatives(poly, autodiff::reverse::detail::wrt(x));
        auto grad_y = derivatives(poly, autodiff::reverse::detail::wrt(y));
        
        // This is (x+y)⁴, so:
        // df/dx = 4(x+y)³ = 4*4³ = 256
        // df/dy = 4(x+y)³ = 4*4³ = 256
        CHECK( val(poly) == approx(pow(1.5+2.5, 4)) );
        CHECK( grad_x[0] == approx(4*pow(1.5+2.5, 3)) );
        CHECK( grad_y[0] == approx(4*pow(1.5+2.5, 3)) );
    }

    SECTION("testing optimization functions - reverse mode")
    {
        // Functions commonly used in optimization that would benefit from testing
        
        var x = 1.0, y = 1.0;
        
        // Extended Rosenbrock function
        var rosenbrock = (1-x)*(1-x) + 100*(y-x*x)*(y-x*x);
        
        auto grad_ros_x = derivatives(rosenbrock, autodiff::reverse::detail::wrt(x));
        auto grad_ros_y = derivatives(rosenbrock, autodiff::reverse::detail::wrt(y));
        
        // At (1,1), this should be the global minimum with zero gradient
        CHECK( val(rosenbrock) == approx(0.0) );
        CHECK( grad_ros_x[0] == approx(0.0).margin(1e-10) );
        CHECK( grad_ros_y[0] == approx(0.0).margin(1e-10) );
        
        // Beale function: f(x,y) = (1.5 - x + xy)² + (2.25 - x + xy²)² + (2.625 - x + xy³)²
        var beale = (1.5 - x + x*y)*(1.5 - x + x*y) + 
                   (2.25 - x + x*y*y)*(2.25 - x + x*y*y) + 
                   (2.625 - x + x*y*y*y)*(2.625 - x + x*y*y*y);
        
        auto grad_beale_x = derivatives(beale, autodiff::reverse::detail::wrt(x));
        auto grad_beale_y = derivatives(beale, autodiff::reverse::detail::wrt(y));
        
        CHECK( std::isfinite(val(beale)) );
        CHECK( std::isfinite(grad_beale_x[0]) );
        CHECK( std::isfinite(grad_beale_y[0]) );
    }

    SECTION("testing neural network-like functions - reverse mode")
    {
        // Functions that resemble neural network computations
        
        var x1 = 0.5, x2 = -0.3, x3 = 0.8;
        
        // Simple neural network layer: tanh(w1*x1 + w2*x2 + w3*x3 + bias)
        double w1 = 1.2, w2 = -0.7, w3 = 0.4, bias = 0.1;
        var layer1 = tanh(w1*x1 + w2*x2 + w3*x3 + bias);
        
        auto grad_l1_x1 = derivatives(layer1, autodiff::reverse::detail::wrt(x1));
        auto grad_l1_x2 = derivatives(layer1, autodiff::reverse::detail::wrt(x2));
        auto grad_l1_x3 = derivatives(layer1, autodiff::reverse::detail::wrt(x3));
        
        // Check that gradients have expected signs and magnitudes
        CHECK( std::isfinite(val(layer1)) );
        CHECK( (val(layer1) >= -1.0 && val(layer1) <= 1.0) ); // tanh range
        
        // Gradient w.r.t. x1 should have same sign as w1
        CHECK( grad_l1_x1[0] * w1 > 0 );
        // Gradient w.r.t. x2 should have same sign as w2
        CHECK( grad_l1_x2[0] * w2 > 0 );
        // Gradient w.r.t. x3 should have same sign as w3
        CHECK( grad_l1_x3[0] * w3 > 0 );
        
        // Multi-layer composition
        var layer2 = sin(layer1 * 2.0 + 0.5);
        var output = exp(-layer2 * layer2); // Gaussian-like output
        
        auto grad_out_x1 = derivatives(output, autodiff::reverse::detail::wrt(x1));
        
        CHECK( std::isfinite(val(output)) );
        CHECK( val(output) > 0.0 ); // exp is always positive
        CHECK( std::isfinite(grad_out_x1[0]) );
    }
}
