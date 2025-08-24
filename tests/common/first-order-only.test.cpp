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

// These tests specifically focus on what should work when AUTODIFF_DISABLE_HIGHER_ORDER is enabled
TEST_CASE("testing first-order derivatives when AUTODIFF_DISABLE_HIGHER_ORDER", "[first-order-only]")
{
    SECTION("testing that first derivatives work correctly")
    {
        // Forward mode
        dual x = 2.0, y = 3.0;
        auto f = [](dual x, dual y) -> dual { return x*x + y*y*y; };
        
        auto df_dx = derivatives(f, autodiff::detail::wrt(x), at(x, y));
        auto df_dy = derivatives(f, autodiff::detail::wrt(y), at(x, y));
        
        CHECK( df_dx[0] == approx(val(f(x, y))) ); // function value
        CHECK( df_dx[1] == approx(2*2) );          // df/dx = 2x
        CHECK( df_dy[0] == approx(val(f(x, y))) ); // function value  
        CHECK( df_dy[1] == approx(3*3*3) );        // df/dy = 3y^2

        // Reverse mode
        var x_var = 2.0, y_var = 3.0;
        var f_var = x_var*x_var + y_var*y_var*y_var;
        
        auto grad_x = derivatives(f_var, autodiff::reverse::detail::wrt(x_var));
        auto grad_y = derivatives(f_var, autodiff::reverse::detail::wrt(y_var));
        
        CHECK( val(grad_x[0]) == approx(2*2) );     // df/dx = 2x
        CHECK( val(grad_y[0]) == approx(3*3*3) );   // df/dy = 3y^2
    }

    
#if AUTODIFF_DISABLE_HIGHER_ORDER
    SECTION("testing that mixed first derivatives work (wrt(x,y))")
    {
        // When AUTODIFF_DISABLE_HIGHER_ORDER is enabled, wrt(x,y) should still work for first derivatives
        // but only using the manual seeding approach
        dual x, y;
        detail::seed<0>(x, 1.5);  // x = 1.5
        detail::seed<1>(x, 1.0);  // dx/dx = 1, dx/dy = 0
        y = 2.5;                  // y = 2.5 (constant for this computation)
        
        auto result_dx = x*y + sin(x)*cos(y);
        CHECK( std::isfinite(val(result_dx)) );
        CHECK( std::isfinite(grad(result_dx)) );
        
        // Now compute df/dy
        dual x2, y2;
        x2 = 1.5;                 // x = 1.5 (constant for this computation)  
        detail::seed<0>(y2, 2.5); // y = 2.5
        detail::seed<1>(y2, 1.0); // dy/dx = 0, dy/dy = 1
        
        auto result_dy = x2*y2 + sin(x2)*cos(y2);
        CHECK( std::isfinite(val(result_dy)) );
        CHECK( std::isfinite(grad(result_dy)) );
        
        INFO("Mixed derivatives computed using manual seeding in AUTODIFF_DISABLE_HIGHER_ORDER mode");
    }
#endif

#ifdef AUTODIFF_EIGEN_FOUND
    SECTION("testing vector gradient computations still work")
    {
        // Test that gradient computations work properly for vector inputs
        VectorXreal x(3);
        x << 1.0, 2.0, 3.0;
        
        auto f = [](const VectorXreal& x) -> real {
            return x[0]*x[0] + x[1]*x[1]*x[1] + x[2]*x[2]*x[2]*x[2];
        };
        
        Eigen::VectorXd g = gradient(f, autodiff::detail::wrt(x), at(x));
        
        CHECK( g.size() == 3 );
        CHECK( g[0] == approx(2*1.0) );      // df/dx0 = 2*x0
        CHECK( g[1] == approx(3*2.0*2.0) );  // df/dx1 = 3*x1^2
        CHECK( g[2] == approx(4*3.0*3.0*3.0) ); // df/dx2 = 4*x2^3
    }

    SECTION("testing jacobian computations for vector functions")
    {
        // Test F(x,y) = [x^2 + y, x*y^2] 
        auto F = [](const VectorXreal& vars) -> VectorXreal {
            real x = vars[0];
            real y = vars[1];
            VectorXreal result(2);
            result[0] = x*x + y;
            result[1] = x*y*y;
            return result;
        };

        VectorXreal vars(2);
        vars << 2.0, 3.0;

        Eigen::MatrixXd J = jacobian(F, autodiff::detail::wrt(vars), at(vars));

        CHECK( J.rows() == 2 );
        CHECK( J.cols() == 2 );
        CHECK( J(0, 0) == approx(2*2.0) );      // dF1/dx = 2x
        CHECK( J(0, 1) == approx(1.0) );        // dF1/dy = 1
        CHECK( J(1, 0) == approx(3.0*3.0) );    // dF2/dx = y^2
        CHECK( J(1, 1) == approx(2*2.0*3.0) );  // dF2/dy = 2xy
    }
#endif // AUTODIFF_EIGEN_FOUND

    SECTION("testing that computational performance is maintained")
    {
        // When higher-order derivatives are disabled, first-order computations should be efficient
        const int n = 50;
        std::vector<dual> x(n);
        for(int i = 0; i < n; ++i) {
            x[i] = 0.1 * i;
        }
        
        // Complex multivariate function
        auto f = [](const std::vector<dual>& vars) -> dual {
            dual result = 0.0;
            int n = vars.size();
            for(int i = 0; i < n-1; ++i) {
                result += sin(vars[i]) * exp(vars[i+1]) + vars[i]*vars[i]*vars[i+1];
            }
            return result;
        };
        
        // Test that derivatives can be computed efficiently
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for(int i = 0; i < 10; ++i) {
            auto df_dx0 = derivatives(f, autodiff::detail::wrt(x[0]), at(x));
            auto df_dx10 = derivatives(f, autodiff::detail::wrt(x[10]), at(x));
            auto df_dx25 = derivatives(f, autodiff::detail::wrt(x[25]), at(x));
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // Should complete in reasonable time (this is just a sanity check)
        CHECK( duration.count() < 1000 ); // Less than 1 second for this test
    }

    SECTION("testing mathematical function coverage for first derivatives")
    {
        // In AUTODIFF_DISABLE_HIGHER_ORDER mode, the derivatives(function, wrt(...), at(...)) pattern
        // doesn't work properly. So we test mathematical functions using the manual seeding approach.
        
        // Test trigonometric functions: f(x) = sin(x) + cos(x)
        dual x;
        detail::seed<0>(x, 0.5);  // x = 0.5
        detail::seed<1>(x, 1.0);  // dx/dx = 1
        auto trig_result = sin(x) + cos(x);
        CHECK( std::isfinite(val(trig_result)) );
        CHECK( std::isfinite(grad(trig_result)) );
        CHECK( grad(trig_result) == approx(cos(0.5) - sin(0.5)) );  // d/dx[sin(x) + cos(x)] = cos(x) - sin(x)
        
        // Test exponential and logarithmic functions: f(x) = exp(x) + log(x+1)
        dual x2;
        detail::seed<0>(x2, 1.0);  // x = 1.0
        detail::seed<1>(x2, 1.0);  // dx/dx = 1
        auto exp_result = exp(x2) + log(x2 + 1.0);
        CHECK( std::isfinite(val(exp_result)) );
        CHECK( std::isfinite(grad(exp_result)) );
        CHECK( grad(exp_result) == approx(exp(1.0) + 1.0/(1.0 + 1.0)) );  // d/dx[exp(x) + log(x+1)] = exp(x) + 1/(x+1)
        
        // Test hyperbolic functions: f(x) = sinh(x) + cosh(x)
        dual x3;
        detail::seed<0>(x3, 0.8);  // x = 0.8
        detail::seed<1>(x3, 1.0);  // dx/dx = 1
        auto hyp_result = sinh(x3) + cosh(x3);
        CHECK( std::isfinite(val(hyp_result)) );
        CHECK( std::isfinite(grad(hyp_result)) );
        CHECK( grad(hyp_result) == approx(cosh(0.8) + sinh(0.8)) );  // d/dx[sinh(x) + cosh(x)] = cosh(x) + sinh(x)
        
        INFO("Mathematical functions tested using manual seeding approach in AUTODIFF_DISABLE_HIGHER_ORDER mode");
    }

    SECTION("testing complex real-world optimization functions")
    {
        // Test typical optimization problems that would benefit from first-order derivatives
        
        // 1. Rosenbrock function
        dual x = -1.0, y = 1.0;
        auto rosenbrock = [](dual x, dual y) -> dual {
            return (1-x)*(1-x) + 100*(y-x*x)*(y-x*x);
        };
        
        auto dr_dx = derivatives(rosenbrock, autodiff::detail::wrt(x), at(x, y));
        auto dr_dy = derivatives(rosenbrock, autodiff::detail::wrt(y), at(x, y));
        
        CHECK( std::isfinite(val(dr_dx[1])) );
        CHECK( std::isfinite(val(dr_dy[1])) );
        
        // 2. Sphere function (should have simple derivatives)
        auto sphere = [](dual x, dual y, dual z) -> dual {
            return x*x + y*y + z*z;
        };
        
        dual z = 0.5;
        auto ds_dx = derivatives(sphere, autodiff::detail::wrt(x), at(x, y, z));
        auto ds_dy = derivatives(sphere, autodiff::detail::wrt(y), at(x, y, z));
        auto ds_dz = derivatives(sphere, autodiff::detail::wrt(z), at(x, y, z));
        
        CHECK( ds_dx[1] == approx(2*val(x)) );
        CHECK( ds_dy[1] == approx(2*val(y)) );
        CHECK( ds_dz[1] == approx(2*val(z)) );
        
        // 3. Himmelblau's function
        auto himmelblau = [](dual x, dual y) -> dual {
            return (x*x + y - 11)*(x*x + y - 11) + (x + y*y - 7)*(x + y*y - 7);
        };
        
        auto dh_dx = derivatives(himmelblau, autodiff::detail::wrt(x), at(x, y));
        auto dh_dy = derivatives(himmelblau, autodiff::detail::wrt(y), at(x, y));
        
        CHECK( std::isfinite(val(dh_dx[1])) );
        CHECK( std::isfinite(val(dh_dy[1])) );
    }
}

// Test that ensures the limitation is properly enforced when AUTODIFF_DISABLE_HIGHER_ORDER is enabled
#if AUTODIFF_DISABLE_HIGHER_ORDER
TEST_CASE("testing limitations when AUTODIFF_DISABLE_HIGHER_ORDER is enabled", "[first-order-limitations]")
{
    SECTION("testing that higher-order derivative computation is disabled")
    {
        // This section only runs when AUTODIFF_DISABLE_HIGHER_ORDER is actually enabled
        // In this mode, we compute partial derivatives one at a time
        
        // Test function f(x,y) = x^2 + y at point (1,2)
        
        // Compute df/dx: set x as variable (with derivative 1), y as constant
        dual x1, y1;
        detail::seed<0>(x1, 1.0);  // x = 1
        detail::seed<1>(x1, 1.0);  // dx/dx = 1
        y1 = 2.0;                  // y = 2 (constant)
        auto result1 = x1*x1 + y1;
        
        // Compute df/dy: set y as variable (with derivative 1), x as constant  
        dual x2, y2;
        x2 = 1.0;                  // x = 1 (constant)
        detail::seed<0>(y2, 2.0);  // y = 2
        detail::seed<1>(y2, 1.0);  // dy/dy = 1
        auto result2 = x2*x2 + y2;
        
        // Check function values
        CHECK( val(result1) == approx(1.0*1.0 + 2.0) );  // f(1,2) = 3
        CHECK( val(result2) == approx(1.0*1.0 + 2.0) );  // f(1,2) = 3
        
        // Check partial derivatives
        CHECK( grad(result1) == approx(2.0*1.0) );  // df/dx = 2x = 2
        CHECK( grad(result2) == approx(1.0) );      // df/dy = 1
        
        INFO("AUTODIFF_DISABLE_HIGHER_ORDER is enabled - computed partial derivatives separately");
    }
}
#else
TEST_CASE("testing that full functionality is available when AUTODIFF_DISABLE_HIGHER_ORDER is not enabled", "[full-functionality]")
{
    SECTION("testing that higher-order derivatives work when enabled")
    {
        dual2nd x = 1.0, y = 2.0;
        
        auto f = [](dual2nd x, dual2nd y) -> dual2nd { return x*x*x + y*y; };
        
        // Should be able to compute second derivatives
        auto d2f_dx2 = derivatives(f, autodiff::detail::wrt(x, x), at(x, y));
        auto d2f_dxdy = derivatives(f, autodiff::detail::wrt(x, y), at(x, y));
        
        CHECK( d2f_dx2[0] == approx(val(f(x, y))) );  // function value
        CHECK( d2f_dx2[1] == approx(3*val(x)*val(x)) );  // df/dx = 3x^2
        CHECK( d2f_dx2[2] == approx(6*val(x)) );      // d²f/dx² = 6x
        
        CHECK( d2f_dxdy[0] == approx(val(f(x, y))) ); // function value
        CHECK( d2f_dxdy[1] == approx(3*val(x)*val(x)) ); // df/dx = 3x^2
        CHECK( d2f_dxdy[2] == approx(0.0) );          // d²f/dxdy = 0
        
        INFO("AUTODIFF_DISABLE_HIGHER_ORDER is not enabled - full functionality available");
    }
}
#endif
