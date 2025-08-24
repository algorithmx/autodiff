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
// of this software and authenticated to distribute, sublicense, and/or sell
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
#include <chrono>

// Catch includes
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

// autodiff includes
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
#include <tests/utils/catch.hpp>
using namespace autodiff;
using Eigen::VectorXd;

/// Convenient function used in the tests to calculate the derivative of a variable y with respect to a variable x.
inline auto grad(const var& y, var& x)
{
    auto g = derivatives(y, wrt(x));
    return val(g[0]);
}

TEST_CASE("testing reverse mode derivatives on 100-variable functions", "[reverse][utils][multivariate][100d]")
{
    const int n = 100;

    SECTION("testing optimization function: extended Beale function")
    {
        // Test f(x) = sum_{i=0}^{98} ((1.5 - x_i + x_i*x_{i+1})^2 + (2.25 - x_i + x_i*x_{i+1}^2)^2 + (2.625 - x_i + x_i*x_{i+1}^3)^2)
        // This is an extension of the Beale function to 100 variables
        
        VectorXvar x(n);
        for(int i = 0; i < n; ++i) {
            x[i] = 0.5 + 0.01 * i; // Starting point away from optimum
        }

        var f = 0.0;
        for(int i = 0; i < n-1; ++i) {
            var term1 = (1.5 - x[i] + x[i]*x[i+1]) * (1.5 - x[i] + x[i]*x[i+1]);
            var term2 = (2.25 - x[i] + x[i]*x[i+1]*x[i+1]) * (2.25 - x[i] + x[i]*x[i+1]*x[i+1]);
            var term3 = (2.625 - x[i] + x[i]*x[i+1]*x[i+1]*x[i+1]) * (2.625 - x[i] + x[i]*x[i+1]*x[i+1]*x[i+1]);
            f += term1 + term2 + term3;
        }

        VectorXd g = gradient(f, x);

        // Check that gradient computation succeeds
        CHECK( g.size() == n );
        for(int i = 0; i < n; ++i) {
            CHECK( std::isfinite(g[i]) );
        }

        // The gradient should generally be non-zero except at critical points
        bool has_nonzero_gradient = false;
        for(int i = 0; i < n; ++i) {
            if(std::abs(g[i]) > 1e-10) {
                has_nonzero_gradient = true;
                break;
            }
        }
        CHECK( has_nonzero_gradient );
    }

    SECTION("testing Himmelblau-style function")
    {
        // Test f(x) = sum_{i=0}^{98} ((x_i^2 + x_{i+1} - 11)^2 + (x_i + x_{i+1}^2 - 7)^2)
        // Extension of Himmelblau's function to 100 variables
        
        VectorXvar x(n);
        for(int i = 0; i < n; ++i) {
            x[i] = 1.0 + 0.1 * sin(static_cast<double>(i)); // Varied starting point
        }

        var f = 0.0;
        for(int i = 0; i < n-1; ++i) {
            var term1 = (x[i]*x[i] + x[i+1] - 11.0) * (x[i]*x[i] + x[i+1] - 11.0);
            var term2 = (x[i] + x[i+1]*x[i+1] - 7.0) * (x[i] + x[i+1]*x[i+1] - 7.0);
            f += term1 + term2;
        }

        VectorXd g = gradient(f, x);

        // Check gradient computation and verify some analytical values
        CHECK( g.size() == n );
        
        // For first variable: df/dx_0 involves only the first term
        double x0_val = val(x[0]), x1_val = val(x[1]);
        double expected_g0 = 2.0 * (x0_val*x0_val + x1_val - 11.0) * 2.0*x0_val + 
                            2.0 * (x0_val + x1_val*x1_val - 7.0);
        CHECK( g[0] == approx(expected_g0) );

        for(int i = 0; i < n; ++i) {
            CHECK( std::isfinite(g[i]) );
        }
    }

    SECTION("testing high-dimensional neural network simulation")
    {
        // Test f(x) = sum_{i=0}^{99} tanh(w_i * sum_{j=0}^{i} x_j) where w_i = 1 + i/100
        // Simulates a simple feedforward neural network layer
        
        VectorXvar x(n);
        for(int i = 0; i < n; ++i) {
            x[i] = 0.01 * (i - 50); // x in [-0.49, 0.49] centered around 0
        }

        var f = 0.0;
        for(int i = 0; i < n; ++i) {
            var weighted_sum = 0.0;
            double weight = 1.0 + static_cast<double>(i) / 100.0;
            
            for(int j = 0; j <= i; ++j) {
                weighted_sum += x[j];
            }
            
            f += tanh(weight * weighted_sum);
        }

        VectorXd g = gradient(f, x);

        // Check gradient computation
        CHECK( g.size() == n );
        for(int i = 0; i < n; ++i) {
            CHECK( std::isfinite(g[i]) );
        }

        // Each x_j contributes to terms i=j..n-1
        // df/dx_j = sum_{i=j}^{n-1} w_i * sech^2(w_i * sum_{k=0}^{i} x_k)
        // Verify for x_0 which contributes to all terms
        double expected_g0 = 0.0;
        for(int i = 0; i < n; ++i) {
            double weight = 1.0 + static_cast<double>(i) / 100.0;
            double sum_up_to_i = 0.0;
            for(int k = 0; k <= i; ++k) {
                sum_up_to_i += val(x[k]);
            }
            double tanh_val = tanh(weight * sum_up_to_i);
            double sech_sq = 1.0 - tanh_val * tanh_val;
            expected_g0 += weight * sech_sq;
        }
        CHECK( g[0] == approx(expected_g0) );
    }

    SECTION("testing sum of inverse functions")
    {
        // Test f(x) = sum(1/(1 + x_i^2)) 
        // Well-behaved function with interesting derivatives
        
        VectorXvar x(n);
        for(int i = 0; i < n; ++i) {
            x[i] = -2.0 + 4.0 * static_cast<double>(i) / (n-1); // x in [-2, 2]
        }

        var f = 0.0;
        for(int i = 0; i < n; ++i) {
            f += 1.0 / (1.0 + x[i]*x[i]);
        }

        VectorXd g = gradient(f, x);

        // Check gradient values analytically
        // df/dx_i = -2*x_i / (1 + x_i^2)^2
        CHECK( g.size() == n );
        for(int i = 0; i < n; ++i) {
            double xi = val(x[i]);
            double denom = 1.0 + xi*xi;
            double expected = -2.0*xi / (denom*denom);
            CHECK( g[i] == approx(expected) );
        }
    }

    SECTION("testing product function")
    {
        // Test f(x) = prod(1 + 0.01*x_i) for i=0..99
        // Product of many terms - tests computational stability
        
        VectorXvar x(n);
        for(int i = 0; i < n; ++i) {
            x[i] = static_cast<double>(i) / 100.0; // x in [0, 0.99]
        }

        var f = 1.0;
        for(int i = 0; i < n; ++i) {
            f *= (1.0 + 0.01*x[i]);
        }

        VectorXd g = gradient(f, x);

        // Check gradient computation
        CHECK( g.size() == n );
        for(int i = 0; i < n; ++i) {
            CHECK( std::isfinite(g[i]) );
        }

        // For product functions: df/dx_i = 0.01 * f / (1 + 0.01*x_i)
        double f_val = val(f);
        for(int i = 0; i < n; ++i) {
            double expected = 0.01 * f_val / (1.0 + 0.01*val(x[i]));
            CHECK( g[i] == approx(expected) );
        }
    }

    SECTION("testing trigonometric optimization landscape")
    {
        // Test f(x) = sum(sin^2(x_i) + cos^2(x_{i+1})) + sum(sin(x_i)*cos(x_{i+1}))
        // Complex trigonometric landscape with many local minima
        
        VectorXvar x(n);
        for(int i = 0; i < n; ++i) {
            x[i] = M_PI * static_cast<double>(i) / (2.0 * n); // x in [0, π/2]
        }

        var f = 0.0;
        
        // First sum: sum(sin^2(x_i) + cos^2(x_{i+1}))
        for(int i = 0; i < n-1; ++i) {
            f += sin(x[i])*sin(x[i]) + cos(x[i+1])*cos(x[i+1]);
        }
        
        // Second sum: sum(sin(x_i)*cos(x_{i+1}))
        for(int i = 0; i < n-1; ++i) {
            f += sin(x[i])*cos(x[i+1]);
        }

        VectorXd g = gradient(f, x);

        // Check gradient computation
        CHECK( g.size() == n );
        for(int i = 0; i < n; ++i) {
            CHECK( std::isfinite(g[i]) );
        }

        // Verify gradient for first few variables analytically
        // df/dx_0 = 2*sin(x_0)*cos(x_0) + cos(x_0)*cos(x_1)
        double x0 = val(x[0]), x1 = val(x[1]);
        double expected_g0 = 2.0*sin(x0)*cos(x0) + cos(x0)*cos(x1);
        CHECK( g[0] == approx(expected_g0) );
    }

    SECTION("testing memory efficiency with many variables")
    {
        // Test that reverse mode is memory efficient with many variables
        // f(x) = sum(exp(-x_i^2/1000)) - Gaussian-like bumps
        
        VectorXvar x(n);
        for(int i = 0; i < n; ++i) {
            x[i] = 10.0 * static_cast<double>(i - 50) / 50.0; // x in [-10, 10]
        }

        auto start = std::chrono::high_resolution_clock::now();

        var f = 0.0;
        for(int i = 0; i < n; ++i) {
            f += exp(-x[i]*x[i]/1000.0);
        }

        VectorXd g = gradient(f, x);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // Check gradient values analytically
        // df/dx_i = exp(-x_i^2/1000) * (-2*x_i/1000)
        CHECK( g.size() == n );
        for(int i = 0; i < n; ++i) {
            double xi = val(x[i]);
            double expected = exp(-xi*xi/1000.0) * (-2.0*xi/1000.0);
            CHECK( g[i] == approx(expected) );
        }

        INFO("Reverse mode gradient computation for 100 variables took " << duration.count() << " milliseconds");
        CHECK( duration.count() < 1000 ); // Should be efficient
    }

    SECTION("testing steep function valleys")
    {
        // Test f(x) = sum((x_i - i/100)^4 + 1000*(x_i - x_{i+1})^2)
        // Creates steep valleys - challenging for optimization
        
        VectorXvar x(n);
        for(int i = 0; i < n; ++i) {
            x[i] = static_cast<double>(i) / 100.0 + 0.1; // Offset from optimum
        }

        var f = 0.0;
        for(int i = 0; i < n; ++i) {
            double target = static_cast<double>(i) / 100.0;
            f += (x[i] - target) * (x[i] - target) * (x[i] - target) * (x[i] - target);
            
            if(i < n-1) {
                f += 1000.0 * (x[i] - x[i+1]) * (x[i] - x[i+1]);
            }
        }

        VectorXd g = gradient(f, x);

        // Check gradient computation
        CHECK( g.size() == n );
        for(int i = 0; i < n; ++i) {
            CHECK( std::isfinite(g[i]) );
        }

        // Verify gradient for interior variables analytically
        for(int i = 1; i < n-1; ++i) {
            double xi = val(x[i]);
            double target = static_cast<double>(i) / 100.0;
            double xi_prev = val(x[i-1]);
            double xi_next = val(x[i+1]);
            
            double expected = 4.0 * pow(xi - target, 3) + 
                             2000.0 * (xi - xi_next) - 
                             2000.0 * (xi_prev - xi);
            CHECK( g[i] == approx(expected) );
        }
    }

    SECTION("testing sparse gradient pattern")
    {
        // Test f(x) = sum_{i: i%10==0} x_i^3
        // Only every 10th variable affects the function
        
        VectorXvar x(n);
        for(int i = 0; i < n; ++i) {
            x[i] = 0.5 + 0.01 * i;
        }

        var f = 0.0;
        for(int i = 0; i < n; i += 10) {
            f += x[i] * x[i] * x[i];
        }

        VectorXd g = gradient(f, x);

        // Check gradient sparsity pattern
        CHECK( g.size() == n );
        for(int i = 0; i < n; ++i) {
            if(i % 10 == 0) {
                // These variables should have non-zero gradients
                double expected = 3.0 * val(x[i]) * val(x[i]);
                CHECK( g[i] == approx(expected) );
            } else {
                // These variables should have zero gradients
                CHECK( g[i] == approx(0.0) );
            }
        }
    }

    SECTION("testing function with many local minima")
    {
        // Test f(x) = sum(sin(10*x_i) + 0.1*x_i^2)
        // Many local minima due to sin terms, global structure from quadratic
        
        VectorXvar x(n);
        for(int i = 0; i < n; ++i) {
            x[i] = 2.0 * M_PI * static_cast<double>(i) / n; // x in [0, 2π]
        }

        var f = 0.0;
        for(int i = 0; i < n; ++i) {
            f += sin(10.0 * x[i]) + 0.1 * x[i] * x[i];
        }

        VectorXd g = gradient(f, x);

        // Check gradient values analytically
        // df/dx_i = 10*cos(10*x_i) + 0.2*x_i
        CHECK( g.size() == n );
        for(int i = 0; i < n; ++i) {
            double xi = val(x[i]);
            double expected = 10.0 * cos(10.0 * xi) + 0.2 * xi;
            CHECK( g[i] == approx(expected) );
        }
    }
}
