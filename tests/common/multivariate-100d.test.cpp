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
using Eigen::VectorXd;

TEST_CASE("testing first-order derivatives on functions with 100 variables", "[multivariate][100d][performance]")
{
    const int n = 100;

    SECTION("testing quadratic function f(x) = sum(x_i^2) - forward mode")
    {
        // Test the simple quadratic function f(x) = x_1^2 + x_2^2 + ... + x_100^2
        // Expected gradient: df/dx_i = 2*x_i
        
        VectorXreal x(n);
        for(int i = 0; i < n; ++i) {
            x[i] = 0.01 * (i + 1); // x = [0.01, 0.02, 0.03, ..., 1.00]
        }

        auto f = [](const VectorXreal& x) -> real {
            real result = 0.0;
            for(int i = 0; i < x.size(); ++i) {
                result += x[i] * x[i];
            }
            return result;
        };

        VectorXd g = gradient(f, autodiff::detail::wrt(x), autodiff::detail::at(x));

        // Check gradient values
        CHECK( g.size() == n );
        for(int i = 0; i < n; ++i) {
            double expected = 2.0 * val(x[i]);
            CHECK( g[i] == approx(expected) );
        }

        // Check function value
        real f_val = f(x);
        double expected_f = 0.0;
        for(int i = 0; i < n; ++i) {
            expected_f += val(x[i]) * val(x[i]);
        }
        CHECK( val(f_val) == approx(expected_f) );
    }

    SECTION("testing quadratic function f(x) = sum(x_i^2) - reverse mode")
    {
        // Same test using reverse mode for comparison
        VectorXvar x(n);
        for(int i = 0; i < n; ++i) {
            x[i] = 0.01 * (i + 1);
        }

        var f = 0.0;
        for(int i = 0; i < n; ++i) {
            f += x[i] * x[i];
        }

        VectorXd g = gradient(f, x);

        // Check gradient values
        CHECK( g.size() == n );
        for(int i = 0; i < n; ++i) {
            double expected = 2.0 * val(x[i]);
            CHECK( g[i] == approx(expected) );
        }
    }

    SECTION("testing polynomial function f(x) = sum(i * x_i^3) - forward mode")
    {
        // Test f(x) = 1*x_1^3 + 2*x_2^3 + ... + 100*x_100^3
        // Expected gradient: df/dx_i = 3*i*x_i^2
        
        VectorXreal x(n);
        for(int i = 0; i < n; ++i) {
            x[i] = 0.1 + 0.005 * i; // x in [0.1, 0.595]
        }

        auto f = [](const VectorXreal& x) -> real {
            real result = 0.0;
            for(int i = 0; i < x.size(); ++i) {
                real coeff = static_cast<double>(i + 1);
                result += coeff * x[i] * x[i] * x[i];
            }
            return result;
        };

        VectorXd g = gradient(f, autodiff::detail::wrt(x), autodiff::detail::at(x));

        // Check gradient values
        CHECK( g.size() == n );
        for(int i = 0; i < n; ++i) {
            double coeff = static_cast<double>(i + 1);
            double expected = 3.0 * coeff * val(x[i]) * val(x[i]);
            CHECK( g[i] == approx(expected) );
        }
    }

    SECTION("testing trigonometric function f(x) = sum(sin(x_i)) - reverse mode")
    {
        // Test f(x) = sin(x_1) + sin(x_2) + ... + sin(x_100)
        // Expected gradient: df/dx_i = cos(x_i)
        
        VectorXvar x(n);
        for(int i = 0; i < n; ++i) {
            x[i] = M_PI * (i + 1) / (2.0 * n); // x in [π/200, π/2]
        }

        var f = 0.0;
        for(int i = 0; i < n; ++i) {
            f += sin(x[i]);
        }

        VectorXd g = gradient(f, x);

        // Check gradient values
        CHECK( g.size() == n );
        for(int i = 0; i < n; ++i) {
            double expected = cos(val(x[i]));
            CHECK( g[i] == approx(expected) );
        }
    }

    SECTION("testing exponential function f(x) = sum(exp(0.1*x_i)) - forward mode")
    {
        // Test f(x) = exp(0.1*x_1) + exp(0.1*x_2) + ... + exp(0.1*x_100)
        // Expected gradient: df/dx_i = 0.1*exp(0.1*x_i)
        
        VectorXreal x(n);
        for(int i = 0; i < n; ++i) {
            x[i] = static_cast<double>(i) / 10.0; // x in [0, 9.9]
        }

        auto f = [](const VectorXreal& x) -> real {
            real result = 0.0;
            for(int i = 0; i < x.size(); ++i) {
                result += exp(0.1 * x[i]);
            }
            return result;
        };

        VectorXd g = gradient(f, autodiff::detail::wrt(x), autodiff::detail::at(x));

        // Check gradient values
        CHECK( g.size() == n );
        for(int i = 0; i < n; ++i) {
            double expected = 0.1 * exp(0.1 * val(x[i]));
            CHECK( g[i] == approx(expected) );
        }
    }

    SECTION("testing coupled function f(x) = sum(x_i * x_{i+1}) - reverse mode")
    {
        // Test f(x) = x_1*x_2 + x_2*x_3 + ... + x_99*x_100
        // Expected gradient: df/dx_1 = x_2, df/dx_i = x_{i-1} + x_{i+1} for i=2..99, df/dx_100 = x_99
        
        VectorXvar x(n);
        for(int i = 0; i < n; ++i) {
            x[i] = 1.0 + 0.01 * i; // x in [1.0, 1.99]
        }

        var f = 0.0;
        for(int i = 0; i < n-1; ++i) {
            f += x[i] * x[i+1];
        }

        VectorXd g = gradient(f, x);

        // Check gradient values
        CHECK( g.size() == n );
        
        // First variable: df/dx_1 = x_2
        CHECK( g[0] == approx(val(x[1])) );
        
        // Middle variables: df/dx_i = x_{i-1} + x_{i+1}
        for(int i = 1; i < n-1; ++i) {
            double expected = val(x[i-1]) + val(x[i+1]);
            CHECK( g[i] == approx(expected) );
        }
        
        // Last variable: df/dx_100 = x_99
        CHECK( g[n-1] == approx(val(x[n-2])) );
    }

    SECTION("testing Rosenbrock-style function - forward mode")
    {
        // Test f(x) = sum((1-x_i)^2 + 100*(x_{i+1}-x_i^2)^2) for i=1..99
        // This is the extended Rosenbrock function
        
        VectorXreal x(n);
        for(int i = 0; i < n; ++i) {
            x[i] = 0.5 + 0.01 * i; // Starting point near [0.5, 1.49]
        }

        auto f = [](const VectorXreal& x) -> real {
            real result = 0.0;
            for(int i = 0; i < x.size()-1; ++i) {
                real term1 = (1.0 - x[i]) * (1.0 - x[i]);
                real term2 = 100.0 * (x[i+1] - x[i]*x[i]) * (x[i+1] - x[i]*x[i]);
                result += term1 + term2;
            }
            return result;
        };

        VectorXd g = gradient(f, autodiff::detail::wrt(x), autodiff::detail::at(x));

        // Check that gradient computation succeeds and produces finite values
        CHECK( g.size() == n );
        for(int i = 0; i < n; ++i) {
            CHECK( std::isfinite(g[i]) );
        }

        // Check specific gradient values for first few variables analytically
        // df/dx_1 = -2*(1-x_1) - 400*x_1*(x_2 - x_1^2)
        double x1 = val(x[0]), x2 = val(x[1]);
        double expected_g1 = -2.0*(1.0-x1) - 400.0*x1*(x2 - x1*x1);
        CHECK( g[0] == approx(expected_g1) );

        // df/dx_2 = -2*(1-x_2) + 200*(x_2 - x_1^2) - 400*x_2*(x_3 - x_2^2)
        double x3 = val(x[2]);
        double expected_g2 = -2.0*(1.0-x2) + 200.0*(x2 - x1*x1) - 400.0*x2*(x3 - x2*x2);
        CHECK( g[1] == approx(expected_g2) );
    }

    SECTION("testing neural network-style activation function - reverse mode")
    {
        // Test f(x) = sum(tanh(sum_{j=1}^{i} x_j)) for i=1..100
        // This simulates a simple feedforward computation
        
        VectorXvar x(n);
        for(int i = 0; i < n; ++i) {
            x[i] = 0.1 * sin(static_cast<double>(i)); // Small random-like values
        }

        var f = 0.0;
        for(int i = 0; i < n; ++i) {
            var sum_up_to_i = 0.0;
            for(int j = 0; j <= i; ++j) {
                sum_up_to_i += x[j];
            }
            f += tanh(sum_up_to_i);
        }

        VectorXd g = gradient(f, x);

        // Check that gradient computation succeeds
        CHECK( g.size() == n );
        for(int i = 0; i < n; ++i) {
            CHECK( std::isfinite(g[i]) );
            // The gradient should be non-zero for most components
            // since each x_j affects multiple terms in the sum
        }

        // Each x_j contributes to terms i=j..n-1, so:
        // df/dx_j = sum_{i=j}^{n-1} sech^2(sum_{k=0}^{i} x_k)
        // We can verify this for the first variable at least
        double expected_g0 = 0.0;
        for(int i = 0; i < n; ++i) {
            double sum_up_to_i = 0.0;
            for(int k = 0; k <= i; ++k) {
                sum_up_to_i += val(x[k]);
            }
            double sech_sq = 1.0 - tanh(sum_up_to_i) * tanh(sum_up_to_i);
            expected_g0 += sech_sq;
        }
        CHECK( g[0] == approx(expected_g0) );
    }

    SECTION("testing performance benchmark")
    {
        // Simple performance test to ensure 100-variable functions are handled efficiently
        VectorXvar x(n);
        for(int i = 0; i < n; ++i) {
            x[i] = static_cast<double>(i) / 100.0;
        }

        auto start = std::chrono::high_resolution_clock::now();

        // Simple function that should be fast to differentiate
        var f = 0.0;
        for(int i = 0; i < n; ++i) {
            f += x[i] * x[i] * x[i] + sin(x[i]);
        }

        VectorXd g = gradient(f, x);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // Check that computation completed
        CHECK( g.size() == n );
        for(int i = 0; i < n; ++i) {
            CHECK( std::isfinite(g[i]) );
        }

        // Performance should be reasonable (less than 1 second)
        INFO("Gradient computation for 100 variables took " << duration.count() << " milliseconds");
        CHECK( duration.count() < 1000 );
    }

    SECTION("testing mixed trigonometric and polynomial function")
    {
        // Test f(x) = sum(x_i^2 * sin(sum_{j=1}^{i} x_j)) - complex coupling
        
        VectorXreal x(n);
        for(int i = 0; i < n; ++i) {
            x[i] = 0.05 + 0.01 * i; // x in [0.05, 1.04]
        }

        auto f = [](const VectorXreal& x) -> real {
            real result = 0.0;
            for(int i = 0; i < x.size(); ++i) {
                real sum_up_to_i = 0.0;
                for(int j = 0; j <= i; ++j) {
                    sum_up_to_i += x[j];
                }
                result += x[i] * x[i] * sin(sum_up_to_i);
            }
            return result;
        };

        VectorXd g = gradient(f, autodiff::detail::wrt(x), autodiff::detail::at(x));

        // Check that gradient computation succeeds
        CHECK( g.size() == n );
        for(int i = 0; i < n; ++i) {
            CHECK( std::isfinite(g[i]) );
        }

        // Verify a few gradient components analytically
        // df/dx_0 = 2*x_0*sin(x_0) + x_0^2*cos(x_0) + sum_{i=1}^{n-1} x_i^2*cos(sum_{j=0}^{i} x_j)
        double sum_0 = val(x[0]);
        double expected_g0 = 2.0*val(x[0])*sin(sum_0) + val(x[0])*val(x[0])*cos(sum_0);
        for(int i = 1; i < n; ++i) {
            double sum_up_to_i = 0.0;
            for(int j = 0; j <= i; ++j) {
                sum_up_to_i += val(x[j]);
            }
            expected_g0 += val(x[i])*val(x[i])*cos(sum_up_to_i);
        }
        CHECK( g[0] == approx(expected_g0) );
    }

    SECTION("testing sparse-like function (only some variables active)")
    {
        // Test f(x) = sum_{i even} x_i^3 + sum_{i odd} exp(x_i)
        // Half the variables appear in cubic terms, half in exponential terms
        
        VectorXvar x(n);
        for(int i = 0; i < n; ++i) {
            x[i] = 0.1 + 0.02 * i; // x in [0.1, 2.08]
        }

        var f = 0.0;
        for(int i = 0; i < n; ++i) {
            if(i % 2 == 0) {
                // Even indices: cubic terms
                f += x[i] * x[i] * x[i];
            } else {
                // Odd indices: exponential terms (scaled to avoid overflow)
                f += exp(0.1 * x[i]);
            }
        }

        VectorXd g = gradient(f, x);

        // Check gradient values
        CHECK( g.size() == n );
        for(int i = 0; i < n; ++i) {
            if(i % 2 == 0) {
                // Even indices: df/dx_i = 3*x_i^2
                double expected = 3.0 * val(x[i]) * val(x[i]);
                CHECK( g[i] == approx(expected) );
            } else {
                // Odd indices: df/dx_i = 0.1*exp(0.1*x_i)
                double expected = 0.1 * exp(0.1 * val(x[i]));
                CHECK( g[i] == approx(expected) );
            }
        }
    }
}
