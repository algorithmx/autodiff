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
#include <tests/utils/catch.hpp>
using namespace autodiff;
using Eigen::VectorXd;

TEST_CASE("testing forward mode derivatives on 100-variable functions", "[forward][utils][multivariate][100d]")
{
    const int n = 100;

    SECTION("testing individual derivative computation with derivatives() function")
    {
        // Test computing individual partial derivatives using the derivatives() function
        // f(x) = sum(i * x_i^2) where i goes from 1 to 100
        
        auto f = [](const VectorXreal& x) -> real {
            real result = 0.0;
            for(int i = 0; i < x.size(); ++i) {
                real coeff = static_cast<double>(i + 1);
                result += coeff * x[i] * x[i];
            }
            return result;
        };

        VectorXreal x(n);
        for(int i = 0; i < n; ++i) {
            x[i] = 0.01 * (i + 1); // x = [0.01, 0.02, ..., 1.00]
        }

        // Test computing individual partial derivatives
        for(int i = 0; i < 10; ++i) { // Test first 10 for efficiency
            auto df_dxi = derivative(f, wrt(x[i]), at(x));
            double expected = 2.0 * static_cast<double>(i + 1) * val(x[i]);
            CHECK( df_dxi == approx(expected) );
        }

        // Test computing full gradient using gradient function
        VectorXd g = gradient(f, wrt(x), at(x));
        CHECK( g.size() == n );
        for(int i = 0; i < n; ++i) {
            double expected = 2.0 * static_cast<double>(i + 1) * val(x[i]);
            CHECK( g[i] == approx(expected) );
        }
    }

    SECTION("testing multi-variable function with cross terms")
    {
        // Test f(x) = sum_{i<j} x_i * x_j (all pairwise products)
        // This creates a function with many cross terms
        
        // For efficiency, test with smaller subset for this complex function
        const int m = 10;
        VectorXreal x(m);
        for(int i = 0; i < m; ++i) {
            x[i] = 0.1 + 0.05 * i; // x = [0.1, 0.15, 0.2, ..., 0.55]
        }

        auto f = [](const VectorXreal& x) -> real {
            real result = 0.0;
            for(int i = 0; i < x.size(); ++i) {
                for(int j = i + 1; j < x.size(); ++j) {
                    result += x[i] * x[j];
                }
            }
            return result;
        };

        VectorXd g = gradient(f, wrt(x), at(x));
        
        // Expected gradient: df/dx_i = sum_{j!=i} x_j
        CHECK( g.size() == m );
        for(int i = 0; i < m; ++i) {
            double expected = 0.0;
            for(int j = 0; j < m; ++j) {
                if(i != j) {
                    expected += val(x[j]);
                }
            }
            CHECK( g[i] == approx(expected) );
        }
    }

    SECTION("testing logarithmic and exponential functions")
    {
        // Test f(x) = sum(log(1 + x_i^2) + exp(-x_i^2/10))
        // Combines logarithmic and exponential terms
        
        VectorXreal x(n);
        for(int i = 0; i < n; ++i) {
            x[i] = 0.5 + 0.01 * i; // x in [0.5, 1.49] to keep log arguments positive
        }

        auto f = [](const VectorXreal& x) -> real {
            real result = 0.0;
            for(int i = 0; i < x.size(); ++i) {
                result += log(1.0 + x[i]*x[i]) + exp(-x[i]*x[i]/10.0);
            }
            return result;
        };

        VectorXd g = gradient(f, wrt(x), at(x));

        // Check gradient values analytically
        // df/dx_i = 2*x_i/(1 + x_i^2) - (x_i/5)*exp(-x_i^2/10)
        CHECK( g.size() == n );
        for(int i = 0; i < n; ++i) {
            double xi = val(x[i]);
            double expected = 2.0*xi/(1.0 + xi*xi) - (xi/5.0)*exp(-xi*xi/10.0);
            CHECK( g[i] == approx(expected) );
        }
    }

    SECTION("testing trigonometric functions with phase shifts")
    {
        // Test f(x) = sum(sin(x_i + i*π/100) + cos(x_i - i*π/100))
        // Each variable has a different phase shift
        
        VectorXreal x(n);
        for(int i = 0; i < n; ++i) {
            x[i] = M_PI/4 + 0.01 * i; // x around π/4
        }

        auto f = [](const VectorXreal& x) -> real {
            real result = 0.0;
            for(int i = 0; i < x.size(); ++i) {
                double phase = static_cast<double>(i) * M_PI / 100.0;
                result += sin(x[i] + phase) + cos(x[i] - phase);
            }
            return result;
        };

        VectorXd g = gradient(f, wrt(x), at(x));

        // Check gradient values analytically
        // df/dx_i = cos(x_i + i*π/100) - sin(x_i - i*π/100)
        CHECK( g.size() == n );
        for(int i = 0; i < n; ++i) {
            double xi = val(x[i]);
            double phase = static_cast<double>(i) * M_PI / 100.0;
            double expected = cos(xi + phase) - sin(xi - phase);
            CHECK( g[i] == approx(expected) );
        }
    }

    SECTION("testing power functions with varying exponents")
    {
        // Test f(x) = sum(x_i^(1 + i/100)) for i=0..99
        // Each variable has a different exponent
        
        VectorXreal x(n);
        for(int i = 0; i < n; ++i) {
            x[i] = 1.1 + 0.001 * i; // x in [1.1, 1.199] to keep powers well-defined
        }

        auto f = [](const VectorXreal& x) -> real {
            real result = 0.0;
            for(int i = 0; i < x.size(); ++i) {
                double exponent = 1.0 + static_cast<double>(i) / 100.0;
                result += pow(x[i], exponent);
            }
            return result;
        };

        VectorXd g = gradient(f, wrt(x), at(x));

        // Check gradient values analytically
        // df/dx_i = (1 + i/100) * x_i^(i/100)
        CHECK( g.size() == n );
        for(int i = 0; i < n; ++i) {
            double xi = val(x[i]);
            double exponent = 1.0 + static_cast<double>(i) / 100.0;
            double expected = exponent * pow(xi, static_cast<double>(i) / 100.0);
            CHECK( g[i] == approx(expected) );
        }
    }

    SECTION("testing rational functions")
    {
        // Test f(x) = sum(x_i / (1 + x_i^2))
        // Rational function that's well-behaved
        
        VectorXreal x(n);
        for(int i = 0; i < n; ++i) {
            x[i] = -5.0 + 10.0 * static_cast<double>(i) / (n-1); // x in [-5, 5]
        }

        auto f = [](const VectorXreal& x) -> real {
            real result = 0.0;
            for(int i = 0; i < x.size(); ++i) {
                result += x[i] / (1.0 + x[i]*x[i]);
            }
            return result;
        };

        VectorXd g = gradient(f, wrt(x), at(x));

        // Check gradient values analytically
        // df/dx_i = (1 - x_i^2) / (1 + x_i^2)^2
        CHECK( g.size() == n );
        for(int i = 0; i < n; ++i) {
            double xi = val(x[i]);
            double denom = 1.0 + xi*xi;
            double expected = (1.0 - xi*xi) / (denom*denom);
            CHECK( g[i] == approx(expected) );
        }
    }

    SECTION("testing chain rule with nested functions")
    {
        // Test f(x) = sum(exp(sin(x_i^2)))
        // Multiple levels of composition
        
        VectorXreal x(n);
        for(int i = 0; i < n; ++i) {
            x[i] = 0.5 + 0.5 * static_cast<double>(i) / (n-1); // x in [0.5, 1.0]
        }

        auto f = [](const VectorXreal& x) -> real {
            real result = 0.0;
            for(int i = 0; i < x.size(); ++i) {
                result += exp(sin(x[i]*x[i]));
            }
            return result;
        };

        VectorXd g = gradient(f, wrt(x), at(x));

        // Check gradient values analytically
        // df/dx_i = exp(sin(x_i^2)) * cos(x_i^2) * 2*x_i
        CHECK( g.size() == n );
        for(int i = 0; i < n; ++i) {
            double xi = val(x[i]);
            double xi_sq = xi * xi;
            double expected = exp(sin(xi_sq)) * cos(xi_sq) * 2.0 * xi;
            CHECK( g[i] == approx(expected) );
        }
    }

    SECTION("testing piecewise smooth function with abs()")
    {
        // Test f(x) = sum(abs(x_i - 0.5))
        // Piecewise linear function
        
        VectorXreal x(n);
        for(int i = 0; i < n; ++i) {
            x[i] = static_cast<double>(i) / (n-1); // x in [0, 1]
        }

        auto f = [](const VectorXreal& x) -> real {
            real result = 0.0;
            for(int i = 0; i < x.size(); ++i) {
                result += abs(x[i] - 0.5);
            }
            return result;
        };

        VectorXd g = gradient(f, wrt(x), at(x));

        // Check gradient values analytically
        // df/dx_i = sign(x_i - 0.5)
        CHECK( g.size() == n );
        for(int i = 0; i < n; ++i) {
            double xi = val(x[i]);
            double expected = (xi > 0.5) ? 1.0 : ((xi < 0.5) ? -1.0 : 0.0);
            if(xi != 0.5) { // Skip the non-differentiable point
                CHECK( g[i] == approx(expected) );
            }
        }
    }

    SECTION("testing performance with complex expression trees")
    {
        // Test f(x) = sum(sin(exp(x_i)) + cos(log(1+x_i^2)) + tanh(x_i^3))
        // Complex expression with deep computational graphs
        
        VectorXreal x(n);
        for(int i = 0; i < n; ++i) {
            x[i] = 0.1 + 0.005 * i; // x in [0.1, 0.595] for stability
        }

        auto start = std::chrono::high_resolution_clock::now();

        auto f = [](const VectorXreal& x) -> real {
            real result = 0.0;
            for(int i = 0; i < x.size(); ++i) {
                result += sin(exp(x[i])) + cos(log(1.0 + x[i]*x[i])) + tanh(x[i]*x[i]*x[i]);
            }
            return result;
        };

        VectorXd g = gradient(f, wrt(x), at(x));

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // Check that computation completed successfully
        CHECK( g.size() == n );
        for(int i = 0; i < n; ++i) {
            CHECK( std::isfinite(g[i]) );
        }

        INFO("Complex gradient computation for 100 variables took " << duration.count() << " milliseconds");
        CHECK( duration.count() < 2000 ); // Should complete within 2 seconds
    }
}
