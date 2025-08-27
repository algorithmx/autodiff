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

#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <vector>

// Catch includes
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

// autodiff includes
#include <autodiff/forward/dual.hpp>
#include <autodiff/reverse/unified_expr.hpp>
#include <autodiff/reverse/var.hpp>
#include <tests/utils/catch.hpp>
using namespace autodiff;

// this is large enough to cause noticable delay in test, indicating potential performance issues
#define NN 25

// Helper function for reverse mode derivatives
inline auto grad(const var& y, var& x)
{
    auto g = derivatives(y, autodiff::reverse::detail::wrt(x));
    return val(g[0]);
}

TEST_CASE("testing first-order derivatives on composite function chains", "[composite][chain][first-order]")
{
    SECTION("testing chain of 100 composite linear functions with forward mode")
    {
        std::cout << "[chain of linear (forward)]" << std::flush;

        // We'll test f1(f2(f3(...f100(x)...))) where each fi(x) = a_i * x + b_i
        // This creates a linear chain where the final derivative can be computed analytically

        const int n = NN;
        std::vector<double> a_coeffs(n), b_coeffs(n);

        // Initialize coefficients - use simple values to make analytical computation easier
        for(int i = 0; i < n; ++i) {
            a_coeffs[i] = 1.0 + (1.0 / n) * i; // coefficients from 1.0 to 1.99
            b_coeffs[i] = 0.1 * i;             // constants from 0 to 9.9
        }

        // Test point
        const double x_val = 2.0;

        // Define the composite function as a lambda
        auto composite_func = [&](dual x) -> dual {
            dual result = x;
            for(int i = 0; i < n; ++i) {
                result = a_coeffs[i] * result + b_coeffs[i];
            }
            return result;
        };

        // Get autodiff derivative using the function approach
        dual x = x_val;
        auto derivatives_result = derivatives(composite_func, autodiff::detail::wrt(x), autodiff::detail::at(x));
        double u = derivatives_result[0];
        double ux = derivatives_result[1];

        // Analytical derivative: d/dx[f1(f2(...fn(x)...))] = f1'(f2(...)) * f2'(f3(...)) * ... * fn'(x)
        // For linear functions f_i(x) = a_i * x + b_i, we have f_i'(x) = a_i
        // So the chain rule gives us: product of all a_i coefficients
        double analytical_derivative = 1.0;
        for(int i = 0; i < n; ++i) {
            analytical_derivative *= a_coeffs[i];
        }

        CHECK(ux == approx(analytical_derivative).epsilon(1e-12));

        // Also verify the function value analytically
        double analytical_value = x_val;
        for(int i = 0; i < n; ++i) {
            analytical_value = a_coeffs[i] * analytical_value + b_coeffs[i];
        }
        CHECK(u == approx(analytical_value).epsilon(1e-12));

        std::cout << "[END]" << std::endl;
    }

    SECTION("testing chain of 100 composite linear functions with reverse mode")
    {
        std::cout << "[chain of linear]" << std::flush;

        // Same test but using reverse mode autodiff

        const int n = NN;
        std::vector<double> a_coeffs(n), b_coeffs(n);

        for(int i = 0; i < n; ++i) {
            a_coeffs[i] = 1.0 + (1.0 / n) * i;
            b_coeffs[i] = 0.1 * i;
        }

        const double x_val = 2.0;
        var x = x_val;

        // Time expression build
        auto t_build_start = std::chrono::high_resolution_clock::now();
        var result = x;
        for(int i = 0; i < n; ++i) {
            result = a_coeffs[i] * result + b_coeffs[i];
        }
        auto t_build_end = std::chrono::high_resolution_clock::now();

        double build_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_build_end - t_build_start).count();

        double analytical_derivative = 1.0;
        for(int i = 0; i < n; ++i) {
            analytical_derivative *= a_coeffs[i];
        }

        auto t_deriv_start = std::chrono::high_resolution_clock::now();
        auto ux = grad(result, x);
        auto t_deriv_end = std::chrono::high_resolution_clock::now();

        double deriv_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_deriv_end - t_deriv_start).count();

        std::cout << " [build=" << build_ms << "ms, deriv=" << deriv_ms << "ms]" << std::flush;

        CHECK(ux == approx(analytical_derivative).epsilon(1e-12));

        double analytical_value = x_val;
        for(int i = 0; i < n; ++i) {
            analytical_value = a_coeffs[i] * analytical_value + b_coeffs[i];
        }
        CHECK(val(result) == approx(analytical_value).epsilon(1e-12));
        std::cout << "[END]" << std::endl;
    }

    SECTION("testing chain of 50 exponential composite functions with reverse mode")
    {
        std::cout << "[chain of exponential]" << std::flush;

        // Test a more complex chain: f_i(x) = exp(0.01 * x) for i = 1..50
        // The chain becomes: exp(0.01 * exp(0.01 * exp(0.01 * ... * exp(0.01 * x))))

        const int n = NN; // Reduced to 50 to avoid numerical overflow
        const double scale = 0.01;
        const double x_val = 1.0;

        var x = x_val;
        var result = x;

        auto t_build_start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < n; ++i) {
            result = exp(scale * result);
        }
        auto t_build_end = std::chrono::high_resolution_clock::now();
        double build_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_build_end - t_build_start).count();

        // Calculate analytical derivative using chain rule
        // d/dx[exp(0.01 * exp(0.01 * ... * exp(0.01 * x)))]
        // This requires computing the intermediate values to get the chain rule terms

        std::vector<double> intermediate_values(n + 1);
        intermediate_values[0] = x_val;

        for(int i = 0; i < n; ++i) {
            intermediate_values[i + 1] = std::exp(scale * intermediate_values[i]);
        }

        // Chain rule: derivative is product of all exp(scale * f_i) * scale terms
        double analytical_derivative = 1.0;
        for(int i = 0; i < n; ++i) {
            analytical_derivative *= scale * std::exp(scale * intermediate_values[i]);
        }

        auto t_deriv_start = std::chrono::high_resolution_clock::now();
        auto ux = grad(result, x);
        auto t_deriv_end = std::chrono::high_resolution_clock::now();
        double deriv_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_deriv_end - t_deriv_start).count();

        std::cout << " [build=" << build_ms << "ms, deriv=" << deriv_ms << "ms]" << std::flush;

        CHECK(ux == approx(analytical_derivative).epsilon(1e-12));
        CHECK(val(result) == approx(intermediate_values[n]).epsilon(1e-12));
        std::cout << "[END]" << std::endl;
    }

    SECTION("testing chain of 100 polynomial composite functions with reverse mode")
    {
        std::cout << "[chain of polynomial]" << std::flush;

        // Test f_i(x) = x^2 + 0.001 for i = 1..100
        // This creates: (((x^2 + 0.001)^2 + 0.001)^2 + 0.001)^2...

        const int n = NN;
        const double c = 0.001; // Small constant to avoid explosive growth
        const double x_val = 1.1;

        var x = x_val;
        var result = x;

        auto t_build_start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < n; ++i) {
            result = result * result + c;
        }
        auto t_build_end = std::chrono::high_resolution_clock::now();
        double build_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_build_end - t_build_start).count();

        // Calculate intermediate values for analytical derivative
        std::vector<double> intermediate_values(n + 1);
        intermediate_values[0] = x_val;

        for(int i = 0; i < n; ++i) {
            intermediate_values[i + 1] = intermediate_values[i] * intermediate_values[i] + c;
        }

        // Chain rule: d/dx[f_n(...f_1(x)...)] = f_n'(...) * ... * f_1'(x)
        // where f_i'(x) = 2x for f_i(x) = x^2 + c
        double analytical_derivative = 1.0;
        for(int i = 0; i < n; ++i) {
            analytical_derivative *= 2.0 * intermediate_values[i];
        }

        auto t_deriv_start = std::chrono::high_resolution_clock::now();
        auto ux = grad(result, x);
        auto t_deriv_end = std::chrono::high_resolution_clock::now();
        double deriv_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_deriv_end - t_deriv_start).count();

        std::cout << " [build=" << build_ms << "ms, deriv=" << deriv_ms << "ms]" << std::flush;

        CHECK(ux == approx(analytical_derivative).epsilon(1e-12));
        CHECK(val(result) == approx(intermediate_values[n]).epsilon(1e-12));
        std::cout << "[END]" << std::endl;
    }

    SECTION("testing chain of 100 composite linear functions with unified reverse mode")
    {
        std::cout << "[chain of linear (unified)]" << std::flush;

        const int n = NN;
        std::vector<double> a_coeffs(n), b_coeffs(n);
        for(int i = 0; i < n; ++i) {
            a_coeffs[i] = 1.0 + (1.0 / n) * i;
            b_coeffs[i] = 0.1 * i;
        }

        const double x_val = 2.0;

        // Create an arena scope for unified variables
        {
            autodiff::reverse::unified::ArenaScope<double> scope;
            auto x = autodiff::reverse::unified::make_var<double>(x_val);

            // Pre-create unified constant versions of coefficients to avoid
            // allocating a temporary unified constant on every loop iteration.
            std::vector<decltype(x)> a_consts(n), b_consts(n);
            for(int i = 0; i < n; ++i) {
                a_consts[i] = autodiff::reverse::unified::make_const<double>(a_coeffs[i]);
                b_consts[i] = autodiff::reverse::unified::make_const<double>(b_coeffs[i]);
            }

            auto result = x;
            auto t_build_start = std::chrono::high_resolution_clock::now();
            for(int i = 0; i < n; ++i) {
                result = a_consts[i] * result + b_consts[i];
            }
            auto t_build_end = std::chrono::high_resolution_clock::now();
            double build_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_build_end - t_build_start).count();

            double analytical_derivative = 1.0;
            for(int i = 0; i < n; ++i)
                analytical_derivative *= a_coeffs[i];

            auto t_deriv_start = std::chrono::high_resolution_clock::now();
            auto grads = autodiff::reverse::unified::derivatives(result, autodiff::reverse::unified::wrt(x));
            auto t_deriv_end = std::chrono::high_resolution_clock::now();
            double deriv_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_deriv_end - t_deriv_start).count();
            double ux = grads[0];

            std::cout << " [build=" << build_ms << "ms, deriv=" << deriv_ms << "ms]" << std::flush;

            CHECK(ux == approx(analytical_derivative).epsilon(1e-12));

            double analytical_value = x_val;
            for(int i = 0; i < n; ++i)
                analytical_value = a_coeffs[i] * analytical_value + b_coeffs[i];
            CHECK(result.value() == approx(analytical_value).epsilon(1e-12));
        }

        std::cout << "[END]" << std::endl;
    }

    SECTION("testing chain of 50 exponential composite functions with unified reverse mode")
    {
        std::cout << "[chain of exponential (unified)]" << std::flush;

        const int n = NN;
        const double scale = 0.01;
        const double x_val = 1.0;

        {
            autodiff::reverse::unified::ArenaScope<double> scope;
            auto x = autodiff::reverse::unified::make_var<double>(x_val);
            auto result = x;

            // Pre-create unified constant for scale to avoid creating temporaries
            // during the loop (scale * result would otherwise create a temp).
            auto scale_c = autodiff::reverse::unified::make_const<double>(scale);

            auto t_build_start = std::chrono::high_resolution_clock::now();
            for(int i = 0; i < n; ++i)
                result = autodiff::reverse::unified::exp(scale_c * result);
            auto t_build_end = std::chrono::high_resolution_clock::now();
            double build_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_build_end - t_build_start).count();

            std::vector<double> intermediate_values(n + 1);
            intermediate_values[0] = x_val;
            for(int i = 0; i < n; ++i)
                intermediate_values[i + 1] = std::exp(scale * intermediate_values[i]);

            double analytical_derivative = 1.0;
            for(int i = 0; i < n; ++i)
                analytical_derivative *= scale * std::exp(scale * intermediate_values[i]);

            auto t_deriv_start = std::chrono::high_resolution_clock::now();
            auto grads = autodiff::reverse::unified::derivatives(result, autodiff::reverse::unified::wrt(x));
            auto t_deriv_end = std::chrono::high_resolution_clock::now();
            double deriv_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_deriv_end - t_deriv_start).count();
            double ux = grads[0];

            std::cout << " [build=" << build_ms << "ms, deriv=" << deriv_ms << "ms]" << std::flush;

            CHECK(ux == approx(analytical_derivative).epsilon(1e-12));
            CHECK(result.value() == approx(intermediate_values[n]).epsilon(1e-12));
        }

        std::cout << "[END]" << std::endl;
    }

    SECTION("testing chain of 100 polynomial composite functions with unified reverse mode")
    {
        std::cout << "[chain of polynomial (unified)]" << std::flush;

        const int n = NN;
        const double c = 0.001;
        const double x_val = 1.1;

        {
            autodiff::reverse::unified::ArenaScope<double> scope;
            auto x = autodiff::reverse::unified::make_var<double>(x_val);
            auto result = x;

            // Pre-create unified constant for c to avoid allocating one each loop
            auto c_const = autodiff::reverse::unified::make_const<double>(c);

            auto t_build_start = std::chrono::high_resolution_clock::now();
            for(int i = 0; i < n; ++i) {
                result = result * result + c_const; // TODO
            }
            auto t_build_end = std::chrono::high_resolution_clock::now();
            double build_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_build_end - t_build_start).count();

            std::vector<double> intermediate_values(n + 1);
            intermediate_values[0] = x_val;
            for(int i = 0; i < n; ++i)
                intermediate_values[i + 1] = intermediate_values[i] * intermediate_values[i] + c;

            double analytical_derivative = 1.0;
            for(int i = 0; i < n; ++i)
                analytical_derivative *= 2.0 * intermediate_values[i];

            auto t_deriv_start = std::chrono::high_resolution_clock::now();
            auto grads = autodiff::reverse::unified::derivatives(result, autodiff::reverse::unified::wrt(x));
            auto t_deriv_end = std::chrono::high_resolution_clock::now();
            double deriv_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_deriv_end - t_deriv_start).count();
            double ux = grads[0];

            std::cout << " [build=" << build_ms << "ms, deriv=" << deriv_ms << "ms]" << std::flush;

            CHECK(ux == approx(analytical_derivative).epsilon(1e-12));
            CHECK(result.value() == approx(intermediate_values[n]).epsilon(1e-12));
        }

        std::cout << "[END]" << std::endl;
    }

    SECTION("testing crazy mixed composite function chain with 50 functions (unified)")
    {
        std::cout << "[crazy mixed composite (unified)]" << std::flush;

        const int n = NN;
        const double x_val = 1.5;

        {
            autodiff::reverse::unified::ArenaScope<double> scope;
            auto x = autodiff::reverse::unified::make_var<double>(x_val);
            auto result = x;

            // Pre-create common unified constants to avoid repeated allocations
            auto one_const = autodiff::reverse::unified::make_const<double>(1.0);
            auto point1_const = autodiff::reverse::unified::make_const<double>(0.1);
            auto point01_const = autodiff::reverse::unified::make_const<double>(0.01);
            auto two_const = autodiff::reverse::unified::make_const<double>(2.0);
            auto half_const = autodiff::reverse::unified::make_const<double>(0.5);

            auto t_build_start = std::chrono::high_resolution_clock::now();
            for(int i = 0; i < n; ++i) {
                switch(i % 10) {
                case 0:
                    result = autodiff::reverse::unified::sin(point1_const * result);
                    break;
                case 1:
                    result = autodiff::reverse::unified::cos(point1_const * result);
                    break;
                case 2:
                    result = autodiff::reverse::unified::exp(point01_const * result);
                    break;
                case 3:
                    result = autodiff::reverse::unified::log(autodiff::reverse::unified::abs(result) + one_const);
                    break;
                case 4:
                    result = result * result + point1_const;
                    break;
                case 5:
                    result = two_const * result + half_const;
                    break;
                case 6:
                    result = autodiff::reverse::unified::sqrt(autodiff::reverse::unified::abs(result) + one_const);
                    break;
                case 7:
                    result = one_const / (autodiff::reverse::unified::abs(result) + one_const);
                    break;
                case 8:
                    result = autodiff::reverse::unified::tanh(point1_const * result);
                    break;
                case 9:
                    result = result * result * result + point01_const * result;
                    break;
                }
            }
            auto t_build_end = std::chrono::high_resolution_clock::now();
            double build_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_build_end - t_build_start).count();

            auto t_deriv_start = std::chrono::high_resolution_clock::now();
            auto grads = autodiff::reverse::unified::derivatives(result, autodiff::reverse::unified::wrt(x));
            auto t_deriv_end = std::chrono::high_resolution_clock::now();
            double deriv_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_deriv_end - t_deriv_start).count();
            double ux = grads[0];

            std::cout << " [build=" << build_ms << "ms, deriv=" << deriv_ms << "ms]" << std::flush;

            CHECK(std::isfinite(result.value()));
            CHECK(std::isfinite(ux));
            CHECK(std::abs(ux) < 1e10);

            // Forward-mode consistency check
            auto mixed_func = [](dual x) -> dual {
                dual result = x;
                for(int i = 0; i < n; ++i) {
                    switch(i % 10) {
                    case 0:
                        result = sin(0.1 * result);
                        break;
                    case 1:
                        result = cos(0.1 * result);
                        break;
                    case 2:
                        result = exp(0.01 * result);
                        break;
                    case 3:
                        result = log(abs(result) + 1.0);
                        break;
                    case 4:
                        result = result * result + 0.1;
                        break;
                    case 5:
                        result = 2.0 * result + 0.5;
                        break;
                    case 6:
                        result = sqrt(abs(result) + 1.0);
                        break;
                    case 7:
                        result = 1.0 / (abs(result) + 1.0);
                        break;
                    case 8:
                        result = tanh(0.1 * result);
                        break;
                    case 9:
                        result = result * result * result + 0.01 * result;
                        break;
                    }
                }
                return result;
            };

            dual x_forward = x_val;
            auto forward_result = derivatives(mixed_func, autodiff::detail::wrt(x_forward), autodiff::detail::at(x_forward));
            double u_forward = forward_result[0];
            double ux_forward = forward_result[1];

            CHECK(result.value() == approx(u_forward).epsilon(1e-12));
            CHECK(ux == approx(ux_forward).epsilon(1e-12));
        }

        std::cout << "[END]" << std::endl;
    }

    SECTION("testing crazy mixed composite function chain with 50 functions")
    {
        std::cout << "[crazy mixed composite]" << std::flush;

        // Test a mixed chain with different function types:
        // sin, cos, exp, log, polynomial, linear functions

        const int n = NN; // Chain of 50 mixed functions
        const double x_val = 1.5;

        var x = x_val;
        var result = x;

        // Define the pattern of functions (cycle through these 10 types)
        auto t_build_start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < n; ++i) {
            switch(i % 10) {
            case 0:
                result = sin(0.1 * result);
                break;
            case 1:
                result = cos(0.1 * result);
                break;
            case 2:
                result = exp(0.01 * result);
                break;
            case 3:
                result = log(abs(result) + 1.0);
                break;
            case 4:
                result = result * result + 0.1;
                break;
            case 5:
                result = 2.0 * result + 0.5;
                break;
            case 6:
                result = sqrt(abs(result) + 1.0);
                break;
            case 7:
                result = 1.0 / (abs(result) + 1.0);
                break;
            case 8:
                result = tanh(0.1 * result);
                break;
            case 9:
                result = result * result * result + 0.01 * result;
                break;
            }
        }

        // For mixed functions, we'll just verify that autodiff produces finite derivatives
        auto t_build_end = std::chrono::high_resolution_clock::now();
        double build_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_build_end - t_build_start).count();

        auto t_deriv_start = std::chrono::high_resolution_clock::now();
        auto ux = grad(result, x);
        auto t_deriv_end = std::chrono::high_resolution_clock::now();

        double deriv_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_deriv_end - t_deriv_start).count();

        std::cout << " [build=" << build_ms << "ms, deriv=" << deriv_ms << "ms]" << std::flush;

        CHECK(std::isfinite(val(result)));
        CHECK(std::isfinite(ux));
        CHECK(std::abs(ux) < 1e10); // Reasonable magnitude check

        // Test consistency: compute the same function with forward mode and compare
        auto mixed_func = [](dual x) -> dual {
            dual result = x;
            for(int i = 0; i < n; ++i) {
                switch(i % 10) {
                case 0:
                    result = sin(0.1 * result);
                    break;
                case 1:
                    result = cos(0.1 * result);
                    break;
                case 2:
                    result = exp(0.01 * result);
                    break;
                case 3:
                    result = log(abs(result) + 1.0);
                    break;
                case 4:
                    result = result * result + 0.1;
                    break;
                case 5:
                    result = 2.0 * result + 0.5;
                    break;
                case 6:
                    result = sqrt(abs(result) + 1.0);
                    break;
                case 7:
                    result = 1.0 / (abs(result) + 1.0);
                    break;
                case 8:
                    result = tanh(0.1 * result);
                    break;
                case 9:
                    result = result * result * result + 0.01 * result;
                    break;
                }
            }
            return result;
        };

        dual x_forward = x_val;
        auto t_fwd_start = std::chrono::high_resolution_clock::now();
        auto forward_result = derivatives(mixed_func, autodiff::detail::wrt(x_forward), autodiff::detail::at(x_forward));
        auto t_fwd_end = std::chrono::high_resolution_clock::now();
        double fwd_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_fwd_end - t_fwd_start).count();
        double u_forward = forward_result[0];
        double ux_forward = forward_result[1];

        std::cout << " [fwd=" << fwd_ms << "ms]" << std::flush;

        // Forward and reverse modes should give the same results
        CHECK(val(result) == approx(u_forward).epsilon(1e-12));
        CHECK(ux == approx(ux_forward).epsilon(1e-12));
        std::cout << "[END]" << std::endl;
    }

    /*
        SECTION("testing ultra-deep linear chain - 1000 functions")
        {
            // Test with an even longer chain of 1000 linear functions
            // f_i(x) = 1.001 * x + 0.0001 for i = 1..1000

            const int n = 1000;
            const double a = 1.001;  // Coefficient slightly > 1
            const double b = 0.0001; // Small constant
            const double x_val = 1.0;

            var x = x_val;
            var result = x;

            for(int i = 0; i < n; ++i) {
                result = a * result + b;
            }

            // Analytical derivative: product of all coefficients
            double analytical_derivative = std::pow(a, n);

            // Analytical function value:
            // After applying f(x) = ax + b repeatedly n times, we get:
            // f^n(x) = a^n * x + b * (a^(n-1) + a^(n-2) + ... + a + 1)
            //        = a^n * x + b * (a^n - 1)/(a - 1)
            double analytical_value = std::pow(a, n) * x_val + b * (std::pow(a, n) - 1.0) / (a - 1.0);

            auto ux = grad(result, x);

            CHECK( ux == approx(analytical_derivative).epsilon(1e-12) );
            CHECK( val(result) == approx(analytical_value).epsilon(1e-12) );
        }
    */
}
