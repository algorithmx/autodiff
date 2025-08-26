//                  _  _
//  _   _|_ _  _|/_|__|_*
// (_||_||_(_)(_|| |  |
//
// automatic differentiation made easier in C++
// https://github.com/algorithmx/autodiff
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.

#include <iostream>
#include <cassert>
#include <cmath>
#include "unified_expr.hpp"

using namespace autodiff::reverse::unified;

void test_basic_arithmetic() {
    std::cout << "=== Testing Basic Arithmetic ===" << std::endl;
    
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
    
    std::cout << "x = " << x.value() << std::endl;
    std::cout << "y = " << y.value() << std::endl;
    std::cout << "x + y = " << z1.value() << std::endl;
    std::cout << "x - y = " << z2.value() << std::endl;
    std::cout << "x * y = " << z3.value() << std::endl;
    std::cout << "x / y = " << z4.value() << std::endl;
    
    // Test with scalars
    auto z5 = x + 1.0;
    auto z6 = 2.0 * x;
    
    std::cout << "x + 1 = " << z5.value() << std::endl;
    std::cout << "2 * x = " << z6.value() << std::endl;
    
    assert(std::abs(z1.value() - 5.0) < 1e-10);
    assert(std::abs(z2.value() - (-1.0)) < 1e-10);
    assert(std::abs(z3.value() - 6.0) < 1e-10);
    assert(std::abs(z4.value() - (2.0/3.0)) < 1e-10);
    assert(std::abs(z5.value() - 3.0) < 1e-10);
    assert(std::abs(z6.value() - 4.0) < 1e-10);
    
    std::cout << "✓ Basic arithmetic tests passed!" << std::endl << std::endl;
}

void test_mathematical_functions() {
    std::cout << "=== Testing Mathematical Functions ===" << std::endl;
    
    auto arena = std::make_shared<ExpressionArena<double>>();
    
    UnifiedVariable<double> x(arena, 1.0);
    
    auto y1 = sin(x);
    auto y2 = cos(x);
    auto y3 = exp(x);
    auto y4 = log(x);
    auto y5 = sqrt(x);
    auto y6 = abs(x);
    
    std::cout << "x = " << x.value() << std::endl;
    std::cout << "sin(x) = " << y1.value() << std::endl;
    std::cout << "cos(x) = " << y2.value() << std::endl;
    std::cout << "exp(x) = " << y3.value() << std::endl;
    std::cout << "log(x) = " << y4.value() << std::endl;
    std::cout << "sqrt(x) = " << y5.value() << std::endl;
    std::cout << "abs(x) = " << y6.value() << std::endl;
    
    // Test power functions
    auto y7 = pow(x, 2.0);
    auto y8 = pow(2.0, x);
    
    std::cout << "x^2 = " << y7.value() << std::endl;
    std::cout << "2^x = " << y8.value() << std::endl;
    
    assert(std::abs(y1.value() - std::sin(1.0)) < 1e-10);
    assert(std::abs(y2.value() - std::cos(1.0)) < 1e-10);
    assert(std::abs(y3.value() - std::exp(1.0)) < 1e-10);
    assert(std::abs(y4.value() - std::log(1.0)) < 1e-10);
    assert(std::abs(y5.value() - std::sqrt(1.0)) < 1e-10);
    assert(std::abs(y6.value() - std::abs(1.0)) < 1e-10);
    assert(std::abs(y7.value() - 1.0) < 1e-10);
    assert(std::abs(y8.value() - 2.0) < 1e-10);
    
    std::cout << "✓ Mathematical function tests passed!" << std::endl << std::endl;
}

void test_derivative_computation() {
    std::cout << "=== Testing Derivative Computation ===" << std::endl;
    
    auto arena = std::make_shared<ExpressionArena<double>>();
    
    // Test f(x) = x^2, f'(x) = 2x
    UnifiedVariable<double> x(arena, 3.0);
    auto f = x * x;
    
    auto grads = derivatives(f, wrt(x));
    double df_dx = grads[0];
    
    std::cout << "f(x) = x^2, x = 3" << std::endl;
    std::cout << "f(3) = " << f.value() << std::endl;
    std::cout << "f'(3) = " << df_dx << std::endl;
    std::cout << "Expected: f'(3) = 6" << std::endl;
    
    assert(std::abs(f.value() - 9.0) < 1e-10);
    assert(std::abs(df_dx - 6.0) < 1e-10);
    
    // Test multivariate function f(x,y) = x*y + sin(x)
    UnifiedVariable<double> y(arena, 2.0);
    auto g = x * y + sin(x);
    
    auto grads2 = derivatives(g, wrt(x, y));
    double dg_dx = grads2[0];
    double dg_dy = grads2[1];
    
    std::cout << std::endl << "g(x,y) = x*y + sin(x), x = 3, y = 2" << std::endl;
    std::cout << "g(3,2) = " << g.value() << std::endl;
    std::cout << "∂g/∂x = " << dg_dx << std::endl;
    std::cout << "∂g/∂y = " << dg_dy << std::endl;
    std::cout << "Expected: ∂g/∂x = y + cos(x) = 2 + cos(3) ≈ " << (2.0 + std::cos(3.0)) << std::endl;
    std::cout << "Expected: ∂g/∂y = x = 3" << std::endl;
    
    assert(std::abs(g.value() - (6.0 + std::sin(3.0))) < 1e-10);
    assert(std::abs(dg_dx - (2.0 + std::cos(3.0))) < 1e-10);
    assert(std::abs(dg_dy - 3.0) < 1e-10);
    
    std::cout << "✓ Derivative computation tests passed!" << std::endl << std::endl;
}

void test_complex_expression() {
    std::cout << "=== Testing Complex Expression ===" << std::endl;
    
    auto arena = std::make_shared<ExpressionArena<double>>();
    
    // Test complex expression: f(x,y,z) = exp(x*y) + log(z) - sin(x+y*z)
    UnifiedVariable<double> x(arena, 1.0);
    UnifiedVariable<double> y(arena, 2.0);
    UnifiedVariable<double> z(arena, 3.0);
    
    auto f = exp(x * y) + log(z) - sin(x + y * z);
    
    std::cout << "f(x,y,z) = exp(x*y) + log(z) - sin(x+y*z)" << std::endl;
    std::cout << "x = 1, y = 2, z = 3" << std::endl;
    std::cout << "f(1,2,3) = " << f.value() << std::endl;
    
    auto grads = derivatives(f, wrt(x, y, z));
    std::cout << "∂f/∂x = " << grads[0] << std::endl;
    std::cout << "∂f/∂y = " << grads[1] << std::endl;
    std::cout << "∂f/∂z = " << grads[2] << std::endl;
    
    // Manually computed expected values
    double expected_f = std::exp(2.0) + std::log(3.0) - std::sin(7.0);
    double expected_df_dx = 2.0 * std::exp(2.0) - std::cos(7.0);
    double expected_df_dy = std::exp(2.0) - 3.0 * std::cos(7.0);
    double expected_df_dz = 1.0/3.0 - 2.0 * std::cos(7.0);
    
    std::cout << "Expected f = " << expected_f << std::endl;
    std::cout << "Expected ∂f/∂x = " << expected_df_dx << std::endl;
    std::cout << "Expected ∂f/∂y = " << expected_df_dy << std::endl;
    std::cout << "Expected ∂f/∂z = " << expected_df_dz << std::endl;
    
    assert(std::abs(f.value() - expected_f) < 1e-10);
    assert(std::abs(grads[0] - expected_df_dx) < 1e-10);
    assert(std::abs(grads[1] - expected_df_dy) < 1e-10);
    assert(std::abs(grads[2] - expected_df_dz) < 1e-10);
    
    std::cout << "✓ Complex expression test passed!" << std::endl << std::endl;
}

void test_arena_efficiency() {
    std::cout << "=== Testing Arena Efficiency ===" << std::endl;
    
    auto arena = std::make_shared<ExpressionArena<double>>();
    
    std::cout << "Initial arena size: " << arena->size() << std::endl;
    
    // Create variables
    UnifiedVariable<double> x(arena, 1.0);
    UnifiedVariable<double> y(arena, 2.0);
    
    std::cout << "After creating 2 variables: " << arena->size() << std::endl;
    
    // Create complex expression
    auto z = x * y + sin(x) - cos(y) + exp(x / y);
    
    std::cout << "After creating complex expression: " << arena->size() << std::endl;
    std::cout << "Expression value: " << z.value() << std::endl;
    
    // Compute derivatives
    auto grads = derivatives(z, wrt(x, y));
    std::cout << "∂z/∂x = " << grads[0] << std::endl;
    std::cout << "∂z/∂y = " << grads[1] << std::endl;
    
    std::cout << "✓ Arena efficiency test completed!" << std::endl << std::endl;
}

int main() {
    try {
        test_basic_arithmetic();
        test_mathematical_functions();
        test_derivative_computation();
        test_complex_expression();
        test_arena_efficiency();
        
        std::cout << "🎉 All tests passed successfully!" << std::endl;
        std::cout << std::endl;
        std::cout << "The unified expression system provides:" << std::endl;
        std::cout << "✓ Flat memory layout for better cache performance" << std::endl;
        std::cout << "✓ Efficient expression tree management via arena" << std::endl;
        std::cout << "✓ Type-safe automatic differentiation" << std::endl;
        std::cout << "✓ Support for complex mathematical expressions" << std::endl;
        std::cout << "✓ Compatible API with the original autodiff library" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
