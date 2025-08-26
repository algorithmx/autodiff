/**
 * @file syntax_sugar_examples.cpp
 * @brief Demonstrates different syntax sugar approaches for the unified expression system
 * 
 * This file shows multiple ways to hide the arena parameter and make the API
 * more convenient to use, similar to the original autodiff library.
 */

#include "autodiff/reverse/unified_expr.hpp"
#include <iostream>
#include <vector>

using namespace autodiff::reverse::unified;

void demonstrate_arena_scope() {
    std::cout << "\n=== 1. ArenaScope (RAII) ===\n";
    
    // Most convenient - automatic arena management
    {
        ArenaScope<double> scope;  // Creates arena automatically
        
        auto x = make_var(2.0);    // No arena parameter needed!
        auto y = make_var(3.0);    // Uses scope's arena
        auto z = sin(x) + cos(y);  // All operations use same arena
        
        std::cout << "x = " << x.value() << std::endl;
        std::cout << "y = " << y.value() << std::endl;
        std::cout << "z = sin(x) + cos(y) = " << z.value() << std::endl;
        
        // Compute derivatives
        auto dz = derivatives(z, wrt(x, y));
        std::cout << "dz/dx = " << dz[0] << std::endl;
        std::cout << "dz/dy = " << dz[1] << std::endl;
        
    } // Arena automatically destroyed here
    
    // Can also use type alias for convenience
    {
        ArenaScope_d scope;  // Same as ArenaScope<double>
        
        auto x = var(1.5);       // Convenient function for double
        auto y = constant(2.5);  // Create constants easily
        auto result = x * y + exp(x);
        
        std::cout << "result = " << result.value() << std::endl;
    }
}

void demonstrate_with_arena_macro() {
    std::cout << "\n=== 2. with_arena() macro ===\n";
    
    auto arena = std::make_shared<ExpressionArena<double>>();
    
    // Use existing arena with macro syntax
    with_arena(arena) {
        auto x = make_var(1.0);
        auto y = make_var(2.0);
        auto z = pow(x, 2) + pow(y, 2);
        
        std::cout << "z = x² + y² = " << z.value() << std::endl;
        
        auto dz = derivatives(z, wrt(x, y));
        std::cout << "dz/dx = " << dz[0] << std::endl;
        std::cout << "dz/dy = " << dz[1] << std::endl;
    }
}

void demonstrate_variable_pool() {
    std::cout << "\n=== 3. VariablePool (Object-Oriented) ===\n";
    
    VariablePool<double> pool;  // High-level interface
    
    auto x = pool.variable(3.0);
    auto y = pool.variable(4.0);
    
    // Compute within pool context
    auto result = pool.compute([&]() {
        return sqrt(pow(x, 2) + pow(y, 2));  // Pythagorean theorem
    });
    
    std::cout << "hypotenuse = " << result.value() << std::endl;
    
    // Get derivatives through pool
    auto derivatives_result = pool.derivatives(result, wrt(x, y));
    std::cout << "d(hypotenuse)/dx = " << derivatives_result[0] << std::endl;
    std::cout << "d(hypotenuse)/dy = " << derivatives_result[1] << std::endl;
    
    std::cout << "Pool has " << pool.expression_count() << " expressions\n";
}

void demonstrate_type_aliases() {
    std::cout << "\n=== 4. Type Aliases ===\n";
    
    // Use convenient type alias
    VariablePool_d pool;  // Same as VariablePool<double>
    
    auto x = pool.variable(M_PI / 4);  // 45 degrees
    auto sin_x = pool.compute([&]() { return sin(x); });
    auto cos_x = pool.compute([&]() { return cos(x); });
    
    std::cout << "sin(π/4) = " << sin_x.value() << std::endl;
    std::cout << "cos(π/4) = " << cos_x.value() << std::endl;
}

void demonstrate_complex_expression() {
    std::cout << "\n=== 5. Complex Expression (Various Approaches) ===\n";
    
    // Approach 1: ArenaScope
    {
        std::cout << "Using ArenaScope:\n";
        ArenaScope_d scope;
        
        auto x = var(1.0);
        auto y = var(2.0);
        auto z = var(3.0);
        
        // Complex expression: f(x,y,z) = sin(x*y) + exp(z/x) + log(y+z)
        auto f = sin(x * y) + exp(z / x) + log(y + z);
        
        std::cout << "f(1,2,3) = " << f.value() << std::endl;
        
        auto df = derivatives(f, wrt(x, y, z));
        std::cout << "∂f/∂x = " << df[0] << std::endl;
        std::cout << "∂f/∂y = " << df[1] << std::endl;
        std::cout << "∂f/∂z = " << df[2] << std::endl;
    }
    
    // Approach 2: VariablePool
    {
        std::cout << "\nUsing VariablePool:\n";
        VariablePool_d pool;
        
        auto x = pool.variable(1.0);
        auto y = pool.variable(2.0);
        auto z = pool.variable(3.0);
        
        auto f = pool.compute([&]() {
            return sin(x * y) + exp(z / x) + log(y + z);
        });
        
        std::cout << "f(1,2,3) = " << f.value() << std::endl;
        
        auto df = pool.derivatives(f, wrt(x, y, z));
        std::cout << "∂f/∂x = " << df[0] << std::endl;
        std::cout << "∂f/∂y = " << df[1] << std::endl;
        std::cout << "∂f/∂z = " << df[2] << std::endl;
    }
}

void demonstrate_error_handling() {
    std::cout << "\n=== 6. Error Handling ===\n";
    
    try {
        // This will fail - no arena set
        auto x = make_var(1.0);
    } catch (const std::exception& e) {
        std::cout << "Expected error: " << e.what() << std::endl;
    }
    
    // Proper usage
    {
        ArenaScope_d scope;
        auto x = make_var(1.0);  // This works
        std::cout << "Created variable successfully: " << x.value() << std::endl;
    }
}

void demonstrate_multiple_arenas() {
    std::cout << "\n=== 7. Multiple Arena Management ===\n";
    
    VariablePool_d pool1;
    VariablePool_d pool2;
    
    auto x1 = pool1.variable(1.0);
    auto x2 = pool2.variable(2.0);
    
    // Each pool manages its own arena
    auto result1 = pool1.compute([&]() { return sin(x1); });
    auto result2 = pool2.compute([&]() { return cos(x2); });
    
    std::cout << "Pool 1 result: " << result1.value() << std::endl;
    std::cout << "Pool 2 result: " << result2.value() << std::endl;
    
    std::cout << "Pool 1 expressions: " << pool1.expression_count() << std::endl;
    std::cout << "Pool 2 expressions: " << pool2.expression_count() << std::endl;
}

int main() {
    std::cout << "=== Unified Expression System - Syntax Sugar Examples ===\n";
    
    demonstrate_arena_scope();
    demonstrate_with_arena_macro();
    demonstrate_variable_pool();
    demonstrate_type_aliases();
    demonstrate_complex_expression();
    demonstrate_error_handling();
    demonstrate_multiple_arenas();
    
    std::cout << "\n=== Summary ===\n";
    std::cout << "The unified expression system provides multiple syntax sugar options:\n";
    std::cout << "1. ArenaScope: RAII-based automatic arena management\n";
    std::cout << "2. with_arena(): Macro for scoped arena usage\n";
    std::cout << "3. VariablePool: High-level object-oriented interface\n";
    std::cout << "4. Type aliases: Convenient shortcuts for common types\n";
    std::cout << "5. Factory functions: make_var(), var(), constant()\n";
    std::cout << "\nAll approaches hide the arena parameter and provide a clean API!\n";
    
    return 0;
}
