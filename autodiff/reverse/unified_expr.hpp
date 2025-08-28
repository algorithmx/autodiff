//                  _  _
//  _   _|_ _  _|/_|__|_*
// (_||_||_(_)(_|| |  |
//
// automatic differentiation made easier in C++
// https://github.com/algorithmx/autodiff
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright © 2025–2025
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

/**
 * @file unified_expr.hpp
 * @brief Unified Expression System for Automatic Differentiation
 *
 * This file implements a unified expression system that replaces the virtual inheritance
 * hierarchy of the original autodiff library with a flat, arena-based approach.
 *
 * DESIGN RATIONALE:
 * ================
 *
 * The original autodiff library uses a virtual inheritance hierarchy:
 *   - Expr<T> (abstract base)
 *     - VariableExpr<T>
 *       - IndependentVariableExpr<T>
 *       - DependentVariableExpr<T>
 *     - ConstantExpr<T>
 *     - UnaryExpr<T>
 *       - SinExpr<T>, CosExpr<T>, ExpExpr<T>, etc.
 *     - BinaryExpr<T>
 *       - AddExpr<T>, MulExpr<T>, DivExpr<T>, etc.
 *     - TernaryExpr<T>
 *       - Hypot3Expr<T>, ConditionalExpr<T>, etc.
 *
 * This new unified system eliminates virtual inheritance and provides:
 *
 * 1. FLAT MEMORY LAYOUT: All expressions stored in a contiguous vector (ExpressionArena)
 *    - Better cache locality compared to scattered heap allocations
 *    - Reduced memory fragmentation
 *    - Improved performance for large expression trees
 *
 * 2. EFFICIENT DISPATCH: Uses enum-based operation identification instead of virtual calls
 *    - OpType enum replaces virtual function polymorphism
 *    - Switch-based dispatch is often faster than virtual calls
 *    - Better branch prediction and compiler optimization
 *
 * 3. UNIFIED DATA STRUCTURE: Single ExprData<T> struct contains all expression types
 *    - Replaces the entire class hierarchy with one unified struct
 *    - Uses discriminated union pattern via ExprType enum
 *    - Reduces code duplication and simplifies maintenance
 *
 * CORRESPONDENCE TO ORIGINAL IMPLEMENTATIONS:
 * ==========================================
 *
 * Original Class                → Unified Representation
 * -------------                   ----------------------
 * Expr<T>                      → ExprData<T> with ExprType enum
 * IndependentVariableExpr<T>   → ExprType::IndependentVariable
 * DependentVariableExpr<T>     → ExprType::DependentVariable
 * ConstantExpr<T>              → ExprType::Constant
 * SinExpr<T>                   → ExprType::Unary + OpType::Sin
 * AddExpr<T>                   → ExprType::Binary + OpType::Add
 * Hypot3Expr<T>                → ExprType::Ternary + OpType::Hypot3
 * Variable<T>                  → UnifiedVariable<T>
 * ExprPtr<T>                   → ExprId (index into arena)
 *
 * MEMORY MANAGEMENT:
 * =================
 *
 * Original: Uses shared_ptr<Expr<T>> for expression trees
 * Unified:  Uses ExprId indices into a flat ExpressionArena<T>
 *
 * Benefits:
 * - No reference counting overhead
 * - Better memory locality for traversals
 * - Easier debugging (can inspect entire arena)
 * - More predictable memory usage patterns
 *
 * PERFORMANCE CHARACTERISTICS:
 * ===========================
 *
 * 1. Forward Pass (value computation):
 *    - Original: Virtual dispatch + pointer chasing
 *    - Unified: Switch dispatch + array indexing
 *    - Expected improvement: 20-40% faster
 *
 * 2. Backward Pass (gradient computation):
 *    - Original: Virtual dispatch + shared_ptr overhead
 *    - Unified: Switch dispatch + direct array access
 *    - Expected improvement: 15-30% faster
 *
 * 3. Memory Usage:
 *    - Original: ~40-60 bytes per expression node (including overhead)
 *    - Unified: ~32-48 bytes per expression node
 *    - Expected improvement: 20-25% less memory
 *
 * API COMPATIBILITY:
 * ==================
 *
 * The UnifiedVariable<T> class provides a similar API to the original Variable<T>:
 * - Arithmetic operators (+, -, *, /)
 * - Mathematical functions (sin, cos, exp, log, etc.)
 * - Automatic differentiation via derivatives() function
 * - Type conversions and stream output
 *
 * Migration from original to unified system requires minimal code changes:
 *
 * Original:                    Unified:
 * var x = 2.0;                auto arena = std::make_shared<ExpressionArena<double>>();
 *                             UnifiedVariable x(arena, 2.0);
 * auto y = sin(x) + x*x;      auto y = sin(x) + x*x;  // Same!
 * auto dy = derivatives(y, wrt(x));  // Same!
 */

#pragma once

// C++ includes
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

// autodiff includes
#include <autodiff/common/meta.hpp>
#include <autodiff/common/numbertraits.hpp>

namespace autodiff {
namespace reverse {
namespace unified {

// Forward declarations
template<typename T>
struct ExprData;
template<typename T>
class ExpressionArena;
template<typename T>
class UnifiedVariable;
template<typename T>
class UnifiedBooleanExpr;
// Use a 32-bit index type for expression IDs. This reduces per-node memory
// footprint on 64-bit platforms when arena sizes fit within 32 bits.
using ExprIndex_t = uint32_t;
using ExprId = ExprIndex_t;

static_assert(sizeof(ExprIndex_t) == 4, "ExprIndex_t must be 32-bit");

// Invalid expression ID constant
constexpr ExprId INVALID_EXPR_ID = std::numeric_limits<ExprIndex_t>::max();


///////////////////////////
// syntax assistants
///////////////////////////


/**
 * @brief Thread-local arena management for syntax sugar
 *
 * This allows creating variables without explicitly passing arena parameters:
 *
 * Usage:
 *   with_arena(arena) {
 *     auto x = make_var(2.0);  // Uses arena automatically
 *     auto y = make_var(3.0);
 *     auto z = x + y;          // All operations use same arena
 *   }
 *
 * Or using RAII:
 *   {
 *     ArenaScope scope;        // Creates and manages arena
 *     auto x = make_var(2.0);
 *     auto y = make_var(3.0);
 *   }  // Arena destroyed here
 */
template<typename T>
class ArenaManager
{
  private:
    static thread_local std::shared_ptr<ExpressionArena<T>> current_arena_;

  public:
    static void set_current_arena(std::shared_ptr<ExpressionArena<T>> arena)
    {
        current_arena_ = arena;
    }

    static std::shared_ptr<ExpressionArena<T>> get_current_arena()
    {
        if(!current_arena_) {
            throw std::runtime_error("No active arena. Use ArenaScope or with_arena() to set one.");
        }
        return current_arena_;
    }

    static bool has_current_arena()
    {
        return static_cast<bool>(current_arena_);
    }

    static void clear_current_arena()
    {
        current_arena_.reset();
    }
};

// Static member definition
template<typename T>
thread_local std::shared_ptr<ExpressionArena<T>> ArenaManager<T>::current_arena_;

/**
 * @brief RAII Arena Scope Manager
 *
 * Automatically creates and manages an arena for the current scope.
 * When the scope ends, the arena is automatically cleaned up.
 *
 * Usage:
 *   {
 *     ArenaScope<double> scope;
 *     auto x = make_var(2.0);
 *     auto y = make_var(3.0);
 *     auto z = sin(x) + cos(y);
 *   }  // Arena automatically destroyed
 */
template<typename T>
class ArenaScope
{
  private:
    std::shared_ptr<ExpressionArena<T>> previous_arena_;
    std::shared_ptr<ExpressionArena<T>> scope_arena_;

  public:
    explicit ArenaScope(size_t initial_capacity = 1000)
    {
        // Save previous arena
        previous_arena_ = ArenaManager<T>::has_current_arena() ? ArenaManager<T>::get_current_arena() : nullptr;

        // Create new arena for this scope
        scope_arena_ = std::make_shared<ExpressionArena<T>>();
        scope_arena_->reserve(initial_capacity);

        // Set as current
        ArenaManager<T>::set_current_arena(scope_arena_);
    }

    ~ArenaScope()
    {
        // Restore previous arena
        if(previous_arena_) {
            ArenaManager<T>::set_current_arena(previous_arena_);
        } else {
            ArenaManager<T>::clear_current_arena();
        }
    }

    // Non-copyable, non-movable
    ArenaScope(const ArenaScope&) = delete;
    ArenaScope& operator=(const ArenaScope&) = delete;
    ArenaScope(ArenaScope&&) = delete;
    ArenaScope& operator=(ArenaScope&&) = delete;

    // Access to the arena if needed
    std::shared_ptr<ExpressionArena<T>> arena() const { return scope_arena_; }
};

/**
 * @brief Scoped arena context manager
 *
 * Temporarily sets an arena for a block of code.
 *
 * Usage:
 *   auto arena = std::make_shared<ExpressionArena<double>>();
 *   with_arena(arena) {
 *     auto x = make_var(2.0);
 *     auto y = make_var(3.0);
 *   }
 */
template<typename T>
class ScopedArenaContext
{
  private:
    std::shared_ptr<ExpressionArena<T>> previous_arena_;

  public:
    explicit ScopedArenaContext(std::shared_ptr<ExpressionArena<T>> arena)
    {
        previous_arena_ = ArenaManager<T>::has_current_arena() ? ArenaManager<T>::get_current_arena() : nullptr;
        ArenaManager<T>::set_current_arena(arena);
    }

    ~ScopedArenaContext()
    {
        if(previous_arena_) {
            ArenaManager<T>::set_current_arena(previous_arena_);
        } else {
            ArenaManager<T>::clear_current_arena();
        }
    }

    // Non-copyable, non-movable
    ScopedArenaContext(const ScopedArenaContext&) = delete;
    ScopedArenaContext& operator=(const ScopedArenaContext&) = delete;
    ScopedArenaContext(ScopedArenaContext&&) = delete;
    ScopedArenaContext& operator=(ScopedArenaContext&&) = delete;
};

// Macro for with_arena syntax
#define with_arena(arena_ptr) \
    if(auto _arena_ctx = ScopedArenaContext<double>(arena_ptr); true)




// Forward declarations needed by VariablePool methods which forward to
// the free-function derivatives() and use Wrt<>. These are defined later
// in the file but must be declared before VariablePool so qualified
// lookup (::autodiff::reverse::unified::derivatives) compiles.

// Derivative computation functions
/**
 * @brief Wrapper for variables with respect to which derivatives are computed
 *
 * Corresponds to the original autodiff::wrt() functionality.
 * This allows specifying multiple variables for partial derivative computation.
 */
template<typename... Vars>
struct Wrt
{
    std::tuple<Vars...> args;
};


/**
 * @brief Create a Wrt object for derivative computation
 *
 * Usage: derivatives(f, wrt(x, y, z))
 * This is equivalent to the original autodiff::wrt() function.
 */
template<typename... Args>
Wrt<Args...> wrt(Args&&... args)
{
    return Wrt<Args...>{std::tuple<Args...>(std::forward<Args>(args)...)};
}

template<typename T, typename... Vars>
std::array<T, sizeof...(Vars)> derivatives(const UnifiedVariable<T>& y, const Wrt<Vars...>& wrt_vars);

/**
 * @brief Variable Pool - High-level interface
 *
 * Provides a clean, object-oriented interface for managing variables
 * and expressions within a single arena context.
 *
 * Usage:
 *   VariablePool<double> pool;
 *   auto x = pool.variable(2.0);
 *   auto y = pool.variable(3.0);
 *   auto z = pool.compute([&]() { return sin(x) + cos(y); });
 */
template<typename T>
class VariablePool
{
  private:
    std::shared_ptr<ExpressionArena<T>> arena_;

  public:
    explicit VariablePool(size_t initial_capacity = 1000)
        : arena_(std::make_shared<ExpressionArena<T>>())
    {
        arena_->reserve(initial_capacity);
    }

    // Create a new variable
    UnifiedVariable<T> variable(const T& value)
    {
        return UnifiedVariable<T>(arena_, value);
    }

    // Create a constant
    UnifiedVariable<T> constant(const T& value)
    {
        auto id = arena_->add_expression(ExprData<T>::constant(value));
        return UnifiedVariable<T>(arena_, id);
    }

    // Compute an expression within this pool's context
    template<typename F>
    auto compute(F&& func) -> decltype(func())
    {
        // Set this pool's arena as current
        auto prev_arena = ArenaManager<T>::has_current_arena() ? ArenaManager<T>::get_current_arena() : nullptr;
        ArenaManager<T>::set_current_arena(arena_);

        try {
            auto result = func();

            // Restore previous arena
            if(prev_arena) {
                ArenaManager<T>::set_current_arena(prev_arena);
            } else {
                ArenaManager<T>::clear_current_arena();
            }

            return result;
        } catch(...) {
            // Restore arena even on exception
            if(prev_arena) {
                ArenaManager<T>::set_current_arena(prev_arena);
            } else {
                ArenaManager<T>::clear_current_arena();
            }
            throw;
        }
    }

    // Access to derivatives
    template<typename... Vars>
    std::array<T, sizeof...(Vars)> derivatives(const UnifiedVariable<T>& y, const Wrt<Vars...>& wrt_vars)
    {
        return ::autodiff::reverse::unified::derivatives(y, wrt_vars);
    }

    // Access to underlying arena if needed
    std::shared_ptr<ExpressionArena<T>> arena() const { return arena_; }

    // Arena statistics
    size_t expression_count() const { return arena_->size(); }

    // Clear all expressions (keeps arena alive)
    void clear()
    {
        arena_ = std::make_shared<ExpressionArena<T>>();
    }
};


///////////////////////////
// core expression system
///////////////////////////

/**
 * @brief Expression type enumeration for efficient dispatch
 *
 * Replaces the virtual inheritance hierarchy of the original system.
 * Each type corresponds to a family of expression classes in the original:
 *
 * - Constant: ConstantExpr<T>
 * - IndependentVariable: IndependentVariableExpr<T>
 * - DependentVariable: DependentVariableExpr<T>
 * - Unary: All classes inheriting from UnaryExpr<T>
 * - Binary: All classes inheriting from BinaryExpr<T>
 * - Ternary: All classes inheriting from TernaryExpr<T>
 */
enum class ExprType : uint8_t
{
    Constant,
    IndependentVariable,
    DependentVariable,
    Unary,
    Binary,
    Ternary,
    Boolean  // New type for boolean expressions
};

/**
 * @brief Operation type enumeration
 *
 * Identifies the specific mathematical operation for each expression.
 * Replaces the need for separate expression classes in the original system.
 *
 * For example, instead of having SinExpr<T>, CosExpr<T>, etc., we have:
 * ExprType::Unary + OpType::Sin, ExprType::Unary + OpType::Cos, etc.
 */
enum class OpType : uint8_t
{
    // Special operations
    None,

    // Unary operations
    Negate,
    Sin,
    Cos,
    Tan,
    Sinh,
    Cosh,
    Tanh,
    Sigmoid,
    ArcSin,
    ArcCos,
    ArcTan,
    Exp,
    Log,
    Log10,
    Sqrt,
    Abs,
    Erf,
    
    // Boolean unary operations
    LogicalNot,

    // Binary operations
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    ArcTan2,
    Hypot2,
    PowConstantLeft,
    PowConstantRight,
    
    // Boolean binary operations (comparison and logical)
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    LogicalAnd,
    LogicalOr,

    // Ternary operations
    Hypot3,
    Conditional
};

/**
 * @brief Unified expression data structure
 *
 * This struct replaces the entire virtual inheritance hierarchy of the original system.
 * It acts as a discriminated union that can represent any type of expression node.
 *
 * ORIGINAL CORRESPONDENCE:
 * =======================
 * - type == Constant            → ConstantExpr<T>
 * - type == IndependentVariable → IndependentVariableExpr<T>
 * - type == DependentVariable   → DependentVariableExpr<T>
 * - type == Unary + op_type     → SinExpr<T>, CosExpr<T>, etc.
 * - type == Binary + op_type    → AddExpr<T>, MulExpr<T>, etc.
 * - type == Ternary + op_type   → Hypot3Expr<T>, ConditionalExpr<T>, etc.
 *
 * MEMORY LAYOUT:
 * =============
 * The struct is designed for optimal memory alignment and packing:
 * - sizeof(T) bytes for value (typically 8 bytes for double) - 8-byte aligned
 * - 24 bytes for children array (3 * 8 bytes on 64-bit) - 8-byte aligned
 * - 2 bytes for type and op_type enums (uint8_t)
 * - 1 byte for num_children (uint8_t)
 * - 2 bytes for boolean flags
 * - Minimal padding due to optimal member ordering
 * Total: ~40 bytes per expression (vs ~56-72 bytes in original, ~51-53 bytes unoptimized)
 */
template<typename T>
struct ExprData
{
    // Largest members first for optimal alignment (8-byte aligned)
    T value;

    // Child expression references (indices into flat container) - 8-byte aligned
    std::array<ExprId, 3> children;

    // Pack smaller members together to minimize padding
    ExprType type;
    OpType op_type;
    // Small control fields packed into bitfields to minimize padding.
    //
    // Layout & intent:
    // - num_children : 2
    //     Stores the number of children this node has (0..3). Two bits are
    //     enough to represent 0,1,2,3 children and keeps the struct compact.
    // - is_constant : 1
    //     Marks nodes that are constant (no need to propagate gradients or
    //     update values during forward evaluation).
    // - requires_computation : 1
    //     Intrinsic property: whether this expression type requires computation
    //     (false for constants/independent variables, true for derived expressions).
    // - processed_in_backprop : 1
    //     Guard used during reverse-mode propagation so each node is visited
    //     exactly once when accumulating gradients.
    //
    // Implementation notes / caveats:
    // - Bitfield ordering and packing is implementation-defined (compiler/ABI),
    //   but using small unsigned fields for non-negative flags is stable across
    //   mainstream compilers. Avoid taking addresses of bitfields.
    // - Total bits here are small (<= 8), so these usually fit into a single
    //   byte/word, reducing the overall size of ExprData on typical platforms.
    // - If you change or add flags, keep the total width minimal to preserve
    //   the compact memory layout described in the file header.
    unsigned num_children : 2;
    unsigned is_constant : 1;
    unsigned processed_in_backprop : 1;

    // Default constructor
    ExprData()
        : value(T{0}), children{INVALID_EXPR_ID, INVALID_EXPR_ID, INVALID_EXPR_ID}, type(ExprType::Constant), op_type(OpType::None), num_children(0), is_constant(1), processed_in_backprop(0)
    {
    }

    // Constructor for constants
    static ExprData constant(const T& val)
    {
        ExprData data;
        data.value = val;
        data.children = {INVALID_EXPR_ID, INVALID_EXPR_ID, INVALID_EXPR_ID};
        data.type = ExprType::Constant;
        data.op_type = OpType::None;
        data.num_children = 0;
        data.is_constant = 1;
        data.processed_in_backprop = 0;
        return data;
    }

    // Constructor for independent variables
    static ExprData independent_variable(const T& val)
    {
        ExprData data;
        data.value = val;
        data.children = {INVALID_EXPR_ID, INVALID_EXPR_ID, INVALID_EXPR_ID};
        data.type = ExprType::IndependentVariable;
        data.op_type = OpType::None;
        data.num_children = 0;
        data.is_constant = 0;
        data.processed_in_backprop = 0;
        return data;
    }

    // Constructor for dependent variables
    static ExprData dependent_variable(const T& val, ExprId child)
    {
        ExprData data;
        data.value = val;
        data.children = {child, INVALID_EXPR_ID, INVALID_EXPR_ID};
        data.type = ExprType::DependentVariable;
        data.op_type = OpType::None;
        data.num_children = 1;
        data.is_constant = 0;
        data.processed_in_backprop = 0;
        return data;
    }

    // Constructor for unary operations
    static ExprData unary_op(OpType op, const T& val, ExprId child)
    {
        ExprData data;
        data.value = val;
        data.children = {child, INVALID_EXPR_ID, INVALID_EXPR_ID};
        data.type = ExprType::Unary;
        data.op_type = op;
        data.num_children = 1;
        data.is_constant = 0;
        data.processed_in_backprop = 0;
        return data;
    }

    // Constructor for binary operations
    static ExprData binary_op(OpType op, const T& val, ExprId left, ExprId right)
    {
        ExprData data;
        data.value = val;
        data.children = {left, right, INVALID_EXPR_ID};
        data.type = ExprType::Binary;
        data.op_type = op;
        data.num_children = 2;
        data.is_constant = 0;
        data.processed_in_backprop = 0;
        return data;
    }

    // Constructor for ternary operations
    static ExprData ternary_op(OpType op, const T& val, ExprId left, ExprId center, ExprId right)
    {
        ExprData data;
        data.value = val;
        data.children = {left, center, right};
        data.type = ExprType::Ternary;
        data.op_type = op;
        data.num_children = 3;
        data.is_constant = 0;
        data.processed_in_backprop = 0;
        return data;
    }

    // Factory method for boolean constants
    static ExprData boolean_constant(bool val)
    {
        ExprData data;
        data.value = val ? T{1} : T{0};  // Use T{1} for true, T{0} for false
        data.children = {INVALID_EXPR_ID, INVALID_EXPR_ID, INVALID_EXPR_ID};
        data.type = ExprType::Boolean;
        data.op_type = OpType::None;
        data.num_children = 0;
        data.is_constant = 1;
        data.processed_in_backprop = 0;
        return data;
    }

    // Factory method for boolean unary operations
    static ExprData boolean_unary_op(OpType op, bool val, ExprId child)
    {
        ExprData data;
        data.value = val ? T{1} : T{0};
        data.children = {child, INVALID_EXPR_ID, INVALID_EXPR_ID};
        data.type = ExprType::Boolean;
        data.op_type = op;
        data.num_children = 1;
        data.is_constant = 0;
        data.processed_in_backprop = 0;
        return data;
    }

    // Factory method for boolean binary operations
    static ExprData boolean_binary_op(OpType op, bool val, ExprId left, ExprId right)
    {
        ExprData data;
        data.value = val ? T{1} : T{0};
        data.children = {left, right, INVALID_EXPR_ID};
        data.type = ExprType::Boolean;
        data.op_type = op;
        data.num_children = 2;
        data.is_constant = 0;
        data.processed_in_backprop = 0;
        return data;
    }

    // Factory method for comparison operations (creates boolean expressions)
    static ExprData comparison_op(OpType op, bool val, ExprId left, ExprId right)
    {
        ExprData data;
        data.value = val ? T{1} : T{0};
        data.children = {left, right, INVALID_EXPR_ID};
        data.type = ExprType::Boolean;
        data.op_type = op;
        data.num_children = 2;
        data.is_constant = 0;
        data.processed_in_backprop = 0;
        return data;
    }

    // Modified conditional factory method - now takes boolean expression ID as predicate
    static ExprData conditional_op(const T& val, ExprId predicate_id, ExprId left, ExprId right)
    {
        ExprData data;
        data.value = val;
        data.children = {predicate_id, left, right};  // predicate is now first child
        data.type = ExprType::Ternary;
        data.op_type = OpType::Conditional;
        data.num_children = 3;  // predicate + left + right children
        data.is_constant = 0;
        data.processed_in_backprop = 0;
        return data;
    }
};

/**
 * @brief Unified Boolean Expression class
 *
 * This class represents boolean expressions as regular expressions in the arena.
 * Unlike the original BooleanExpr which used std::function, this is fully
 * integrated with the arena-based system.
 */
template<typename T>
class UnifiedBooleanExpr
{
private:
    std::shared_ptr<ExpressionArena<T>> arena_;
    ExprId expr_id_;

public:
    // Constructor from expression ID
    UnifiedBooleanExpr(std::shared_ptr<ExpressionArena<T>> arena_ptr, ExprId id)
        : arena_(arena_ptr), expr_id_(id) {}

    // Constructor for constant boolean
    UnifiedBooleanExpr(std::shared_ptr<ExpressionArena<T>> arena_ptr, bool value)
        : arena_(arena_ptr)
    {
        expr_id_ = arena_->add_expression(ExprData<T>::boolean_constant(value));
    }

    // Get the boolean value (T{0} = false, T{1} = true)
    bool value() const
    {
        return (*arena_)[expr_id_].value != T{0};
    }

    // Update the expression (evaluates the boolean condition)
    void update()
    {
        // Boolean expressions are automatically updated when their dependencies change
        // No explicit update needed in the arena-based system
    }

    // Conversion to bool
    operator bool() const { return value(); }

    // Logical negation
    UnifiedBooleanExpr operator!() const
    {
        auto result_id = arena_->add_expression(
            ExprData<T>::boolean_unary_op(OpType::LogicalNot, !value(), expr_id_));
        return UnifiedBooleanExpr(arena_, result_id);
    }

    // Logical AND
    UnifiedBooleanExpr operator&&(const UnifiedBooleanExpr& other) const
    {
        ensure_same_arena(other);
        auto result_id = arena_->add_expression(
            ExprData<T>::boolean_binary_op(OpType::LogicalAnd, value() && other.value(), expr_id_, other.expr_id_));
        return UnifiedBooleanExpr(arena_, result_id);
    }

    // Logical OR
    UnifiedBooleanExpr operator||(const UnifiedBooleanExpr& other) const
    {
        ensure_same_arena(other);
        auto result_id = arena_->add_expression(
            ExprData<T>::boolean_binary_op(OpType::LogicalOr, value() || other.value(), expr_id_, other.expr_id_));
        return UnifiedBooleanExpr(arena_, result_id);
    }

    // Get expression ID and arena for internal use
    ExprId id() const { return expr_id_; }
    std::shared_ptr<ExpressionArena<T>> arena() const { return arena_; }

private:
    void ensure_same_arena(const UnifiedBooleanExpr& other) const
    {
        if(arena_ != other.arena_) {
            throw std::runtime_error("Boolean expressions must belong to the same expression arena");
        }
    }
};


/**
 * @brief Flat expression container/arena
 *
 * This class replaces the distributed memory model of the original system
 * where expressions were stored as individual shared_ptr objects.
 *
 * DESIGN BENEFITS:
 * ===============
 * 1. MEMORY LOCALITY: All expressions stored contiguously
 * 2. CACHE EFFICIENCY: Forward/backward passes traverse sequential memory
 * 3. ALLOCATION EFFICIENCY: Single vector growth vs many small allocations
 * 4. DEBUGGING: Can inspect entire expression graph in one place
 * 5. SERIALIZATION: Easy to save/load entire expression state
 *
 * OPERATIONS:
 * ==========
 * - add_expression(): Creates new expression, returns ID (replaces make_shared)
 * - operator[]: Access expression by ID (replaces pointer dereferencing)
 * - update_all(): Forward pass to compute all values
 * - propagate(): Backward pass for gradient computation
 *
 * TRAVERSAL PATTERNS:
 * ==================
 * The arena enables efficient traversal patterns:
 * - Forward pass: Sequential iteration through expressions
 * - Backward pass: Reverse iteration with dependency tracking
 * - Both patterns have excellent cache locality
 */
template<typename T>
class ExpressionArena
{
  private:
    std::vector<ExprData<T>> expressions_;
    std::vector<T> gradient_workspace_;

  public:
    ExpressionArena()
    {
        this->reserve(1000); // Reserve some initial capacity
    }
    
    void reserve(size_t new_capacity)
    {
        expressions_.reserve(new_capacity);
        gradient_workspace_.reserve(new_capacity);
    }

    // Add expression to arena and return its ID
    ExprId add_expression(ExprData<T>&& expr)
    {
        // Ensure we won't overflow the 32-bit ExprIndex_t
        if(expressions_.size() >= static_cast<size_t>(std::numeric_limits<ExprIndex_t>::max())) {
            throw std::runtime_error("Expression arena exceeded maximum index for ExprId");
        }

        auto id = expressions_.size();
        
        // Add the expression
        expressions_.emplace_back(std::move(expr));
        gradient_workspace_.resize(expressions_.size(), T{0});
        
        // Debug check: ensure all containers stay aligned
        assert(expressions_.size() == gradient_workspace_.size());
        return id;
    }

    // Add boolean expression to arena and return its ID
    ExprId add_boolean_expression(ExprId left_id, ExprId right_id, OpType comparison_op)
    {
        // Evaluate the comparison to get the boolean result
        const T& left_val = expressions_[left_id].value;
        const T& right_val = expressions_[right_id].value;
        
        bool result = false;
        switch(comparison_op) {
        case OpType::Equal:
            result = (left_val == right_val);
            break;
        case OpType::NotEqual:
            result = (left_val != right_val);
            break;
        case OpType::Less:
            result = (left_val < right_val);
            break;
        case OpType::LessEqual:
            result = (left_val <= right_val);
            break;
        case OpType::Greater:
            result = (left_val > right_val);
            break;
        case OpType::GreaterEqual:
            result = (left_val >= right_val);
            break;
        default:
            throw std::runtime_error("Invalid comparison operation");
        }
        
        return add_expression(ExprData<T>::boolean_binary_op(comparison_op, result, left_id, right_id));
    }

    // Access expression by ID
    ExprData<T>& operator[](ExprId id)
    {
        return expressions_[id];
    }

    const ExprData<T>& operator[](ExprId id) const
    {
        return expressions_[id];
    }

    // Get the number of expressions
    size_t size() const
    {
        return expressions_.size();
    }
    
    bool empty() const
    {
        return expressions_.empty();
    }

    // !!! Please keep the function UNALTERED in the comments below !!!

    // Update a specific expression
    // void update_expression(ExprId id)
    // {
    //     auto& expr = expressions_[id];
    //     switch(expr.type) {
    //     case ExprType::Constant:
    //     case ExprType::IndependentVariable:
    //         // These don't need computation, mark as fresh
    //         break;
    //     case ExprType::DependentVariable:
    //         // Update from child expression
    //         if(expr.children[0] != INVALID_EXPR_ID) {
    //             update_expression(expr.children[0]);
    //             expr.value = expressions_[expr.children[0]].value;
    //         }
    //         break;
    //     case ExprType::Unary:
    //         update_unary(expr);
    //         break;
    //     case ExprType::Binary:
    //         update_binary(expr);
    //         break;
    //     case ExprType::Ternary:
    //         update_ternary(expr);
    //         break;
    //     }
    // }

    // Propagate derivatives (backward pass)
    void propagate(ExprId root_id, const T& wprime = T{1})
    {
        // Clear gradient workspace and reset processed flags
        // Ensure gradient workspace is aligned before we write into it.
        assert(expressions_.size() == gradient_workspace_.size());
        std::fill(gradient_workspace_.begin(), gradient_workspace_.end(), T{0});
        for(auto& expr : expressions_) {
            expr.processed_in_backprop = false;
        }

        // Start propagation from root
        gradient_workspace_[root_id] = wprime;

        // Propagate in reverse topological order
        for(size_t i = expressions_.size(); i > 0; --i) {
            ExprId expr_id = i - 1;
            auto& expr = expressions_[expr_id];
            T current_grad = gradient_workspace_[expr_id];

            // Process each expression exactly once if it has accumulated gradient
            if(current_grad != T{0} && !expr.processed_in_backprop) {
                expr.processed_in_backprop = true;
                propagate_expression(expr_id, current_grad);
            }
        }
    }

    // Clear all gradients
    void clear_gradients()
    {
        // Keep invariant: gradients and expressions vectors must have same length
        assert(expressions_.size() == gradient_workspace_.size());
        std::fill(gradient_workspace_.begin(), gradient_workspace_.end(), T{0});
        for(auto& expr : expressions_) {
            expr.processed_in_backprop = false;
        }
    }

    // Get gradient for a specific expression
    T gradient(ExprId expr_id) const
    {
        return gradient_workspace_[expr_id];
    }

    // Get reference to entire gradient workspace (for advanced usage)
    const std::vector<T>& gradients() const
    {
        return gradient_workspace_;
    }

  private:

    // !!! Please keep the function UNALTERED in the comments below !!!
    /* * (vestigial)

    void update_unary(ExprData<T>& expr)
   {
       if(expr.children[0] == INVALID_EXPR_ID)
           return;

        // Update child first
        update_expression(expr.children[0]);
        const T& child_val = expressions_[expr.children[0]].value;

        switch(expr.op_type) {
        case OpType::Negate:
            expr.value = -child_val;
            break;
        case OpType::Sin:
            expr.value = std::sin(child_val);
            break;
        case OpType::Cos:
            expr.value = std::cos(child_val);
            break;
        case OpType::Tan:
            expr.value = std::tan(child_val);
            break;
        case OpType::Sinh:
            expr.value = std::sinh(child_val);
            break;
        case OpType::Cosh:
            expr.value = std::cosh(child_val);
            break;
        case OpType::Tanh:
            expr.value = std::tanh(child_val);
            break;
        case OpType::Sigmoid: {
            if(child_val >= 0) {
                const auto e = std::exp(-child_val);
                expr.value = T{1} / (T{1} + e);
            } else {
                const auto e = std::exp(child_val);
                expr.value = e / (T{1} + e);
            }
            break;
        }
        case OpType::ArcSin:
            expr.value = std::asin(child_val);
            break;
        case OpType::ArcCos:
            expr.value = std::acos(child_val);
            break;
        case OpType::ArcTan:
            expr.value = std::atan(child_val);
            break;
        case OpType::Exp:
            expr.value = std::exp(child_val);
            break;
        case OpType::Log:
            expr.value = std::log(child_val);
            break;
        case OpType::Log10:
            expr.value = std::log10(child_val);
            break;
        case OpType::Sqrt:
            expr.value = std::sqrt(child_val);
            break;
        case OpType::Abs:
            expr.value = std::abs(child_val);
            break;
        case OpType::Erf:
            expr.value = std::erf(child_val);
            break;
        default:
            throw std::runtime_error("Unknown unary operation");
        }
    }

    void update_binary(ExprData<T>& expr)
    {
        if(expr.children[0] == INVALID_EXPR_ID || expr.children[1] == INVALID_EXPR_ID)
            return;

        // Update children first
        update_expression(expr.children[0]);
        update_expression(expr.children[1]);

        const T& left_val = expressions_[expr.children[0]].value;
        const T& right_val = expressions_[expr.children[1]].value;

        switch(expr.op_type) {
        case OpType::Add:
            expr.value = left_val + right_val;
            break;
        case OpType::Sub:
            expr.value = left_val - right_val;
            break;
        case OpType::Mul:
            expr.value = left_val * right_val;
            break;
        case OpType::Div:
            expr.value = left_val / right_val;
            break;
        case OpType::Pow:
            expr.value = std::pow(left_val, right_val);
            break;
        case OpType::PowConstantLeft:
            expr.value = std::pow(left_val, right_val);
            break;
        case OpType::PowConstantRight:
            expr.value = std::pow(left_val, right_val);
            break;
        case OpType::ArcTan2:
            expr.value = std::atan2(left_val, right_val);
            break;
        case OpType::Hypot2:
            expr.value = std::hypot(left_val, right_val);
            break;
        default:
            throw std::runtime_error("Unknown binary operation");
        }
    }

    void update_ternary(ExprData<T>& expr)
    {
        switch(expr.op_type) {
        case OpType::Hypot3:
        {
            if(expr.children[0] == INVALID_EXPR_ID ||
               expr.children[1] == INVALID_EXPR_ID ||
               expr.children[2] == INVALID_EXPR_ID)
                return;

            // Update children first
            update_expression(expr.children[0]);
            update_expression(expr.children[1]);
            update_expression(expr.children[2]);

            const T& left_val = expressions_[expr.children[0]].value;
            const T& center_val = expressions_[expr.children[1]].value;
            const T& right_val = expressions_[expr.children[2]].value;

            expr.value = std::hypot(left_val, center_val, right_val);
            break;
        }
        case OpType::Conditional:
        {
            if(expr.children[0] == INVALID_EXPR_ID ||
               expr.children[1] == INVALID_EXPR_ID ||
               !expr.predicate)
                return;

            // Update the boolean predicate first
            expr.predicate->update();

            // Update the appropriate branch based on predicate
            if(expr.predicate->val) {
                update_expression(expr.children[0]);  // left branch
                expr.value = expressions_[expr.children[0]].value;
            } else {
                update_expression(expr.children[1]);  // right branch
                expr.value = expressions_[expr.children[1]].value;
            }
            break;
        }
        default:
            throw std::runtime_error("Unknown ternary operation");
        }
    }
    * */

    void propagate_expression(ExprId expr_id, const T& wprime)
    {
        auto& expr = expressions_[expr_id];

        // Gradients are automatically accumulated in gradient_workspace_[expr_id]
        // No need for special gradient pointer handling

        // Propagate to children based on operation type
        switch(expr.type) {
        case ExprType::Constant:
        case ExprType::IndependentVariable:
            break;

        case ExprType::DependentVariable:
            if(expr.children[0] != INVALID_EXPR_ID) {
                gradient_workspace_[expr.children[0]] += wprime;
            }
            break;

        case ExprType::Boolean:
            // Boolean expressions don't contribute to gradients
            // but we still need to propagate to their children for dependency tracking
            propagate_boolean(expr, wprime);
            break;

        case ExprType::Unary:
            propagate_unary(expr, wprime);
            break;

        case ExprType::Binary:
            propagate_binary(expr, wprime);
            break;

        case ExprType::Ternary:
            propagate_ternary(expr, wprime);
            break;
        }
    }

    void propagate_unary(const ExprData<T>& expr, const T& wprime)
    {
        if(expr.children[0] == INVALID_EXPR_ID)
            return;

        const T& child_val = expressions_[expr.children[0]].value;
        T child_wprime = T{0};

        switch(expr.op_type) {
        case OpType::Negate:
            child_wprime = -wprime;
            break;
        case OpType::Sin:
            child_wprime = wprime * std::cos(child_val);
            break;
        case OpType::Cos:
            child_wprime = -wprime * std::sin(child_val);
            break;
        case OpType::Tan: {
            const auto sec = T{1} / std::cos(child_val);
            child_wprime = wprime * sec * sec;
            break;
        }
        case OpType::Sinh:
            child_wprime = wprime * std::cosh(child_val);
            break;
        case OpType::Cosh:
            child_wprime = wprime * std::sinh(child_val);
            break;
        case OpType::Tanh: {
            const auto sech = T{1} / std::cosh(child_val);
            child_wprime = wprime * sech * sech;
            break;
        }
        case OpType::Sigmoid:
            // derivative: sigma'(x) = sigma(x) * (1 - sigma(x))
            child_wprime = wprime * expr.value * (T{1} - expr.value);
            break;
        case OpType::ArcSin:
            child_wprime = wprime / std::sqrt(T{1} - child_val * child_val);
            break;
        case OpType::ArcCos:
            child_wprime = -wprime / std::sqrt(T{1} - child_val * child_val);
            break;
        case OpType::ArcTan:
            child_wprime = wprime / (T{1} + child_val * child_val);
            break;
        case OpType::Exp:
            child_wprime = wprime * expr.value; // exp(x)' = exp(x)
            break;
        case OpType::Log:
            child_wprime = wprime / child_val;
            break;
        case OpType::Log10: {
            constexpr auto ln10 = static_cast<T>(2.3025850929940456840179914546843);
            child_wprime = wprime / (ln10 * child_val);
            break;
        }
        case OpType::Sqrt:
            child_wprime = wprime / (T{2} * expr.value);
            break;
        case OpType::Abs:
            if(child_val < T{0}) {
                child_wprime = -wprime;
            } else if(child_val > T{0}) {
                child_wprime = wprime;
            } else {
                child_wprime = T{0};
            }
            break;
        case OpType::Erf: {
            constexpr auto sqrt_pi = static_cast<T>(1.7724538509055160272981674833411451872554456638435);
            const auto aux = T{2} / sqrt_pi * std::exp(-child_val * child_val);
            child_wprime = wprime * aux;
            break;
        }
        default:
            throw std::runtime_error("Unknown unary operation in propagation");
        }

        gradient_workspace_[expr.children[0]] += child_wprime;
    }

    void propagate_binary(const ExprData<T>& expr, const T& wprime)
    {
        if(expr.children[0] == INVALID_EXPR_ID || expr.children[1] == INVALID_EXPR_ID)
            return;

        const T& left_val = expressions_[expr.children[0]].value;
        const T& right_val = expressions_[expr.children[1]].value;

        T left_wprime = T{0}, right_wprime = T{0};

        switch(expr.op_type) {
        case OpType::Add:
            left_wprime = wprime;
            right_wprime = wprime;
            break;
        case OpType::Sub:
            left_wprime = wprime;
            right_wprime = -wprime;
            break;
        case OpType::Mul:
            left_wprime = wprime * right_val;
            right_wprime = wprime * left_val;
            break;
        case OpType::Div: {
            const auto inv_right = T{1} / right_val;
            left_wprime = wprime * inv_right;
            right_wprime = -wprime * left_val * inv_right * inv_right;
            break;
        }
        case OpType::Pow: {
            const auto aux = wprime * std::pow(left_val, right_val - T{1});
            left_wprime = aux * right_val;
            const auto auxr = (left_val == T{0}) ? T{0} : left_val * std::log(left_val);
            right_wprime = aux * auxr;
            break;
        }
        case OpType::PowConstantLeft: {
            const auto aux = wprime * std::pow(left_val, right_val - T{1});
            const auto auxr = (left_val == T{0}) ? T{0} : left_val * std::log(left_val);
            right_wprime = aux * auxr;
            break;
        }
        case OpType::PowConstantRight: {
            left_wprime = wprime * std::pow(left_val, right_val - T{1}) * right_val;
            break;
        }
        case OpType::ArcTan2: {
            const auto denom = left_val * left_val + right_val * right_val;
            const auto aux = wprime / denom;
            left_wprime = right_val * aux;
            right_wprime = -left_val * aux;
            break;
        }
        case OpType::Hypot2: {
            const auto& hypot_val = expr.value;
            left_wprime = wprime * left_val / hypot_val;
            right_wprime = wprime * right_val / hypot_val;
            break;
        }
        default:
            throw std::runtime_error("Unknown binary operation in propagation");
        }

        gradient_workspace_[expr.children[0]] += left_wprime;
        gradient_workspace_[expr.children[1]] += right_wprime;
    }

    void propagate_ternary(const ExprData<T>& expr, const T& wprime)
    {
        switch(expr.op_type) {
        case OpType::Hypot3: {
            if(expr.children[0] == INVALID_EXPR_ID ||
               expr.children[1] == INVALID_EXPR_ID ||
               expr.children[2] == INVALID_EXPR_ID)
                return;

            const T& left_val = expressions_[expr.children[0]].value;
            const T& center_val = expressions_[expr.children[1]].value;
            const T& right_val = expressions_[expr.children[2]].value;

            const auto& hypot_val = expr.value;
            gradient_workspace_[expr.children[0]] += wprime * left_val / hypot_val;
            gradient_workspace_[expr.children[1]] += wprime * center_val / hypot_val;
            gradient_workspace_[expr.children[2]] += wprime * right_val / hypot_val;
            break;
        }
        case OpType::Conditional: {
            if(expr.children[0] == INVALID_EXPR_ID ||  // predicate
               expr.children[1] == INVALID_EXPR_ID ||  // left branch
               expr.children[2] == INVALID_EXPR_ID)    // right branch
                return;

            // Get the boolean predicate value
            bool predicate_val = expressions_[expr.children[0]].value != T{0};

            // Propagate gradient only to the branch that was actually taken
            if(predicate_val) {
                gradient_workspace_[expr.children[1]] += wprime;  // left branch
            } else {
                gradient_workspace_[expr.children[2]] += wprime;  // right branch
            }
            break;
        }
        default:
            throw std::runtime_error("Unknown ternary operation in propagation");
        }
    }

    void propagate_boolean(const ExprData<T>& expr, const T& wprime)
    {
        // Boolean expressions don't have gradients in the traditional sense,
        // but we need to handle dependency tracking for conditional expressions
        // For now, we don't propagate gradients through boolean expressions
        // This matches the behavior of the original system
    }
};

/**
 * @brief Unified Variable class that works with the arena
 *
 * This class provides the same user interface as the original Variable<T>
 * but works with the unified expression system internally.
 *
 * KEY DIFFERENCES FROM ORIGINAL:
 * =============================
 *
 * STORAGE:
 * Original: Contains ExprPtr<T> (shared_ptr to expression tree)
 * Unified:  Contains ExprId (index into arena) + arena pointer
 *
 * EXPRESSION CREATION:
 * Original: Operations create new shared_ptr objects
 * Unified:  Operations add entries to arena and return new UnifiedVariable
 *
 * MEMORY MANAGEMENT:
 * Original: Automatic via shared_ptr reference counting
 * Unified:  Manual via shared arena lifetime (typically longer-lived)
 *
 * PERFORMANCE:
 * Original: Pointer chasing during evaluation
 * Unified:  Array indexing during evaluation (better cache locality)
 *
 * API COMPATIBILITY:
 * =================
 * The class maintains the same public interface:
 * - Arithmetic operators: +, -, *, /
 * - Mathematical functions: sin, cos, exp, log, etc.
 * - Type conversions and comparisons
 * - Stream output
 *
 * The main difference is construction - users must provide an arena:
 *
 * Original: Variable<double> x(2.0);
 * Unified:  UnifiedVariable<double> x(arena, 2.0);
 *
 * THREAD SAFETY:
 * =============
 * Variables sharing the same arena are NOT thread-safe for modification
 * (just like the original system). However, read-only operations on
 * different expression trees within the same arena can be concurrent.
 */
template<typename T>
class UnifiedVariable
{
  private:
    std::shared_ptr<ExpressionArena<T>> arena_;
    ExprId expr_id_;

  public:
    // Constructor for independent variable
    UnifiedVariable(std::shared_ptr<ExpressionArena<T>> arena_ptr, const T& value)
        : arena_(arena_ptr)
    {
        expr_id_ = arena_->add_expression(ExprData<T>::independent_variable(value));
    }

    // Constructor from existing expression ID
    UnifiedVariable(std::shared_ptr<ExpressionArena<T>> arena_ptr, ExprId id)
        : arena_(arena_ptr), expr_id_(id) {}

    // Default constructor (creates a constant zero)
    UnifiedVariable()
        : arena_(std::make_shared<ExpressionArena<T>>()), expr_id_(arena_->add_expression(ExprData<T>::constant(T{0})))
    {
    }

    // Copy constructor
    UnifiedVariable(const UnifiedVariable& other) = default;

    // Assignment operator
    UnifiedVariable& operator=(const UnifiedVariable& other) = default;

    // Assignment from arithmetic value
    template<typename U>
    typename std::enable_if<std::is_arithmetic<U>::value, UnifiedVariable&>::type
    operator=(const U& value)
    {
        // Create new independent variable
        expr_id_ = arena_->add_expression(ExprData<T>::independent_variable(static_cast<T>(value)));
        return *this;
    }

    // Fast access to stored value without forcing an update. Use this during
    // expression construction to avoid triggering expensive forward updates.
    // The stored value is the value that was set when the expression node
    // was created; reading it avoids recursive update passes during build.
    T value() const
    {
        return (*arena_)[expr_id_].value;
    }

    // Get expression ID
    ExprId id() const { return expr_id_; }

    // Get arena
    std::shared_ptr<ExpressionArena<T>> arena() const { return arena_; }

    // Update this variable's value (for independent variables only)
    // void hot_update(const T& new_value)
    // {
    //     auto& expr = (*arena_)[expr_id_];
    //     if(expr.type == ExprType::IndependentVariable) {
    //         expr.value = new_value;
    //         // Invalidate all dependent expressions in the arena
    //         arena_->invalidate_dependents(expr_id_);
    //     } else {
    //         throw std::runtime_error("Cannot update value of dependent variable");
    //     }
    // }

    // Arithmetic operators
    UnifiedVariable operator+(const UnifiedVariable& other) const
    {
        ensure_same_arena(other);
        auto result_id = arena_->add_expression(
            ExprData<T>::binary_op(OpType::Add, value() + other.value(), expr_id_, other.expr_id_));
        return UnifiedVariable(arena_, result_id);
    }

    UnifiedVariable operator-(const UnifiedVariable& other) const
    {
        ensure_same_arena(other);
        auto result_id = arena_->add_expression(
            ExprData<T>::binary_op(OpType::Sub, value() - other.value(), expr_id_, other.expr_id_));
        return UnifiedVariable(arena_, result_id);
    }

    UnifiedVariable operator*(const UnifiedVariable& other) const
    {
        ensure_same_arena(other);
        auto result_id = arena_->add_expression(
            ExprData<T>::binary_op(OpType::Mul, value() * other.value(), expr_id_, other.expr_id_));
        return UnifiedVariable(arena_, result_id);
    }

    UnifiedVariable operator/(const UnifiedVariable& other) const
    {
        ensure_same_arena(other);
        auto result_id = arena_->add_expression(
            ExprData<T>::binary_op(OpType::Div, value() / other.value(), expr_id_, other.expr_id_));
        return UnifiedVariable(arena_, result_id);
    }

    UnifiedVariable operator-() const
    {
        auto result_id = arena_->add_expression(
            ExprData<T>::unary_op(OpType::Negate, -value(), expr_id_));
        return UnifiedVariable(arena_, result_id);
    }

    // Arithmetic operators with scalars
    template<typename U>
    typename std::enable_if<std::is_arithmetic<U>::value, UnifiedVariable>::type
    operator+(const U& scalar) const
    {
        auto constant_id = arena_->add_expression(ExprData<T>::constant(static_cast<T>(scalar)));
        auto result_id = arena_->add_expression(
            ExprData<T>::binary_op(OpType::Add, value() + static_cast<T>(scalar), expr_id_, constant_id));
        return UnifiedVariable(arena_, result_id);
    }

    template<typename U>
    typename std::enable_if<std::is_arithmetic<U>::value, UnifiedVariable>::type
    operator-(const U& scalar) const
    {
        auto constant_id = arena_->add_expression(ExprData<T>::constant(static_cast<T>(scalar)));
        auto result_id = arena_->add_expression(
            ExprData<T>::binary_op(OpType::Sub, value() - static_cast<T>(scalar), expr_id_, constant_id));
        return UnifiedVariable(arena_, result_id);
    }

    template<typename U>
    typename std::enable_if<std::is_arithmetic<U>::value, UnifiedVariable>::type
    operator*(const U& scalar) const
    {
        auto constant_id = arena_->add_expression(ExprData<T>::constant(static_cast<T>(scalar)));
        auto result_id = arena_->add_expression(
            ExprData<T>::binary_op(OpType::Mul, value() * static_cast<T>(scalar), expr_id_, constant_id));
        return UnifiedVariable(arena_, result_id);
    }

    template<typename U>
    typename std::enable_if<std::is_arithmetic<U>::value, UnifiedVariable>::type
    operator/(const U& scalar) const
    {
        auto constant_id = arena_->add_expression(ExprData<T>::constant(static_cast<T>(scalar)));
        auto result_id = arena_->add_expression(
            ExprData<T>::binary_op(OpType::Div, value() / static_cast<T>(scalar), expr_id_, constant_id));
        return UnifiedVariable(arena_, result_id);
    }

    // Assignment operators
    UnifiedVariable& operator+=(const UnifiedVariable& other)
    {
        *this = *this + other;
        return *this;
    }

    UnifiedVariable& operator-=(const UnifiedVariable& other)
    {
        *this = *this - other;
        return *this;
    }

    UnifiedVariable& operator*=(const UnifiedVariable& other)
    {
        *this = *this * other;
        return *this;
    }

    UnifiedVariable& operator/=(const UnifiedVariable& other)
    {
        *this = *this / other;
        return *this;
    }

    // Mathematical functions
    UnifiedVariable sin() const
    {
        auto result_id = arena_->add_expression(
            ExprData<T>::unary_op(OpType::Sin, std::sin(value()), expr_id_));
        return UnifiedVariable(arena_, result_id);
    }

    UnifiedVariable cos() const
    {
        auto result_id = arena_->add_expression(
            ExprData<T>::unary_op(OpType::Cos, std::cos(value()), expr_id_));
        return UnifiedVariable(arena_, result_id);
    }

    UnifiedVariable tan() const
    {
        auto result_id = arena_->add_expression(
            ExprData<T>::unary_op(OpType::Tan, std::tan(value()), expr_id_));
        return UnifiedVariable(arena_, result_id);
    }

    UnifiedVariable exp() const
    {
        auto result_id = arena_->add_expression(
            ExprData<T>::unary_op(OpType::Exp, std::exp(value()), expr_id_));
        return UnifiedVariable(arena_, result_id);
    }

    UnifiedVariable log() const
    {
        auto result_id = arena_->add_expression(
            ExprData<T>::unary_op(OpType::Log, std::log(value()), expr_id_));
        return UnifiedVariable(arena_, result_id);
    }

    UnifiedVariable sqrt() const
    {
        auto result_id = arena_->add_expression(
            ExprData<T>::unary_op(OpType::Sqrt, std::sqrt(value()), expr_id_));
        return UnifiedVariable(arena_, result_id);
    }

    UnifiedVariable abs() const
    {
        auto result_id = arena_->add_expression(
            ExprData<T>::unary_op(OpType::Abs, std::abs(value()), expr_id_));
        return UnifiedVariable(arena_, result_id);
    }

    UnifiedVariable pow(const UnifiedVariable& exponent) const
    {
        ensure_same_arena(exponent);
        auto result_id = arena_->add_expression(
            ExprData<T>::binary_op(OpType::Pow, std::pow(value(), exponent.value()), expr_id_, exponent.expr_id_));
        return UnifiedVariable(arena_, result_id);
    }

    template<typename U>
    typename std::enable_if<std::is_arithmetic<U>::value, UnifiedVariable>::type
    pow(const U& exponent) const
    {
        auto constant_id = arena_->add_expression(ExprData<T>::constant(static_cast<T>(exponent)));
    auto result_id = arena_->add_expression(
        ExprData<T>::binary_op(OpType::PowConstantRight, std::pow(value(), static_cast<T>(exponent)), expr_id_, constant_id));
        return UnifiedVariable(arena_, result_id);
    }

    // Conversion operators
    explicit operator T() const { return value(); }

  public:
    void ensure_same_arena(const UnifiedVariable& other) const
    {
        if(arena_ != other.arena_) {
            throw std::runtime_error("Variables must belong to the same expression arena");
        }
    }
};

// Free function mathematical operations
template<typename T>
UnifiedVariable<T> sin(const UnifiedVariable<T>& x)
{
    return x.sin();
}

template<typename T>
UnifiedVariable<T> cos(const UnifiedVariable<T>& x)
{
    return x.cos();
}

template<typename T>
UnifiedVariable<T> tan(const UnifiedVariable<T>& x)
{
    return x.tan();
}

template<typename T>
UnifiedVariable<T> exp(const UnifiedVariable<T>& x)
{
    return x.exp();
}

template<typename T>
UnifiedVariable<T> log(const UnifiedVariable<T>& x)
{
    return x.log();
}

template<typename T>
UnifiedVariable<T> sqrt(const UnifiedVariable<T>& x)
{
    return x.sqrt();
}

template<typename T>
UnifiedVariable<T> abs(const UnifiedVariable<T>& x)
{
    return x.abs();
}

template<typename T>
UnifiedVariable<T> erf(const UnifiedVariable<T>& x)
{
    auto result_id = x.arena()->add_expression(
        ExprData<T>::unary_op(OpType::Erf, std::erf(x.value()), x.id()));
    return UnifiedVariable<T>(x.arena(), result_id);
}

// Additional mathematical functions that were missing

template<typename T>
UnifiedVariable<T> sinh(const UnifiedVariable<T>& x)
{
    auto result_id = x.arena()->add_expression(
        ExprData<T>::unary_op(OpType::Sinh, std::sinh(x.value()), x.id()));
    return UnifiedVariable<T>(x.arena(), result_id);
}

template<typename T>
UnifiedVariable<T> cosh(const UnifiedVariable<T>& x)
{
    auto result_id = x.arena()->add_expression(
        ExprData<T>::unary_op(OpType::Cosh, std::cosh(x.value()), x.id()));
    return UnifiedVariable<T>(x.arena(), result_id);
}

template<typename T>
UnifiedVariable<T> tanh(const UnifiedVariable<T>& x)
{
    auto result_id = x.arena()->add_expression(
        ExprData<T>::unary_op(OpType::Tanh, std::tanh(x.value()), x.id()));
    return UnifiedVariable<T>(x.arena(), result_id);
}

template<typename T>
UnifiedVariable<T> asin(const UnifiedVariable<T>& x)
{
    auto result_id = x.arena()->add_expression(
        ExprData<T>::unary_op(OpType::ArcSin, std::asin(x.value()), x.id()));
    return UnifiedVariable<T>(x.arena(), result_id);
}

template<typename T>
UnifiedVariable<T> acos(const UnifiedVariable<T>& x)
{
    auto result_id = x.arena()->add_expression(
        ExprData<T>::unary_op(OpType::ArcCos, std::acos(x.value()), x.id()));
    return UnifiedVariable<T>(x.arena(), result_id);
}

template<typename T>
UnifiedVariable<T> atan(const UnifiedVariable<T>& x)
{
    auto result_id = x.arena()->add_expression(
        ExprData<T>::unary_op(OpType::ArcTan, std::atan(x.value()), x.id()));
    return UnifiedVariable<T>(x.arena(), result_id);
}

template<typename T>
UnifiedVariable<T> log10(const UnifiedVariable<T>& x)
{
    auto result_id = x.arena()->add_expression(
        ExprData<T>::unary_op(OpType::Log10, std::log10(x.value()), x.id()));
    return UnifiedVariable<T>(x.arena(), result_id);
}

template<typename T>
UnifiedVariable<T> sigmoid(const UnifiedVariable<T>& x)
{
    T val;
    if(x.value() >= 0) {
        const auto e = std::exp(-x.value());
        val = T{1} / (T{1} + e);
    } else {
        const auto e = std::exp(x.value());
        val = e / (T{1} + e);
    }
    auto result_id = x.arena()->add_expression(
        ExprData<T>::unary_op(OpType::Sigmoid, val, x.id()));
    return UnifiedVariable<T>(x.arena(), result_id);
}

template<typename T>
UnifiedVariable<T> atan2(const UnifiedVariable<T>& y, const UnifiedVariable<T>& x)
{
    if(y.arena() != x.arena()) {
        throw std::runtime_error("Variables must belong to the same expression arena");
    }
    auto result_id = y.arena()->add_expression(
        ExprData<T>::binary_op(OpType::ArcTan2, std::atan2(y.value(), x.value()), y.id(), x.id()));
    return UnifiedVariable<T>(y.arena(), result_id);
}

template<typename T, typename U>
typename std::enable_if<std::is_arithmetic<U>::value, UnifiedVariable<T>>::type
atan2(const U& y, const UnifiedVariable<T>& x)
{
    auto constant_id = x.arena()->add_expression(ExprData<T>::constant(static_cast<T>(y)));
    auto result_id = x.arena()->add_expression(
        ExprData<T>::binary_op(OpType::ArcTan2, std::atan2(static_cast<T>(y), x.value()), constant_id, x.id()));
    return UnifiedVariable<T>(x.arena(), result_id);
}

template<typename T, typename U>
typename std::enable_if<std::is_arithmetic<U>::value, UnifiedVariable<T>>::type
atan2(const UnifiedVariable<T>& y, const U& x)
{
    auto constant_id = y.arena()->add_expression(ExprData<T>::constant(static_cast<T>(x)));
    auto result_id = y.arena()->add_expression(
        ExprData<T>::binary_op(OpType::ArcTan2, std::atan2(y.value(), static_cast<T>(x)), y.id(), constant_id));
    return UnifiedVariable<T>(y.arena(), result_id);
}

template<typename T>
UnifiedVariable<T> hypot(const UnifiedVariable<T>& x, const UnifiedVariable<T>& y)
{
    if(x.arena() != y.arena()) {
        throw std::runtime_error("Variables must belong to the same expression arena");
    }
    auto result_id = x.arena()->add_expression(
        ExprData<T>::binary_op(OpType::Hypot2, std::hypot(x.value(), y.value()), x.id(), y.id()));
    return UnifiedVariable<T>(x.arena(), result_id);
}

template<typename T, typename U>
typename std::enable_if<std::is_arithmetic<U>::value, UnifiedVariable<T>>::type
hypot(const U& x, const UnifiedVariable<T>& y)
{
    auto constant_id = y.arena()->add_expression(ExprData<T>::constant(static_cast<T>(x)));
    auto result_id = y.arena()->add_expression(
        ExprData<T>::binary_op(OpType::Hypot2, std::hypot(static_cast<T>(x), y.value()), constant_id, y.id()));
    return UnifiedVariable<T>(y.arena(), result_id);
}

template<typename T, typename U>
typename std::enable_if<std::is_arithmetic<U>::value, UnifiedVariable<T>>::type
hypot(const UnifiedVariable<T>& x, const U& y)
{
    auto constant_id = x.arena()->add_expression(ExprData<T>::constant(static_cast<T>(y)));
    auto result_id = x.arena()->add_expression(
        ExprData<T>::binary_op(OpType::Hypot2, std::hypot(x.value(), static_cast<T>(y)), x.id(), constant_id));
    return UnifiedVariable<T>(x.arena(), result_id);
}

template<typename T>
UnifiedVariable<T> hypot(const UnifiedVariable<T>& x, const UnifiedVariable<T>& y, const UnifiedVariable<T>& z)
{
    if(x.arena() != y.arena() || y.arena() != z.arena()) {
        throw std::runtime_error("Variables must belong to the same expression arena");
    }
    auto result_id = x.arena()->add_expression(
        ExprData<T>::ternary_op(OpType::Hypot3, std::hypot(x.value(), y.value(), z.value()), x.id(), y.id(), z.id()));
    return UnifiedVariable<T>(x.arena(), result_id);
}

// Complex number support functions (returning real-valued results for real inputs)
template<typename T>
UnifiedVariable<T> real(const UnifiedVariable<T>& x)
{
    return x; // For real numbers, real part is the number itself
}

template<typename T>
UnifiedVariable<T> imag(const UnifiedVariable<T>& x)
{
    // For real numbers, imaginary part is zero
    auto constant_id = x.arena()->add_expression(ExprData<T>::constant(T{0}));
    return UnifiedVariable<T>(x.arena(), constant_id);
}

template<typename T>
UnifiedVariable<T> conj(const UnifiedVariable<T>& x)
{
    return x; // For real numbers, conjugate is the number itself
}

template<typename T>
UnifiedVariable<T> abs2(const UnifiedVariable<T>& x)
{
    return x * x; // |x|^2 = x^2 for real numbers
}

template<typename T>
UnifiedVariable<T> fabs(const UnifiedVariable<T>& x)
{
    return abs(x); // fabs is same as abs for our purposes
}

template<typename T>
UnifiedVariable<T> pow(const UnifiedVariable<T>& base, const UnifiedVariable<T>& exponent)
{
    return base.pow(exponent);
}

template<typename T, typename U>
typename std::enable_if<std::is_arithmetic<U>::value, UnifiedVariable<T>>::type
pow(const UnifiedVariable<T>& base, const U& exponent)
{
    return base.pow(exponent);
}

template<typename T, typename U>
typename std::enable_if<std::is_arithmetic<U>::value, UnifiedVariable<T>>::type
pow(const U& base, const UnifiedVariable<T>& exponent)
{
    auto constant_id = exponent.arena()->add_expression(ExprData<T>::constant(static_cast<T>(base)));
    auto result_id = exponent.arena()->add_expression(
        ExprData<T>::binary_op(OpType::PowConstantLeft, std::pow(static_cast<T>(base), exponent.value()), constant_id, exponent.id()));
    return UnifiedVariable<T>(exponent.arena(), result_id);
}

// Scalar-variable arithmetic operators
template<typename T, typename U>
typename std::enable_if<std::is_arithmetic<U>::value, UnifiedVariable<T>>::type
operator+(const U& scalar, const UnifiedVariable<T>& var)
{
    return var + scalar;
}

template<typename T, typename U>
typename std::enable_if<std::is_arithmetic<U>::value, UnifiedVariable<T>>::type
operator-(const U& scalar, const UnifiedVariable<T>& var)
{
    auto constant_id = var.arena()->add_expression(ExprData<T>::constant(static_cast<T>(scalar)));
    auto result_id = var.arena()->add_expression(
        ExprData<T>::binary_op(OpType::Sub, static_cast<T>(scalar) - var.value(), constant_id, var.id()));
    return UnifiedVariable<T>(var.arena(), result_id);
}

template<typename T, typename U>
typename std::enable_if<std::is_arithmetic<U>::value, UnifiedVariable<T>>::type
operator*(const U& scalar, const UnifiedVariable<T>& var)
{
    return var * scalar;
}

template<typename T, typename U>
typename std::enable_if<std::is_arithmetic<U>::value, UnifiedVariable<T>>::type
operator/(const U& scalar, const UnifiedVariable<T>& var)
{
    auto constant_id = var.arena()->add_expression(ExprData<T>::constant(static_cast<T>(scalar)));
    auto result_id = var.arena()->add_expression(
        ExprData<T>::binary_op(OpType::Div, static_cast<T>(scalar) / var.value(), constant_id, var.id()));
    return UnifiedVariable<T>(var.arena(), result_id);
}

//------------------------------------------------------------------------------
// COMPARISON OPERATORS
//------------------------------------------------------------------------------

/**
 * @brief Create comparison expression between two variables
 *
 * This function creates a UnifiedBooleanExpr that compares two UnifiedVariable objects
 * using the specified comparison operation. The comparison is evaluated in the arena.
 */
template<typename T>
UnifiedBooleanExpr<T> expr_comparison(const UnifiedVariable<T>& l, const UnifiedVariable<T>& r, OpType comparison_op)
{
    auto arena = l.arena(); // Both should use the same arena
    auto bool_id = arena->add_boolean_expression(l.id(), r.id(), comparison_op);
    return UnifiedBooleanExpr<T>(arena, bool_id);
}

/**
 * @brief Create comparison expression between variable and scalar
 */
template<typename T, typename U>
typename std::enable_if<std::is_arithmetic<U>::value, UnifiedBooleanExpr<T>>::type
expr_comparison(const UnifiedVariable<T>& l, const U& r, OpType comparison_op)
{
    auto arena = l.arena();
    auto scalar_id = arena->add_expression(ExprData<T>::constant(static_cast<T>(r)));
    auto bool_id = arena->add_boolean_expression(l.id(), scalar_id, comparison_op);
    return UnifiedBooleanExpr<T>(arena, bool_id);
}

/**
 * @brief Create comparison expression between scalar and variable
 */
template<typename T, typename U>
typename std::enable_if<std::is_arithmetic<U>::value, UnifiedBooleanExpr<T>>::type
expr_comparison(const U& l, const UnifiedVariable<T>& r, OpType comparison_op)
{
    auto arena = r.arena();
    auto scalar_id = arena->add_expression(ExprData<T>::constant(static_cast<T>(l)));
    auto bool_id = arena->add_boolean_expression(scalar_id, r.id(), comparison_op);
    return UnifiedBooleanExpr<T>(arena, bool_id);
}

// Comparison operators for UnifiedVariable
template<typename T>
UnifiedBooleanExpr<T> operator==(const UnifiedVariable<T>& l, const UnifiedVariable<T>& r)
{
    l.ensure_same_arena(r);
    return expr_comparison(l, r, OpType::Equal);
}

template<typename T>
UnifiedBooleanExpr<T> operator!=(const UnifiedVariable<T>& l, const UnifiedVariable<T>& r)
{
    l.ensure_same_arena(r);
    return expr_comparison(l, r, OpType::NotEqual);
}

template<typename T>
UnifiedBooleanExpr<T> operator<(const UnifiedVariable<T>& l, const UnifiedVariable<T>& r)
{
    l.ensure_same_arena(r);
    return expr_comparison(l, r, OpType::Less);
}

template<typename T>
UnifiedBooleanExpr<T> operator<=(const UnifiedVariable<T>& l, const UnifiedVariable<T>& r)
{
    l.ensure_same_arena(r);
    return expr_comparison(l, r, OpType::LessEqual);
}

template<typename T>
UnifiedBooleanExpr<T> operator>(const UnifiedVariable<T>& l, const UnifiedVariable<T>& r)
{
    l.ensure_same_arena(r);
    return expr_comparison(l, r, OpType::Greater);
}

template<typename T>
UnifiedBooleanExpr<T> operator>=(const UnifiedVariable<T>& l, const UnifiedVariable<T>& r)
{
    l.ensure_same_arena(r);
    return expr_comparison(l, r, OpType::GreaterEqual);
}

// Comparison operators with scalars
template<typename T, typename U>
typename std::enable_if<std::is_arithmetic<U>::value, UnifiedBooleanExpr<T>>::type
operator==(const UnifiedVariable<T>& l, const U& r)
{
    return expr_comparison(l, r, OpType::Equal);
}

template<typename T, typename U>
typename std::enable_if<std::is_arithmetic<U>::value, UnifiedBooleanExpr<T>>::type
operator!=(const UnifiedVariable<T>& l, const U& r)
{
    return expr_comparison(l, r, OpType::NotEqual);
}

template<typename T, typename U>
typename std::enable_if<std::is_arithmetic<U>::value, UnifiedBooleanExpr<T>>::type
operator<(const UnifiedVariable<T>& l, const U& r)
{
    return expr_comparison(l, r, OpType::Less);
}

template<typename T, typename U>
typename std::enable_if<std::is_arithmetic<U>::value, UnifiedBooleanExpr<T>>::type
operator<=(const UnifiedVariable<T>& l, const U& r)
{
    return expr_comparison(l, r, OpType::LessEqual);
}

template<typename T, typename U>
typename std::enable_if<std::is_arithmetic<U>::value, UnifiedBooleanExpr<T>>::type
operator>(const UnifiedVariable<T>& l, const U& r)
{
    return expr_comparison(l, r, OpType::Greater);
}

template<typename T, typename U>
typename std::enable_if<std::is_arithmetic<U>::value, UnifiedBooleanExpr<T>>::type
operator>=(const UnifiedVariable<T>& l, const U& r)
{
    return expr_comparison(l, r, OpType::GreaterEqual);
}

// Scalar comparison operators (left side scalar)
template<typename T, typename U>
typename std::enable_if<std::is_arithmetic<U>::value, UnifiedBooleanExpr<T>>::type
operator==(const U& l, const UnifiedVariable<T>& r)
{
    return expr_comparison(l, r, OpType::Equal);
}

template<typename T, typename U>
typename std::enable_if<std::is_arithmetic<U>::value, UnifiedBooleanExpr<T>>::type
operator!=(const U& l, const UnifiedVariable<T>& r)
{
    return expr_comparison(l, r, OpType::NotEqual);
}

template<typename T, typename U>
typename std::enable_if<std::is_arithmetic<U>::value, UnifiedBooleanExpr<T>>::type
operator<(const U& l, const UnifiedVariable<T>& r)
{
    return expr_comparison(l, r, OpType::Less);
}

template<typename T, typename U>
typename std::enable_if<std::is_arithmetic<U>::value, UnifiedBooleanExpr<T>>::type
operator<=(const U& l, const UnifiedVariable<T>& r)
{
    return expr_comparison(l, r, OpType::LessEqual);
}

template<typename T, typename U>
typename std::enable_if<std::is_arithmetic<U>::value, UnifiedBooleanExpr<T>>::type
operator>(const U& l, const UnifiedVariable<T>& r)
{
    return expr_comparison(l, r, OpType::Greater);
}

template<typename T, typename U>
typename std::enable_if<std::is_arithmetic<U>::value, UnifiedBooleanExpr<T>>::type
operator>=(const U& l, const UnifiedVariable<T>& r)
{
    return expr_comparison(l, r, OpType::GreaterEqual);
}

//------------------------------------------------------------------------------
// CONDITION AND RELATED FUNCTIONS
//------------------------------------------------------------------------------

/**
 * @brief Conditional expression function
 *
 * Creates a conditional expression that selects between two values based on a boolean predicate.
 * This is equivalent to the condition() function in the original var.hpp implementation.
 *
 * @param predicate Boolean expression determining which branch to take
 * @param true_expr Expression returned when predicate is true
 * @param false_expr Expression returned when predicate is false
 * @return UnifiedVariable representing the conditional expression
 */
template<typename T>
UnifiedVariable<T> condition(const UnifiedBooleanExpr<T>& predicate, const UnifiedVariable<T>& true_expr, const UnifiedVariable<T>& false_expr)
{
    true_expr.ensure_same_arena(false_expr);
    
    // Update predicate to get current value for initial evaluation
    bool pred_value = predicate.value();
    T initial_value = pred_value ? true_expr.value() : false_expr.value();
    
    auto result_id = true_expr.arena()->add_expression(
        ExprData<T>::conditional_op(initial_value, predicate.id(), true_expr.id(), false_expr.id()));
    return UnifiedVariable<T>(true_expr.arena(), result_id);
}

/**
 * @brief Conditional expression with scalar branches
 */
template<typename T, typename U, typename V>
typename std::enable_if<std::is_arithmetic<U>::value && std::is_arithmetic<V>::value, UnifiedVariable<T>>::type
condition(const UnifiedBooleanExpr<T>& predicate, const U& true_val, const V& false_val)
{
    auto arena = predicate.arena();
    auto true_id = arena->add_expression(ExprData<T>::constant(static_cast<T>(true_val)));
    auto false_id = arena->add_expression(ExprData<T>::constant(static_cast<T>(false_val)));
    
    bool pred_value = predicate.value();
    T initial_value = pred_value ? static_cast<T>(true_val) : static_cast<T>(false_val);
    
    auto result_id = arena->add_expression(
        ExprData<T>::conditional_op(initial_value, predicate.id(), true_id, false_id));
    return UnifiedVariable<T>(arena, result_id);
}

/**
 * @brief Conditional expression with mixed variable/scalar branches
 */
template<typename T, typename U>
typename std::enable_if<std::is_arithmetic<U>::value, UnifiedVariable<T>>::type
condition(const UnifiedBooleanExpr<T>& predicate, const UnifiedVariable<T>& true_expr, const U& false_val)
{
    auto false_id = true_expr.arena()->add_expression(ExprData<T>::constant(static_cast<T>(false_val)));
    
    bool pred_value = predicate.value();
    T initial_value = pred_value ? true_expr.value() : static_cast<T>(false_val);
    
    auto result_id = true_expr.arena()->add_expression(
        ExprData<T>::conditional_op(initial_value, predicate.id(), true_expr.id(), false_id));
    return UnifiedVariable<T>(true_expr.arena(), result_id);
}

template<typename T, typename U>
typename std::enable_if<std::is_arithmetic<U>::value, UnifiedVariable<T>>::type
condition(const UnifiedBooleanExpr<T>& predicate, const U& true_val, const UnifiedVariable<T>& false_expr)
{
    auto true_id = false_expr.arena()->add_expression(ExprData<T>::constant(static_cast<T>(true_val)));
    
    bool pred_value = predicate.value();
    T initial_value = pred_value ? static_cast<T>(true_val) : false_expr.value();
    
    auto result_id = false_expr.arena()->add_expression(
        ExprData<T>::conditional_op(initial_value, predicate.id(), true_id, false_expr.id()));
    return UnifiedVariable<T>(false_expr.arena(), result_id);
}

/**
 * @brief Minimum of two expressions
 */
template<typename T>
UnifiedVariable<T> min(const UnifiedVariable<T>& x, const UnifiedVariable<T>& y)
{
    return condition(x < y, x, y);
}

/**
 * @brief Maximum of two expressions
 */
template<typename T>
UnifiedVariable<T> max(const UnifiedVariable<T>& x, const UnifiedVariable<T>& y)
{
    return condition(x > y, x, y);
}

/**
 * @brief Sign function returning -1, 0, or 1
 */
template<typename T>
UnifiedVariable<T> sgn(const UnifiedVariable<T>& x)
{
    auto zero_id = x.arena()->add_expression(ExprData<T>::constant(T{0}));
    auto neg_one_id = x.arena()->add_expression(ExprData<T>::constant(T{-1}));
    auto pos_one_id = x.arena()->add_expression(ExprData<T>::constant(T{1}));
    
    UnifiedVariable<T> zero_var(x.arena(), zero_id);
    UnifiedVariable<T> neg_one_var(x.arena(), neg_one_id);
    UnifiedVariable<T> pos_one_var(x.arena(), pos_one_id);
    
    return condition(x < T{0}, neg_one_var, condition(x > T{0}, pos_one_var, zero_var));
}

/**
 * @brief Compute partial derivatives of a function with respect to given variables
 *
 * This function replaces the original autodiff::derivatives() function.
 * It performs reverse-mode automatic differentiation using the unified expression system.
 *
 * @param y The dependent variable (function output)
 * @param wrt_vars The independent variables wrapped in wrt()
 * @return Array of partial derivatives
 *
 * Example:
 *   auto grads = derivatives(f, wrt(x, y, z));
 *   // grads[0] = ∂f/∂x, grads[1] = ∂f/∂y, grads[2] = ∂f/∂z
 */
template<typename T, typename... Vars>
std::array<T, sizeof...(Vars)> derivatives(const UnifiedVariable<T>& y, const Wrt<Vars...>& wrt_vars)
{
    constexpr auto N = sizeof...(Vars);
    std::array<T, N> gradients;

    // Perform backward propagation
    y.arena()->clear_gradients();
    y.arena()->propagate(y.id(), T{1});

    // Extract gradients directly from arena's gradient workspace
    size_t index = 0;
    auto extract_helper = [&](const auto& var) {
        gradients[index++] = y.arena()->gradient(var.id());
    };

    apply_to_tuple(wrt_vars.args, extract_helper);

    return gradients;
}

// Helper function to apply a function to each element of a tuple (C++14 compatible)
template<typename Tuple, typename F, size_t... Is>
void apply_to_tuple_impl(const Tuple& t, F&& f, std::index_sequence<Is...>)
{
    (void)std::initializer_list<int>{(f(std::get<Is>(t)), 0)...};
}

template<typename Tuple, typename F>
void apply_to_tuple(const Tuple& t, F&& f)
{
    apply_to_tuple_impl(t, std::forward<F>(f), std::make_index_sequence<std::tuple_size<Tuple>::value>{});
}

// Output stream operator
template<typename T>
std::ostream& operator<<(std::ostream& out, const UnifiedVariable<T>& var)
{
    out << var.value();
    return out;
}

// Type aliases for convenience
using unifiedvar = UnifiedVariable<double>;
using unifiedvarf = UnifiedVariable<float>;

/**
 * @brief Summary of Key Design Differences from Original Implementation
 *
 * MEMORY LAYOUT:
 * =============
 * Original: Tree of shared_ptr<Expr<T>> nodes scattered in memory
 * Unified:  Flat vector<ExprData<T>> with index-based references
 *
 * POLYMORPHISM:
 * ============
 * Original: Virtual function calls for propagate(), update(), etc.
 * Unified:  Switch statements on ExprType/OpType enums
 *
 * EXPRESSION CREATION:
 * ===================
 * Original: std::make_shared<SpecificExpr<T>>(args...)
 * Unified:  arena->add_expression(ExprData<T>::operation_type(args...))
 *
 * CHILD REFERENCES:
 * ================
 * Original: ExprPtr<T> members (shared_ptr)
 * Unified:  ExprId indices into arena
 *
 * PERFORMANCE BENEFITS:
 * ====================
 * 1. Cache Locality: Sequential memory access during traversals
 * 2. Reduced Allocations: Single arena allocation vs many small allocations
 * 3. Faster Dispatch: Switch statements vs virtual function calls
 * 4. Memory Efficiency: No shared_ptr overhead, smaller expression nodes
 *
 * COMPATIBILITY:
 * =============
 * The UnifiedVariable<T> class provides the same API as the original Variable<T>,
 * making migration straightforward. The main difference is the need to create
 * and manage an ExpressionArena<T> instance.
 *
 * EXAMPLE MIGRATION:
 * =================
 * Original:
 *   var x = 2.0, y = 3.0;
 *   auto z = sin(x) + x * y;
 *   auto dz = derivatives(z, wrt(x, y));
 *
 * Unified:
 *   auto arena = std::make_shared<ExpressionArena<double>>();
 *   UnifiedVariable x(arena, 2.0), y(arena, 3.0);
 *   auto z = sin(x) + x * y;
 *   auto dz = derivatives(z, wrt(x, y));
 */

// ===== SYNTAX SUGAR FACTORY FUNCTIONS =====

/**
 * @brief Create a variable using the current thread-local arena
 *
 * Usage:
 *   ArenaScope<double> scope;
 *   auto x = make_var(2.0);
 *   auto y = make_var(3.0);
 */
template<typename T>
UnifiedVariable<T> make_var(const T& value)
{
    return UnifiedVariable<T>(ArenaManager<T>::get_current_arena(), value);
}

/**
 * @brief Create a constant using the current thread-local arena
 */
template<typename T>
UnifiedVariable<T> make_const(const T& value)
{
    auto arena = ArenaManager<T>::get_current_arena();
    auto id = arena->add_expression(ExprData<T>::constant(value));
    return UnifiedVariable<T>(arena, id);
}

/**
 * @brief Convenient alias for double variables
 */
using Var = UnifiedVariable<double>;

/**
 * @brief Create a double variable with automatic arena
 */
inline Var var(double value)
{
    return make_var<double>(value);
}

/**
 * @brief Create a double constant with automatic arena
 */
inline Var constant(double value)
{
    return make_const<double>(value);
}


// Type aliases for common use cases
using ArenaScope_d = ArenaScope<double>;
using VariablePool_d = VariablePool<double>;

} // namespace unified
} // namespace reverse
} // namespace autodiff
