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
#include <unordered_map> // for per-root topo cache
#include <mutex>         // for thread-safe cache access

#include <unordered_set> // for parallel-safe per-build visited sets


// autodiff includes
#include <autodiff/common/meta.hpp>
#include <autodiff/common/numbertraits.hpp>
#include <autodiff/reverse/unified_common.hpp>


///////////////////////////
// syntax assistants
///////////////////////////


namespace autodiff {
namespace reverse {
namespace unified {


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


}
}
}