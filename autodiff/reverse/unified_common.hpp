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



}
}
}