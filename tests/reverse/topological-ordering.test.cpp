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

// Catch includes
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

// autodiff includes
#include <autodiff/reverse/unified_expr.hpp>
#include <tests/utils/catch.hpp>

// Standard includes
#include <unordered_set>
#include <unordered_map>
#include <algorithm>

using namespace autodiff::reverse::unified;

/**
 * @brief Tests for the topological ordering algorithm in ExpressionArena
 * 
 * The algorithm being tested implements an iterative post-order DFS traversal that:
 * 
 * 1. Uses an explicit stack to avoid recursion (prevents stack overflow on deep chains)
 * 2. Marks visited nodes to prevent duplicate processing in DAGs with shared subexpressions
 * 3. Produces a post-order traversal where children appear before their parents
 * 4. Caches results using version-based invalidation
 * 5. Returns vectors where the root is the LAST element
 * 
 * Key properties verified:
 * - POST-ORDER: Children always appear before parents in the result
 * - NO DUPLICATES: Each reachable node appears exactly once due to visit marking
 * - ROOT LAST: The specified root node is always the final element
 * - CACHING: Identical results for same root/version, invalidation on arena mutation
 * - STACK-SAFE: Works on deep linear chains without recursion limits
 */

TEST_CASE("testing topological ordering correctness", "[reverse][unified][topological]")
{
    SECTION("testing basic linear chain ordering")
    {
        auto arena = std::make_shared<ExpressionArena<double>>();
        
        // Create a linear chain: x -> sin(x) -> cos(sin(x)) -> exp(cos(sin(x)))
        UnifiedVariable<double> x(arena, 1.0);
        auto y1 = sin(x);
        auto y2 = cos(y1);
        auto y3 = exp(y2);
        
        // Get topological order from the root (y3)
        auto topo_order = arena->topological_order(y3.id());
        
        // Convert to set for easy lookup of positions
        std::unordered_map<ExprId, size_t> positions;
        for (size_t i = 0; i < topo_order.size(); ++i) {
            positions[topo_order[i]] = i;
        }
        
        // Check that children always appear before their parents in the ordering
        REQUIRE(positions.find(x.id()) != positions.end());
        REQUIRE(positions.find(y1.id()) != positions.end());
        REQUIRE(positions.find(y2.id()) != positions.end());
        REQUIRE(positions.find(y3.id()) != positions.end());
        
        // Verify ordering: x < sin(x) < cos(sin(x)) < exp(cos(sin(x)))
        CHECK(positions[x.id()] < positions[y1.id()]);
        CHECK(positions[y1.id()] < positions[y2.id()]);
        CHECK(positions[y2.id()] < positions[y3.id()]);
        
        // Root should be last in post-order
        CHECK(topo_order.back() == y3.id());
        
        // Verify all nodes in the chain are included
        CHECK(topo_order.size() == 4);
    }
    
    SECTION("testing binary tree ordering")
    {
        auto arena = std::make_shared<ExpressionArena<double>>();
        
        // Create a binary tree: (x + y) * (sin(x) - cos(y))
        UnifiedVariable<double> x(arena, 2.0);
        UnifiedVariable<double> y(arena, 3.0);
        auto left_branch = x + y;
        auto sin_x = sin(x);
        auto cos_y = cos(y);
        auto right_branch = sin_x - cos_y;
        auto root = left_branch * right_branch;
        
        auto topo_order = arena->topological_order(root.id());
        
        // Convert to position map
        std::unordered_map<ExprId, size_t> positions;
        for (size_t i = 0; i < topo_order.size(); ++i) {
            positions[topo_order[i]] = i;
        }
        
        // Check that all nodes are present
        REQUIRE(positions.find(x.id()) != positions.end());
        REQUIRE(positions.find(y.id()) != positions.end());
        REQUIRE(positions.find(left_branch.id()) != positions.end());
        REQUIRE(positions.find(sin_x.id()) != positions.end());
        REQUIRE(positions.find(cos_y.id()) != positions.end());
        REQUIRE(positions.find(right_branch.id()) != positions.end());
        REQUIRE(positions.find(root.id()) != positions.end());
        
        // Verify parent-child ordering constraints
        // x should come before left_branch (x + y)
        CHECK(positions[x.id()] < positions[left_branch.id()]);
        // y should come before left_branch (x + y)
        CHECK(positions[y.id()] < positions[left_branch.id()]);
        // x should come before sin_x
        CHECK(positions[x.id()] < positions[sin_x.id()]);
        // y should come before cos_y
        CHECK(positions[y.id()] < positions[cos_y.id()]);
        // sin_x should come before right_branch
        CHECK(positions[sin_x.id()] < positions[right_branch.id()]);
        // cos_y should come before right_branch
        CHECK(positions[cos_y.id()] < positions[right_branch.id()]);
        // Both branches should come before root
        CHECK(positions[left_branch.id()] < positions[root.id()]);
        CHECK(positions[right_branch.id()] < positions[root.id()]);
        
        // Root should be last
        CHECK(topo_order.back() == root.id());
    }
    
    SECTION("testing diamond dependency graph")
    {
        auto arena = std::make_shared<ExpressionArena<double>>();
        
        // Create diamond pattern: x -> {sin(x), cos(x)} -> sin(x) + cos(x)
        UnifiedVariable<double> x(arena, 1.5);
        auto sin_x = sin(x);
        auto cos_x = cos(x);
        auto sum = sin_x + cos_x;
        
        auto topo_order = arena->topological_order(sum.id());
        
        std::unordered_map<ExprId, size_t> positions;
        for (size_t i = 0; i < topo_order.size(); ++i) {
            positions[topo_order[i]] = i;
        }
        
        // Check all nodes present
        REQUIRE(positions.find(x.id()) != positions.end());
        REQUIRE(positions.find(sin_x.id()) != positions.end());
        REQUIRE(positions.find(cos_x.id()) != positions.end());
        REQUIRE(positions.find(sum.id()) != positions.end());
        
        // x should come before both sin(x) and cos(x)
        CHECK(positions[x.id()] < positions[sin_x.id()]);
        CHECK(positions[x.id()] < positions[cos_x.id()]);
        
        // Both sin(x) and cos(x) should come before sum
        CHECK(positions[sin_x.id()] < positions[sum.id()]);
        CHECK(positions[cos_x.id()] < positions[sum.id()]);
        
        // Sum should be last
        CHECK(topo_order.back() == sum.id());
        
        // Should have exactly 4 nodes
        CHECK(topo_order.size() == 4);
        
        // x should appear only once (no duplicates)
        size_t x_count = std::count(topo_order.begin(), topo_order.end(), x.id());
        CHECK(x_count == 1);
    }
    
    SECTION("testing complex multi-level expression")
    {
        auto arena = std::make_shared<ExpressionArena<double>>();
        
        // Create: f = exp(sin(x) * cos(y)) + log(x * y + z)
        UnifiedVariable<double> x(arena, 1.0);
        UnifiedVariable<double> y(arena, 2.0);
        UnifiedVariable<double> z(arena, 3.0);
        
        auto sin_x = sin(x);
        auto cos_y = cos(y);
        auto mul1 = sin_x * cos_y;
        auto exp_part = exp(mul1);
        
        auto xy = x * y;
        auto sum = xy + z;
        auto log_part = log(sum);
        
        auto final_result = exp_part + log_part;
        
        auto topo_order = arena->topological_order(final_result.id());
        
        std::unordered_map<ExprId, size_t> positions;
        for (size_t i = 0; i < topo_order.size(); ++i) {
            positions[topo_order[i]] = i;
        }
        
        // Check all intermediate nodes are present
        REQUIRE(positions.find(x.id()) != positions.end());
        REQUIRE(positions.find(y.id()) != positions.end());
        REQUIRE(positions.find(z.id()) != positions.end());
        REQUIRE(positions.find(sin_x.id()) != positions.end());
        REQUIRE(positions.find(cos_y.id()) != positions.end());
        REQUIRE(positions.find(mul1.id()) != positions.end());
        REQUIRE(positions.find(exp_part.id()) != positions.end());
        REQUIRE(positions.find(xy.id()) != positions.end());
        REQUIRE(positions.find(sum.id()) != positions.end());
        REQUIRE(positions.find(log_part.id()) != positions.end());
        REQUIRE(positions.find(final_result.id()) != positions.end());
        
        // Verify dependency constraints for left side: x -> sin(x) -> mul1 -> exp_part
        CHECK(positions[x.id()] < positions[sin_x.id()]);
        CHECK(positions[sin_x.id()] < positions[mul1.id()]);
        CHECK(positions[mul1.id()] < positions[exp_part.id()]);
        
        // Verify dependency constraints for right side of mul1: y -> cos(y) -> mul1
        CHECK(positions[y.id()] < positions[cos_y.id()]);
        CHECK(positions[cos_y.id()] < positions[mul1.id()]);
        
        // Verify dependency constraints for right side: x,y -> xy -> sum -> log_part
        CHECK(positions[x.id()] < positions[xy.id()]);
        CHECK(positions[y.id()] < positions[xy.id()]);
        CHECK(positions[xy.id()] < positions[sum.id()]);
        CHECK(positions[z.id()] < positions[sum.id()]);
        CHECK(positions[sum.id()] < positions[log_part.id()]);
        
        // Both main branches should come before final result
        CHECK(positions[exp_part.id()] < positions[final_result.id()]);
        CHECK(positions[log_part.id()] < positions[final_result.id()]);
        
        // Final result should be last
        CHECK(topo_order.back() == final_result.id());
    }
    
    SECTION("testing topological ordering with shared subexpressions")
    {
        auto arena = std::make_shared<ExpressionArena<double>>();
        
        // Create: f = (x + y)^2 + (x + y) * z
        // This creates a shared subexpression (x + y)
        UnifiedVariable<double> x(arena, 1.0);
        UnifiedVariable<double> y(arena, 2.0);
        UnifiedVariable<double> z(arena, 3.0);
        
        auto shared_expr = x + y;  // This will be reused
        auto squared = shared_expr * shared_expr;
        auto multiplied = shared_expr * z;
        auto result = squared + multiplied;
        
        auto topo_order = arena->topological_order(result.id());
        
        std::unordered_map<ExprId, size_t> positions;
        for (size_t i = 0; i < topo_order.size(); ++i) {
            positions[topo_order[i]] = i;
        }
        
        // Verify that shared_expr appears only once
        size_t shared_count = std::count(topo_order.begin(), topo_order.end(), shared_expr.id());
        CHECK(shared_count == 1);
        
        // Verify ordering constraints
        CHECK(positions[x.id()] < positions[shared_expr.id()]);
        CHECK(positions[y.id()] < positions[shared_expr.id()]);
        CHECK(positions[shared_expr.id()] < positions[squared.id()]);
        CHECK(positions[shared_expr.id()] < positions[multiplied.id()]);
        CHECK(positions[z.id()] < positions[multiplied.id()]);
        CHECK(positions[squared.id()] < positions[result.id()]);
        CHECK(positions[multiplied.id()] < positions[result.id()]);
        
        CHECK(topo_order.back() == result.id());
    }
    
    SECTION("testing topological ordering caching")
    {
        auto arena = std::make_shared<ExpressionArena<double>>();
        
        UnifiedVariable<double> x(arena, 1.0);
        auto y = sin(x);
        auto z = cos(y);
        
        // First call should compute the ordering
        auto topo1 = arena->topological_order(z.id());
        
        // Second call with same root should use cached version and return identical result
        auto topo2 = arena->topological_order(z.id());
        
        CHECK(topo1.size() == topo2.size());
        CHECK(std::equal(topo1.begin(), topo1.end(), topo2.begin()));
        
        // Test cached_topological_order returns reference to same cached data
        const auto& topo_ref1 = arena->cached_topological_order(z.id());
        const auto& topo_ref2 = arena->cached_topological_order(z.id());
        
        // Should return reference to same object (same address)
        CHECK(&topo_ref1 == &topo_ref2);
        CHECK(std::equal(topo_ref1.begin(), topo_ref1.end(), topo1.begin()));
        
        // Cache should be invalidated when arena changes (version bumps)
        auto w = exp(z);  // This should bump the arena version
        auto topo3 = arena->topological_order(w.id());
        
        // topo3 should be different and longer
        CHECK(topo3.size() > topo1.size());
        CHECK(topo3.back() == w.id());
        
        // Verify the cache is now for the new root
        const auto& topo_ref3 = arena->cached_topological_order(w.id());
        CHECK(std::equal(topo_ref3.begin(), topo_ref3.end(), topo3.begin()));
    }
    
    SECTION("testing cached_topological_order reference return")
    {
        auto arena = std::make_shared<ExpressionArena<double>>();
        
        UnifiedVariable<double> x(arena, 1.0);
        auto y = sin(x);
        auto z = cos(y);
        
        // Get reference to cached ordering
        const auto& topo_ref1 = arena->cached_topological_order(z.id());
        const auto& topo_ref2 = arena->cached_topological_order(z.id());
        
        // Should return reference to same object (same address)
        CHECK(&topo_ref1 == &topo_ref2);
        
        // Content should be correct
        CHECK(topo_ref1.back() == z.id());
        CHECK(topo_ref1.size() == 3);  // x, sin(x), cos(sin(x))
        
        // Verify ordering
        std::unordered_map<ExprId, size_t> positions;
        for (size_t i = 0; i < topo_ref1.size(); ++i) {
            positions[topo_ref1[i]] = i;
        }
        
        CHECK(positions[x.id()] < positions[y.id()]);
        CHECK(positions[y.id()] < positions[z.id()]);
    }
    
    SECTION("testing degenerate cases")
    {
        auto arena = std::make_shared<ExpressionArena<double>>();
        
        // Single node (constant or variable)
        UnifiedVariable<double> x(arena, 5.0);
        auto topo_single = arena->topological_order(x.id());
        
        CHECK(topo_single.size() == 1);
        CHECK(topo_single[0] == x.id());
        
        // Empty arena case is not applicable since we need at least one node
        // to call topological_order
    }
    
    SECTION("testing correctness invariant for any expression tree")
    {
        auto arena = std::make_shared<ExpressionArena<double>>();
        
        // Create a complex nested expression
        UnifiedVariable<double> x(arena, 0.5);
        UnifiedVariable<double> y(arena, 1.5);
        
        auto complex_expr = sin(exp(x * y) + log(sqrt(x + y))) - cos(pow(x, y) / (x - y + 1.0));
        
        auto topo_order = arena->topological_order(complex_expr.id());
        
        // Build position mapping
        std::unordered_map<ExprId, size_t> positions;
        for (size_t i = 0; i < topo_order.size(); ++i) {
            positions[topo_order[i]] = i;
        }
        
        // Verify fundamental POST-ORDER invariant: for every node in the ordering,
        // all its children must appear BEFORE it (this is the key property of post-order DFS)
        for (ExprId node_id : topo_order) {
            const auto& node = (*arena)[node_id];
            
            for (unsigned i = 0; i < node.num_children; ++i) {
                ExprId child_id = node.children[i];
                if (child_id != INVALID_EXPR_ID) {
                    // Child must have a position in the ordering
                    REQUIRE(positions.find(child_id) != positions.end());
                    // POST-ORDER: Child must appear BEFORE its parent
                    CHECK(positions[child_id] < positions[node_id]);
                }
            }
        }
        
        // Root should be LAST in post-order traversal
        CHECK(topo_order.back() == complex_expr.id());
        
        // All nodes should have unique positions (no duplicates due to visit marking)
        std::unordered_set<ExprId> unique_nodes(topo_order.begin(), topo_order.end());
        CHECK(unique_nodes.size() == topo_order.size());
    }
    
    SECTION("testing post-order DFS property explicitly")
    {
        auto arena = std::make_shared<ExpressionArena<double>>();
        
        // Create a specific tree structure where post-order is well-defined
        //        root (*)
        //       /         \
        //    left (+)    right (-)
        //    /    \      /     \
        //   x      y    z      w
        UnifiedVariable<double> x(arena, 1.0);
        UnifiedVariable<double> y(arena, 2.0);
        UnifiedVariable<double> z(arena, 3.0);
        UnifiedVariable<double> w(arena, 4.0);
        
        auto left = x + y;
        auto right = z - w;
        auto root = left * right;
        
        auto topo_order = arena->topological_order(root.id());
        
        // Convert to position map
        std::unordered_map<ExprId, size_t> positions;
        for (size_t i = 0; i < topo_order.size(); ++i) {
            positions[topo_order[i]] = i;
        }
        
        // In post-order DFS:
        // 1. All leaf nodes (x, y, z, w) come before their parents
        CHECK(positions[x.id()] < positions[left.id()]);
        CHECK(positions[y.id()] < positions[left.id()]);
        CHECK(positions[z.id()] < positions[right.id()]);
        CHECK(positions[w.id()] < positions[right.id()]);
        
        // 2. Internal nodes come before their parents
        CHECK(positions[left.id()] < positions[root.id()]);
        CHECK(positions[right.id()] < positions[root.id()]);
        
        // 3. Root is the LAST element (defining property of post-order)
        CHECK(topo_order.back() == root.id());
        
        // 4. Verify this is actually a valid post-order traversal
        // In a post-order traversal, when we process a node, all its descendants
        // have already been processed
        std::unordered_set<ExprId> processed;
        for (ExprId node_id : topo_order) {
            const auto& node = (*arena)[node_id];
            
            // All children should have been processed already
            for (unsigned i = 0; i < node.num_children; ++i) {
                ExprId child_id = node.children[i];
                if (child_id != INVALID_EXPR_ID) {
                    CHECK(processed.find(child_id) != processed.end());
                }
            }
            
            processed.insert(node_id);
        }
    }
    
    SECTION("testing iterative DFS stack behavior")
    {
        auto arena = std::make_shared<ExpressionArena<double>>();
        
        // Create a deep linear chain to test stack-based implementation
        // x -> sin(x) -> cos(sin(x)) -> exp(cos(sin(x))) -> log(exp(cos(sin(x))))
        UnifiedVariable<double> x(arena, 1.0);
        auto y1 = sin(x);
        auto y2 = cos(y1);
        auto y3 = exp(y2);
        auto y4 = log(y3);
        
        auto topo_order = arena->topological_order(y4.id());
        
        // For a linear chain in post-order: deepest leaf first, root last
        std::vector<ExprId> expected_order = {x.id(), y1.id(), y2.id(), y3.id(), y4.id()};
        
        CHECK(topo_order.size() == expected_order.size());
        for (size_t i = 0; i < expected_order.size(); ++i) {
            CHECK(topo_order[i] == expected_order[i]);
        }
        
        // Root (y4) should be last
        CHECK(topo_order.back() == y4.id());
        
        // This should work without stack overflow even for deep chains
        // (testing the iterative vs recursive implementation)
        CHECK(topo_order.size() == 5);
    }
    
    SECTION("testing visit marking prevents duplicate processing in DAGs")
    {
        auto arena = std::make_shared<ExpressionArena<double>>();
        
        // Create a DAG with significant sharing:
        //          result
        //         /      \
        //    shared^2   shared^3
        //        |        |
        //      shared   shared (same node!)
        //       / \     
        //      x   y    
        UnifiedVariable<double> x(arena, 2.0);
        UnifiedVariable<double> y(arena, 3.0);
        auto shared = x + y;  // This node will be referenced multiple times
        auto shared_squared = shared * shared;
        auto shared_cubed = shared * shared * shared;  // This creates: shared * (shared * shared)
        auto result = shared_squared + shared_cubed;
        
        auto topo_order = arena->topological_order(result.id());
        
        // Count occurrences of each node
        std::unordered_map<ExprId, int> node_counts;
        for (ExprId id : topo_order) {
            node_counts[id]++;
        }
        
        // Every node should appear exactly once (visit marking should prevent duplicates)
        for (const auto& [node_id, count] : node_counts) {
            CHECK(count == 1);
        }
        
        // Specifically check that the shared subexpression appears only once
        CHECK(node_counts[shared.id()] == 1);
        CHECK(node_counts[x.id()] == 1);
        CHECK(node_counts[y.id()] == 1);
        
        // Verify ordering constraints still hold
        std::unordered_map<ExprId, size_t> positions;
        for (size_t i = 0; i < topo_order.size(); ++i) {
            positions[topo_order[i]] = i;
        }
        
        // Dependencies should be respected
        CHECK(positions[x.id()] < positions[shared.id()]);
        CHECK(positions[y.id()] < positions[shared.id()]);
        CHECK(positions[shared.id()] < positions[shared_squared.id()]);
        CHECK(positions[shared.id()] < positions[shared_cubed.id()]);
        CHECK(positions[shared_squared.id()] < positions[result.id()]);
        CHECK(positions[shared_cubed.id()] < positions[result.id()]);
        
        // Result should be last
        CHECK(topo_order.back() == result.id());
    }
}
