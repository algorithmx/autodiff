//                  _  _
//  _   _|_ _  _|o_|__|_
// (_||_||_(_)(_|| |  |
//
// Tests for TopoOrderingEngine (unified reverse autodiff)

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <autodiff/reverse/unified_expr.hpp>
#include <tests/utils/catch.hpp>

#include <unordered_map>
#include <unordered_set>
#include <thread>
#include <future>

using namespace autodiff::reverse::unified;

TEST_CASE("TopoOrderingEngine: basic post-order correctness", "[reverse][unified][topo-engine]")
{
    auto arena = std::make_shared<ExpressionArena<double>>();

    // Build a small DAG: x -> sin(x) -> cos(sin(x)) -> exp(cos(sin(x)))
    UnifiedVariable<double> x(arena, 1.0);
    auto y1 = sin(x);
    auto y2 = cos(y1);
    auto y3 = exp(y2);

    // Create an engine bound to this arena; version_fn can return a constant for testing
    TopoOrderingEngine<double> engine(
        [arena]() -> size_t { return arena->size(); },
        [arena](ExprId id) -> const ExprData<double>& { return (*arena)[id]; },
        []() -> size_t { return 0; }, // legacy version not needed for functional correctness here
        8 // cache capacity
    );
    engine.on_reserve(32);
    engine.on_append(arena->size(), 0);

    auto order = engine.topological_order(y3.id());

    // Positions map for easy assertions
    std::unordered_map<ExprId, size_t> pos;
    for (size_t i = 0; i < order.size(); ++i) pos[order[i]] = i;

    REQUIRE(pos.count(x.id()));
    REQUIRE(pos.count(y1.id()));
    REQUIRE(pos.count(y2.id()));
    REQUIRE(pos.count(y3.id()));

    CHECK(pos[x.id()] < pos[y1.id()]);
    CHECK(pos[y1.id()] < pos[y2.id()]);
    CHECK(pos[y2.id()] < pos[y3.id()]);

    CHECK(order.back() == y3.id());
}

TEST_CASE("TopoOrderingEngine: legacy cached_topological_order reference stability", "[reverse][unified][topo-engine]")
{
    auto arena = std::make_shared<ExpressionArena<double>>();

    UnifiedVariable<double> x(arena, 2.0);
    auto y = sin(x);
    auto z = cos(y);

    TopoOrderingEngine<double> engine(
        [arena]() -> size_t { return arena->size(); },
        [arena](ExprId id) -> const ExprData<double>& { return (*arena)[id]; },
        []() -> size_t { return 0; },
        8
    );
    engine.on_append(arena->size(), 0);

    // First call populates cache; second returns same reference
    const auto& ref1 = engine.cached_topological_order(z.id());
    const auto& ref2 = engine.cached_topological_order(z.id());
    CHECK(&ref1 == &ref2);
    CHECK(ref1.back() == z.id());
}

TEST_CASE("TopoOrderingEngine: parallel-safe per-root cache builds concurrently", "[reverse][unified][topo-engine]")
{
    auto arena = std::make_shared<ExpressionArena<double>>();

    // Two roots; share one leaf to create overlap
    UnifiedVariable<double> x(arena, 1.0);
    UnifiedVariable<double> y(arena, 3.0);
    auto r1 = exp(sin(x) + cos(y));
    auto r2 = log(cos(x) + sin(y));

    TopoOrderingEngine<double> engine(
        [arena]() -> size_t { return arena->size(); },
        [arena](ExprId id) -> const ExprData<double>& { return (*arena)[id]; },
        []() -> size_t { return 0; },
        8
    );
    engine.on_append(arena->size(), 0);

    // Build in parallel
    auto fut1 = std::async(std::launch::async, [&](){ return &engine.cached_topological_order_parallel_safe(r1.id()); });
    auto fut2 = std::async(std::launch::async, [&](){ return &engine.cached_topological_order_parallel_safe(r2.id()); });

    const auto* ord1 = fut1.get();
    const auto* ord2 = fut2.get();

    REQUIRE(ord1 != nullptr);
    REQUIRE(ord2 != nullptr);
    CHECK(!ord1->empty());
    CHECK(!ord2->empty());
    CHECK(ord1->back() == r1.id());
    CHECK(ord2->back() == r2.id());

    // Repeated calls should return the same references (per-root cache hit)
    const auto& ord1_again = engine.cached_topological_order_parallel_safe(r1.id());
    const auto& ord2_again = engine.cached_topological_order_parallel_safe(r2.id());
    CHECK(ord1 == &ord1_again);
    CHECK(ord2 == &ord2_again);
}

TEST_CASE("TopoOrderingEngine: per-root LRU eviction (capacity=1)", "[reverse][unified][topo-engine]")
{
    auto arena = std::make_shared<ExpressionArena<double>>();

    UnifiedVariable<double> x(arena, 1.0);
    UnifiedVariable<double> y(arena, 2.0);
    auto r1 = sin(x) + y;
    auto r2 = cos(y) + x;

    // Capacity 1 to force eviction
    TopoOrderingEngine<double> engine(
        [arena]() -> size_t { return arena->size(); },
        [arena](ExprId id) -> const ExprData<double>& { return (*arena)[id]; },
        []() -> size_t { return 0; },
        1
    );
    engine.on_append(arena->size(), 0);

    const auto& a1 = engine.cached_topological_order_parallel_safe(r1.id());
    // Using a different root should evict the previous one (capacity=1)
    const auto& b1 = engine.cached_topological_order_parallel_safe(r2.id());
    (void)b1; // silence unused warning

    // Request r1 again; the returned reference should likely differ (rebuilt and reinserted)
    const auto& a2 = engine.cached_topological_order_parallel_safe(r1.id());

    // Not strictly guaranteed by standard containers, but in our implementation it should
    // be a different vector instance because the old entry was evicted.
    CHECK(&a1 != &a2); // TODO FAILED
    CHECK(a2.back() == r1.id());
}


TEST_CASE("TopoOrderingEngine: deep linear chain (stack safety and order)", "[reverse][unified][topo-engine]")
{
    auto arena = std::make_shared<ExpressionArena<double>>();

    // Build a deep chain: x -> y1 -> y2 -> ... -> yN
    UnifiedVariable<double> x(arena, 1.0);
    const int N = 2000; // deep enough to expose recursion issues if any
    UnifiedVariable<double> curr = x;
    std::vector<ExprId> ids; ids.reserve(N+1);
    ids.push_back(x.id());
    for (int i = 0; i < N; ++i) {
        // Alternate ops to diversify
        if (i % 4 == 0) curr = sin(curr);
        else if (i % 4 == 1) curr = cos(curr);
        else if (i % 4 == 2) curr = exp(curr);
        else curr = log(curr);
        ids.push_back(curr.id());
    }

    TopoOrderingEngine<double> engine(
        [arena]() -> size_t { return arena->size(); },
        [arena](ExprId id) -> const ExprData<double>& { return (*arena)[id]; },
        []() -> size_t { return 0; },
        8
    );
    engine.on_append(arena->size(), 0);

    const auto& order = engine.cached_topological_order(ids.back());
    REQUIRE(order.size() == ids.size());

    // In a chain, order should match ids (post-order: leaf .. root)
    for (size_t i = 0; i < ids.size(); ++i) {
        CHECK(order[i] == ids[i]);
    }
    CHECK(order.back() == ids.back());
}

TEST_CASE("TopoOrderingEngine: diamond/DAG overlap no duplicates and post-order", "[reverse][unified][topo-engine]")
{
    auto arena = std::make_shared<ExpressionArena<double>>();

    // Diamond shape with shared leaf: x -> {sin(x), cos(x)} -> sum
    UnifiedVariable<double> x(arena, 1.5);
    auto sin_x = sin(x);
    auto cos_x = cos(x);
    auto sum = sin_x + cos_x;

    TopoOrderingEngine<double> engine(
        [arena]() -> size_t { return arena->size(); },
        [arena](ExprId id) -> const ExprData<double>& { return (*arena)[id]; },
        []() -> size_t { return 0; },
        8
    );
    engine.on_append(arena->size(), 0);

    const auto& order = engine.cached_topological_order(sum.id());

    std::unordered_map<ExprId, size_t> pos;
    for (size_t i = 0; i < order.size(); ++i) pos[order[i]] = i;

    // No duplicates
    std::unordered_set<ExprId> uniq(order.begin(), order.end());
    CHECK(uniq.size() == order.size());

    // Post-order constraints
    REQUIRE(pos.count(x.id()));
    REQUIRE(pos.count(sin_x.id()));
    REQUIRE(pos.count(cos_x.id()));
    REQUIRE(pos.count(sum.id()));

    CHECK(pos[x.id()] < pos[sin_x.id()]);
    CHECK(pos[x.id()] < pos[cos_x.id()]);
    CHECK(pos[sin_x.id()] < pos[sum.id()]);
    CHECK(pos[cos_x.id()] < pos[sum.id()]);
    CHECK(order.back() == sum.id());
}

TEST_CASE("TopoOrderingEngine: overlapping roots per-root cache hit", "[reverse][unified][topo-engine]")
{
    auto arena = std::make_shared<ExpressionArena<double>>();

    // Shared subgraph for two roots
    UnifiedVariable<double> x(arena, 2.0);
    UnifiedVariable<double> y(arena, 3.0);
    auto shared = x + y;         // shared subexpression
    auto r1 = sin(shared) + exp(x);
    auto r2 = cos(shared) + log(y);

    TopoOrderingEngine<double> engine(
        [arena]() -> size_t { return arena->size(); },
        [arena](ExprId id) -> const ExprData<double>& { return (*arena)[id]; },
        []() -> size_t { return 0; },
        16
    );
    engine.on_append(arena->size(), 0);

    const auto& o1 = engine.cached_topological_order_parallel_safe(r1.id());
    const auto& o2 = engine.cached_topological_order_parallel_safe(r2.id());

    // Second calls should return same references (cache hit)
    const auto& o1_again = engine.cached_topological_order_parallel_safe(r1.id());
    const auto& o2_again = engine.cached_topological_order_parallel_safe(r2.id());
    CHECK(&o1 == &o1_again);
    CHECK(&o2 == &o2_again);

    // Both orders must end with their respective roots
    CHECK(o1.back() == r1.id());
    CHECK(o2.back() == r2.id());
}
