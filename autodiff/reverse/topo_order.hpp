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

namespace autodiff {
namespace reverse {
namespace unified {



// Dedicated topological ordering engine to encapsulate visitation and caching.
// This improves readability, testability, and allows multiple policies (legacy vs. parallel-safe).
// It operates on a minimal accessor interface, decoupled from ExpressionArena internals.
template<typename T>
class TopoOrderingEngine
{
public:
    using NodeAccessor = std::function<const ExprData<T>&(ExprId)>;
    using SizeFn       = std::function<size_t()>;
    using VersionFn    = std::function<size_t()>; // legacy append-version

    explicit TopoOrderingEngine(SizeFn size_fn, NodeAccessor at_fn, VersionFn ver_fn, size_t cache_capacity = 64)
        : size_fn_(std::move(size_fn)), at_fn_(std::move(at_fn)), version_fn_(std::move(ver_fn)), topo_cache_capacity_(cache_capacity)
    {}

    // Reservations/notifications from the owner. These are optional optimizations.
    void on_reserve(size_t cap)
    {
        visit_gen_.reserve(cap);
        legacy_topo_cache_.reserve(cap);
    }

    void on_append(size_t new_size, size_t /*new_version*/)
    {
        if(visit_gen_.size() < new_size) visit_gen_.resize(new_size, 0u);
        // Appends do not invalidate per-root caches; structure_epoch_ remains unchanged.
    }

    // Legacy single-root cache (invalidated by append-version changes)
    std::vector<ExprId> topological_order(ExprId root_id) const
    {
        std::vector<ExprId> result;
        const size_t n = size_fn_();
        if(root_id == INVALID_EXPR_ID || n == 0) return result;

        // If cache matches current version and root, return a copy (legacy semantics)
        if(legacy_topo_built_version_ == version_fn_() && legacy_topo_root_ == root_id)
            return legacy_topo_cache_;

        // Generation-based visitation to avoid O(N) clears
        uint32_t gen = ++current_gen_;
        if(gen == 0) { // wrap-around: rare full reset
            std::fill(visit_gen_.begin(), visit_gen_.end(), 0u);
            current_gen_ = 1; gen = 1;
        }

        result.reserve(n);
        std::vector<std::pair<ExprId, bool>> stack; stack.reserve(256);
        auto push_if_valid = [&](ExprId cid){
            if(cid == INVALID_EXPR_ID) return;
            if(static_cast<size_t>(cid) >= n) return;
            if(visit_gen_[cid] == gen) return;
            visit_gen_[cid] = gen;
            stack.emplace_back(cid, false);
        };
        push_if_valid(root_id);

        while(!stack.empty()) {
            auto [id, expanded] = stack.back(); stack.pop_back();
            if(expanded) { result.push_back(id); continue; }
            stack.emplace_back(id, true);
            const ExprData<T>& node = at_fn_(id);
            for(unsigned ci = 0; ci < node.num_children && ci < node.children.size(); ++ci) push_if_valid(node.children[ci]);
        }

        legacy_topo_cache_ = result;
        legacy_topo_root_ = root_id;
        legacy_topo_built_version_ = version_fn_();
        return legacy_topo_cache_;
    }

    const std::vector<ExprId>& cached_topological_order(ExprId root_id) const
    {
        if(legacy_topo_built_version_ != version_fn_() || legacy_topo_root_ != root_id)
            (void)topological_order(root_id);
        return legacy_topo_cache_;
    }

    // Parallel-safe, per-root cached order. DFS builds use per-build local visited sets.
    const std::vector<ExprId>& cached_topological_order_parallel_safe(ExprId root_id) const
    {
        static const std::vector<ExprId> kEmpty;
        const size_t n = size_fn_();
        if(root_id == INVALID_EXPR_ID || n == 0) return kEmpty;

        // Fast path: cache hit
        {
            std::lock_guard<std::mutex> lock(mtx_);
            auto it = per_root_cache_.find(root_id);
            if(it != per_root_cache_.end() && it->second.epoch == structure_epoch_) {
                it->second.last_used = ++use_tick_;
                return *it->second.order;
            }
        }

        // Build without holding the lock
        std::vector<ExprId> order; order.reserve(n);
        std::vector<std::pair<ExprId,bool>> stack; stack.reserve(256);
        std::unordered_set<ExprId> visited; visited.reserve(256);
        auto push_local = [&](ExprId cid){
            if(cid == INVALID_EXPR_ID) return;
            if(static_cast<size_t>(cid) >= n) return;
            if(!visited.insert(cid).second) return;
            stack.emplace_back(cid, false);
        };
        push_local(root_id);
        while(!stack.empty()){
            auto [id, expanded] = stack.back(); stack.pop_back();
            if(expanded) { order.push_back(id); continue; }
            stack.emplace_back(id, true);
            const ExprData<T>& node = at_fn_(id);
            for(unsigned ci = 0; ci < node.num_children && ci < node.children.size(); ++ci) push_local(node.children[ci]);
        }

        // Publish with LRU eviction
        {
            std::lock_guard<std::mutex> lock(mtx_);
            auto& entry = per_root_cache_[root_id];
            entry.order = std::make_shared<std::vector<ExprId>>(std::move(order));
            entry.epoch = structure_epoch_;
            entry.last_used = ++use_tick_;
            if(per_root_cache_.size() > topo_cache_capacity_) {
                auto lru_it = per_root_cache_.begin();
                for(auto it = per_root_cache_.begin(); it != per_root_cache_.end(); ++it) {
                    if(it->first == root_id) continue;
                    if(it->second.last_used < lru_it->second.last_used) lru_it = it;
                }
                if(!per_root_cache_.empty() && lru_it != per_root_cache_.end()) per_root_cache_.erase(lru_it);
            }
            return *per_root_cache_[root_id].order;
        }
    }

private:
    // Access interface to owner
    SizeFn size_fn_;
    NodeAccessor at_fn_;
    VersionFn version_fn_;

    // Legacy single-root cache fields
    mutable std::vector<ExprId> legacy_topo_cache_;
    mutable ExprId legacy_topo_root_ = INVALID_EXPR_ID;
    mutable size_t legacy_topo_built_version_ = static_cast<size_t>(-1);

    // Generation-based visitation
    mutable std::vector<uint32_t> visit_gen_;
    mutable uint32_t current_gen_ = 1;

    // Per-root cache (parallel-safe)
    struct Entry { std::shared_ptr<std::vector<ExprId>> order; size_t epoch = 0; uint64_t last_used = 0; };
    mutable std::unordered_map<ExprId, Entry> per_root_cache_;
    mutable std::mutex mtx_;
    size_t topo_cache_capacity_ = 64;
    mutable uint64_t use_tick_ = 0;

    // Structural epoch (edges of existing nodes). Appends do not change this.
    size_t structure_epoch_ = 0;
};



} // namespace unified
} // namespace reverse
} // 