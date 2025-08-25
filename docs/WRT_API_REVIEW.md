# `wrt(...)` API Review

Date: 2025-08-25  
Module: `autodiff` (forward & reverse)  
Subject: Design assessment of the `wrt` descriptor ("with respect to")

## 1. Overview
`wrt` wraps a heterogeneous variadic pack (scalars, autodiff numbers, Eigen vector-like objects, repeated variables) into a tuple-based descriptor `Wrt<...>` used by algorithms (`derivative`, `derivatives`, `gradient`, `jacobian`, `hessian`, etc.). It encodes both the *set / order* of differentiation variables and (implicitly) some higher-order semantics.

## 2. Strengths
- Uniform descriptor across forward & reverse differentiation entry points.  
- Variadic + forwarding supports mixing scalars, vectors, slices, repetition.  
- Repetition provides a compact way to request mixed / higher-order partials.  
- Zero (or negligible) runtime overhead—constexpr tuple + inlining.  
- Static assertions catch order misuse & const mistakes early.  
- Integrates naturally with existing function templates without extra builders.

## 3. Pain Points / Ambiguities
| # | Issue | Impact |
|---|-------|--------|
|1|Implicit rule: single `wrt(x)` with higher-order dual auto-yields higher derivatives (last var absorbs remaining orders). Not clearly distinguished from repetition.|Conceptual confusion.|
|2|Repetition overloaded: means "higher order for same var" *and* ordering for mixed partial extraction.|Harder mental model.|
|3|Forwarding references risk dangling if temporaries (Eigen expressions) are passed (e.g. `wrt(x.segment<3>(0))`).|UB risk.|
|4|Duplicated `Wrt` definitions (forward vs reverse) → divergence risk.|Maintenance cost.|
|5|Const disallowed via static assert—surprising when users keep const vectors of duals.|Friction.|
|6|Static assert messages long, generic—lack contextual numeric details.|Debug friction.|
|7|Pre-C++20 style; no concepts / constrained templates for clarity.|Readability.|
|8|Manual `seed` / `unseed` (forward) without RAII guard—exception / early-return hazard if future changes add throws.|Safety.|
|9|Mixed derivative ordering entirely by argument order & repetition; no helper to introspect mapping.|User error risk.|
|10|Flattening vectors into gradient slots undocumented (index mapping).|Usability.|
|11|Symmetric name `wrt` hides that reverse mode path has different mechanics (no seeding).|Expectations misaligned.|

### 3.a Source Code Reference for Hidden Higher-Order Semantics
The implicit behavior in forward mode whereby `wrt(x)` (with a higher-order dual type) produces all higher-order derivatives without repetition stems from the fallback branch inside the seeding loop in `autodiff/forward/utils/derivative.hpp`:

```cpp
template<typename Var, typename... Vars, typename T>
AUTODIFF_DEVICE_FUNC auto seed(const Wrt<Var&, Vars&...>& wrt, T&& seedval)
{
  constexpr static auto N = Order<Var>;          // maximum derivative order provided by the dual type
  constexpr static auto size = 1 + sizeof...(Vars); // number of explicit variables passed to wrt(...)
  For<N>([&](auto i) constexpr {
    if constexpr (i.index < size)
      seed<i.index + 1>(std::get<i>(wrt.args), seedval);   // normal seeding: i-th derivative uses i-th variable
    else
      seed<i.index + 1>(std::get<size - 1>(wrt.args), seedval); // fallback: reuse LAST variable for remaining higher orders
  });
}
```

Effect:
- When only one variable is supplied (`size == 1`), the `else` branch handles every order beyond the first, repeatedly seeding the same variable for orders 2..N.
- Thus `wrt(x)` with `x: dual4th` implicitly behaves like `wrt(x, x, x, x)` for seeding purposes, yielding `f0, fx, fxx, fxxx, fxxxx` without explicit repetition.
- This is invoked transparently via `eval(f, at, wrt)` which always calls `seed(wrt);` before evaluation and `unseed(wrt);` after.

Documentation Gap Filled:
Add an explicit explanation (now captured here) that: "If fewer variables are passed in `wrt(...)` than the derivative order supported by their dual type, the last variable is automatically reused to fill the remaining higher-order derivative slots." This clarifies why `wrt(x)` suffices for higher orders and distinguishes it from intentional repetition for mixed partials (`wrt(x, x, y)`).

### 3.b Overloaded Meaning of Repetition (Detailed Explanation)
Repetition of variables inside `wrt(...)` currently serves two distinct semantic roles that become visually indistinguishable:

1. Pure higher-order request for a single variable (e.g. wanting f_xx, f_xxx).  
2. Encoding a multi-index sequence for mixed partials (e.g. f_xxy vs f_xyy).

Because of the implicit last-variable fill rule (see 3.a), users do not need to repeat a variable to obtain its successive higher derivatives: `wrt(x)` with a 4th-order dual already yields f, f_x, f_xx, f_xxx, f_xxxx. Yet examples of mixed partials require explicit sequencing (`wrt(x, x, y)` vs `wrt(x, y, y)`). This dual use makes it unclear whether a repeated entry was intentional for mixed ordering or an (unnecessary) attempt to obtain higher order powers of a single variable.

Illustrative contrasts:
- Pure higher-order only: `wrt(x)`  ≈  `wrt(x, x, x, x)` (redundant spelling).  
- Mixed derivative paths:  
  - `wrt(x, x, y)` encodes derivative slots (x, x, y) → enabling extraction of f_x, f_xx, f_xxy (and beyond if order permits).  
  - `wrt(x, y, y)` encodes (x, y, y) → f_x, f_xy, f_xyy.  

Ambiguity outcomes:
- Readability: Reviewer cannot tell whether an early repetition (e.g. `wrt(x, x, z)`) is for mixed semantics (x twice then z) or an unnecessary higher-order hint.  
- Error propensity: Accidental duplication (typo) silently changes both requested mixed sequence and effective order context.  
- Learning curve: Users must grasp both the implicit fill rule and explicit sequencing simultaneously.

Why this matters: Disentangling “order depth” from “variable sequence” would allow the code to communicate intent explicitly (e.g. a separate construct for pure higher-order depth, and leaving `wrt` ordering solely for mixed multi-index specification). Until then, documentation should stress: "Repetition inside `wrt` should be read first as an ordering of derivative slots for mixed partials; it is not required to obtain successive pure derivatives of a single variable." 

Potential remediation directions (see Section 9): introduce explicit higher-order descriptors (e.g. `higher_order(x,3)`), or a builder that separates `sequence(x, x, y)` from `orders(x,4)`.

## 4. Higher-Order Semantics Concerns
Mechanisms currently overlap: (a) dual type order (e.g. `dual4th`), (b) repetition (`wrt(x,x,y)`), (c) implicit "fill remaining orders with last variable" rule when fewer variables than derivative order. Combined effect complicates predictability and pedagogy.

## 5. Performance Considerations
- Forward `gradient/jacobian/hessian` re-evaluate function per variable (directional seeding per loop). For moderate dimensions, multi-direction seeding (packing derivative directions) could reduce evaluations.  
- Reverse `derivatives` loops perform bind/unbind passes—could employ scoped batching.  
- Repeated computations of per-item lengths (`wrt_item_length`) inside loops; can precompute once.  
- Template instantiation bloat possible with many distinct `wrt` argument combinations.

## 6. Safety & Robustness
- No compile-time guard against non-differentiable types beyond static assertions deep inside loops.  
- Potential dangling references storing forwarding refs to temporaries.  
- No RAII guard for seeding; misuse could leave variables seeded if control flow changes.

## 7. Consistency & Naming
- `wrt` mutates state (forward seeding) whereas `at` is pure—docs should flag side-effect nature.  
- Public re-exports (`using detail::wrt`) appear in multiple headers; centralization could avoid ODR hazards if inline variables/functions added later.

## 8. Documentation Gaps (Actionable)
Add an explicit table clarifying forms:

| Form | Prereq Type | Result (example dual4th) | Meaning |
|------|-------------|---------------------------|---------|
|`wrt(x)`|`x: dual4th`|`f0, fx, fxx, fxxx, fxxxx`|Last var absorbs higher orders| 
|`wrt(x,x)`|`x: dual4th`|`f0, fx, fxx (mixed logic not needed)`|Explicit repetition (same first 2 orders)| 
|`wrt(x,y)`|`x,y: dual`|`fx, fy`|First-order partials| 
|`wrt(x,x,y)`|`x,y: dual3rd`|`fx, fxx, fxy`|Mixed/higher combination| 
|`wrt(vec, x)`|`vec: vector<dual>`|Flattens vec then x|Gradient stacking order| 

Include mapping rules and warnings: avoid temporaries; repetition semantics; index mapping for vectors.

## 9. Improvement Recommendations (Prioritized)
1. Documentation overhaul: codify higher-order inference rule; provide mapping helper examples.  
2. Introduce explicit descriptor for higher order of single variable—e.g. `orders<4>(x)` or `higher_order(x,3)`—to separate concerns from repetition for mixing.  
3. Add concepts (or enable_if) to constrain `wrt` inputs and emit clearer errors (e.g. `requires Differentiable<T>`).  
4. Implement RAII seeding guard (`SeedScope guard(wrt);`) automatically used inside `eval` wrappers.  
5. Unify `Wrt` into shared header (`common/wrt.hpp`) to eliminate duplication.  
6. Provide `wrt_mapping(wrt_list)` returning vector of (pointer/reference, start_index, length).  
7. Refine static_assert messages (parameterized counts & suggestions).  
8. Precompute length & offsets once per call; reuse in loops.  
9. Optional multi-direction forward Jacobian strategy for small n (pack derivative orders).  
10. Batch bind/unbind in reverse mode (single pass or guard object).  

## 10. Low-Effort Wins
- Add `constexpr size_t wrt_flat_length(const Wrt<...>&)` public helper.  
- Add doc note: "Bind Eigen expressions to named variables before passing to `wrt` to avoid dangling references."  
- Provide example for extracting subset gradient indices.  
- Short alias `vars(...)` (optional) as more semantic alternative alongside `wrt` (could just wrap `wrt`).

## 11. Potential Future Extensions
- Fluent builder: `make_wrt(x,y).higher_order(3).mixed(x,y)` for explicit intent.  
- Introspection metadata object capturing variable identity & offsets for downstream tooling / bindings.  
- Compile-time duplicate-detection (warn when repetition seems unintentional).  
- Structured return type for gradients with named fields when C++ reflection matures.

## 12. Suggested RAII Sketch (Illustrative Only)
```cpp
template<class WrtList>
class SeedScope {
public:
  explicit SeedScope(const WrtList& w) : w_(w) { seed(w_); }
  ~SeedScope() { unseed(w_); }
  SeedScope(const SeedScope&) = delete;
  SeedScope& operator=(const SeedScope&) = delete;
private:
  const WrtList& w_;
};
```
Used as:
```cpp
auto eval_guarded(const Fun& f, const At<Args...>& at, const Wrt<Vars...>& w){
  SeedScope guard(w);
  return f(std::get<ArgsIndex>(at.args)...); // existing mechanics
}
```

## 13. Risk / Benefit Matrix (Abbrev.)
| Change | Effort | Risk | Benefit |
|--------|--------|------|---------|
|Doc clarification|Low|Low|High (user clarity)|
|Unified `Wrt`|Med|Low|Med (maint.)|
|Concept constraints|Med|Low|Med (safety)|
|RAII seeding|Low|Low|Med (safety)|
|Explicit higher-order descriptor|Med|Med|High (semantics)|
|Multi-direction Jacobian|High|Med|High (perf)|

## 14. Summary
`wrt` delivers a powerful, concise interface with zero-overhead abstractions, but conflates variable selection with higher-order intent and silently leverages implicit rules that can surprise users. Addressing documentation clarity first, then separating higher-order signaling, tightening type/lifetime safety, and unifying duplicated implementations will substantially improve maintainability and user experience without large architectural upheaval.

---
Generated review intended for maintainers to guide incremental refinement of the `wrt` API.
