Autodiff Library
===

Concise, API-accurate guide for using the autodiff C++ library in forward and reverse modes (derived from headers and tests).

## 1. Architecture Overview
- Forward mode (files under `autodiff/forward/`): expression-template dual numbers supporting higher-order derivatives via nested dual types (`HigherOrderDual<N, T>`). Convenience aliases: `dual` (1st), `dual2nd`, `dual3rd`, `dual4th` (unless `AUTODIFF_DISABLE_HIGHER_ORDER`). Also a compact `real` family (lightweight alternative) in `forward/real.hpp`.
- Reverse mode (files under `autodiff/reverse/`): dynamic expression graph with `var` (`Variable<T>` internally) enabling gradients; optional higher-order (Hessian) if higher-order not disabled.
- Utility layers: `wrt(...)`, `at(...)`, `along(...)` descriptors for seeding; algorithms: `derivative`, `derivatives`, `gradient`, `jacobian`, `hessian`, `taylorseries`.
- Eigen integration: specialized traits + helpers in `forward/*/eigen.hpp` and `reverse/var/eigen.hpp` (auto gradient/Jacobian/Hessian into `Eigen::VectorXd` / `MatrixXd`).

## 2. Core Types & Aliases
Forward:
```cpp
using autodiff::dual;     // 1st order dual (dual1st)
using autodiff::dual2nd;  // 2nd order (if enabled)
using autodiff::dual3rd;  // 3rd order
using autodiff::dual4th;  // 4th order
using autodiff::real;     // Alternative forward type (Real<N,T>)
```
Reverse:
```cpp
using autodiff::var;      // reverse mode variable (dynamic tape node)
```
Value extraction (both modes):
```cpp
val(expr)   // returns underlying numeric value (double, etc.)
```

## 3. Descriptor Helpers (Forward Mode)
Provided in `forward/utils/derivative.hpp` under namespace `autodiff`:
- `wrt(x, y, ...)`: declare variables with respect to which derivatives are computed (supports mixing scalars, Eigen vectors, repeating variables for higher mixed orders, e.g. `wrt(x, x, y)` seeds successive orders).
- `at(x, y, ...)`: bind current variable objects to be evaluated (their current values & derivative storage used).
- `along(v)`: directional derivative / Taylor expansion direction descriptor.
- `seed(...)` / `unseed(...)`: manual control of derivative seeds (rarely needed directly; algorithms manage seeding/unseeding automatically).

## 4. Forward Mode APIs
All overload sets live in namespace `autodiff` (via `using` imports from `detail`).

Scalar / vector derivative extraction (after building an expression):
```cpp
dual x = 2.0;
dual y = 3.0;
dual f = sin(x*y) + x*x;
double dfdx = derivative(f, wrt(x));          // ∂f/∂x
double dfdy = derivative(f, wrt(y));          // ∂f/∂y
```

Function form (lazily seeds each variable):
```cpp
auto f = [](auto x, auto y) { return sin(x*y) + x*x; };
dual x = 2.0, y = 3.0;
double dfdx = derivative(f, wrt(x), at(x, y));
double dfdy = derivative(f, wrt(y), at(x, y));
```

Higher-order & mixed derivatives (needs higher-order dual type, e.g. `dual2nd` or repetition in `wrt`):
```cpp
dual2nd x = 1.0, y = 2.0;
auto g = [](auto x, auto y) { return exp(x*y); };
auto all = derivatives(g, wrt(x, y), at(x, y)); // array of value + partials up to order of dual
double f0  = all[0];     // f
double fx  = all[1];     // ∂f/∂x
double fxy = all[2];     // ∂²f/∂x∂y (ordering follows seeding rules)
```
Alternative explicit order query:
```cpp
double fxx = derivative<2>(g, wrt(x), at(x, y));
```

Gradients / Jacobians / Hessians (forward):
```cpp
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
using namespace autodiff;

VectorXdual x(5); x << 2,3,5,7,11;
auto s = [](const auto& x){ return (x * x).sum(); }; // scalar
auto v = [](const auto& x){ return x.sin(); };       // vector-valued

dual y; VectorXd g; // pre-allocated forms supported
gradient(s, wrt(x), at(x), y, g);          // fills y (value) and g (∇s)
MatrixXd J = jacobian(v, wrt(x), at(x));   // each row: component derivatives
MatrixXd H = hessian(s, wrt(x), at(x));    // needs dual2nd (or higher) elements in x
```

Directional derivatives & Taylor series:
```cpp
VectorXdual2nd x(3); x << 1,2,3;
VectorXd vdir(3); vdir << 0.1, -0.2, 0.05; // direction
auto f = [](const auto& x){ return x.sin().sum(); };
auto series = taylorseries(f, along(vdir), at(x));
auto coeffs = series.derivatives(); // array of directional derivatives order 0..N
double val_dir_t = series(0.01);    // evaluate truncated series at t=0.01
```

Mixed-variable subset ordering (as seen in tests):
```cpp
auto gfun = [](const auto& x){ return (x.log() * x).sum(); };
auto grad_sub = gradient(gfun, wrt(x[3], x[0], x[4]), at(x)); // only selected entries
```

Notes:
- Repeating variables in `wrt(x, x, y)` seeds successive derivative levels for higher-order mixed partial extraction.
- `gradient`, `jacobian`, and `hessian` have overloads accepting pre-allocated Eigen storage (`gradient(f, wrt(x), at(x), u, g)` etc.).
- `hessian` exploits symmetry; forward mode computes pairwise seeding combos internally.

## 5. Reverse Mode APIs
Include:
```cpp
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
using namespace autodiff;

VectorXvar x(5); x << 1,2,3,4,5;
var y = x.sum();
VectorXd g = gradient(y, x);       // ∇y (all ones)
MatrixXd H = hessian(y, x, g);     // (zeros) — only if higher-order not disabled

// Another example
y = (x.array().sin()).prod();
g = gradient(y, x);
#ifndef AUTODIFF_DISABLE_HIGHER_ORDER
H = hessian(y, x, g);  // builds gradient expression graph first, then second pass
#endif
```
Key points:
- Reverse mode builds an expression graph lazily as you compose `var` expressions (operator overloads in `var/var.hpp`).
- `gradient(y, x)` binds derivative accumulation buffers to each variable then calls `propagate(1)`. Always pass the terminal scalar `y` first.
- `hessian(y, x, g)` first forms gradient expressions (second-order propagation) then evaluates each second derivative via repeated propagation.
- Use `val(v)` to extract numeric values from `var` or gradient components (tests demonstrate `val(g[i])`).

## 6. Choosing Forward vs Reverse
- Use forward mode when number of independent variables is small (or you need higher-order derivatives, Taylor series along directions, mixed partials).
- Use reverse mode when the function is scalar-valued with many inputs (efficient gradient) and when you only need first derivatives (optional Hessian if enabled, but cost scales roughly with N gradient evaluations).

## 7. Eigen Integration Summary
- Forward: Include `<autodiff/forward/dual/eigen.hpp>` or `<autodiff/forward/real/eigen.hpp>` to get `VectorXdual`, `ArrayXdual`, etc., plus algorithms acting on Eigen objects.
- Reverse: Include `<autodiff/reverse/var/eigen.hpp>` for `VectorXvar` and trait registration enabling arithmetic & standard ops.
- All Jacobian/Hessian outputs are plain floating point (`double`) matrices/vectors, not dual/var objects (except gradient components returned inside forward Hessian internal computations before extraction).

## 8. Taylor Series Utility
`taylorseries(f, along(v), at(x))` returns a `TaylorSeries<N,V>` object (N = order of internal dual numbers, V = scalar or vector type). Use `.derivatives()` to access directional derivative coefficients (0..N). Evaluation uses Horner-like accumulation with factorial division baked into iteration.

## 9. Practical Patterns
- Scalar objective + many inputs: reverse `var`, `gradient(y,x)`.
- Vector output + moderate inputs: forward `jacobian(f, wrt(x), at(x))`.
- Need mixed partials up to 4th: forward with `dual4th` and appropriate `wrt(...)` repetition.
- Directional higher-order expansion: forward `taylorseries` with `along(direction)`.
- Subset gradients: pass selected entries or slices in `wrt(...)` (e.g., `wrt(x.tail(4), x[0])`).
- Memory reuse: pass pre-allocated Eigen objects to `gradient/jacobian/hessian` overloads.

## 10. Minimal Build Integration
```cmake
find_package(autodiff REQUIRED)
add_executable(myprog main.cpp)
target_link_libraries(myprog PRIVATE autodiff::autodiff)
```
```cpp
#include <autodiff/forward/dual.hpp>
int main(){ autodiff::dual x = 2.0; auto f = x * x; double fx = derivative(f, wrt(x)); }
```

## 11. Reference: Key API Signatures (Simplified)
```cpp
// Seeding descriptors
auto wrt(vars...); auto at(vars...); auto along(vecs...);

// Forward evaluation helpers
eval(f, at(x,...), wrt(x,...));
derivative<order=1>(f, wrt(x,...), at(x,...));
derivatives(f, wrt(x,...), at(x,...)); // array (value + higher orders)
gradient(f, wrt(x,...), at(x,...));    // Eigen vector (scalar f)
jacobian(f, wrt(x,...), at(x,...));    // Eigen matrix (vector f)
hessian(f, wrt(x,...), at(x,...));     // Eigen matrix (scalar f, higher-order enabled)
taylorseries(f, along(v), at(x,...));  // TaylorSeries<N,V>

// Reverse (y scalar var; x vector of var)
gradient(y, x);            // VectorXd
hessian(y, x, g);          // MatrixXd (needs higher order support)

// Value extraction
val(expr_or_var);
```

## 12. Limitations & Flags
- `AUTODIFF_DISABLE_HIGHER_ORDER` removes higher-order derivative storage: forward mode only first-order; reverse mode Hessian unsupported (throws).
- Mixed derivative ordering depends on `wrt` argument ordering & repetition.
- Reverse mode current public utilities cover gradients and (optionally) Hessians; higher than 2nd order not exposed in simplified API.

## 13. Where to Look Next
- Examples: `examples/forward/` (directional taylor series, gradients, Hessians) & `examples/reverse/`.
- Tests: `tests/forward/utils/gradient.test.cpp` (patterns for wrt subsets, reuse) and `tests/reverse/var/eigen.test.cpp` (reverse gradients/Hessians).
- Source: `forward/utils/gradient.hpp`, `forward/utils/derivative.hpp`, `reverse/var/var.hpp` for deeper customization.

