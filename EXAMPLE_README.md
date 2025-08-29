# Unified Reverse Mode Autodiff Example

This example demonstrates the usage of the unified reverse mode automatic differentiation implementation in `autodiff/reverse/unified_expr.hpp`. 

## Example Overview

The example creates a complex mathematical function composed of:

1. **100 random terms**, each containing:
   - A **Legendre polynomial** of order l=0 to l=10 in spherical coordinates (θ, φ)
   - **Associated Legendre polynomials** P_l^m(cos(θ)) with azimuthal components cos(m·φ) or sin(|m|·φ)
   - A **composition** with an ordinary polynomial of degree up to 5
   - A **random coefficient** multiplying each term

2. **Function structure**: 
   ```
   f(θ, φ) = Σ(i=1 to 100) c_i * P_i(Y_l^m(θ, φ))
   ```
   Where:
   - `c_i` are random coefficients
   - `Y_l^m(θ, φ)` are spherical harmonics (real part)
   - `P_i(x)` are ordinary polynomials of degree ≤ 5

## Key Features Demonstrated

### 1. **Unified Variable Creation**
```cpp
auto arena = std::make_shared<ExpressionArena<double>>();
UnifiedVariable<double> theta(arena, theta_val);
UnifiedVariable<double> phi(arena, phi_val);
```

### 2. **Mathematical Functions**
- Trigonometric functions: `sin()`, `cos()`, `tan()`
- Exponential functions: `exp()`, `log()`, `sqrt()`
- Power functions: `pow()`
- Arithmetic operations: `+`, `-`, `*`, `/`

### 3. **Gradient Computation**
```cpp
auto f = compute_complex_function(theta, phi, /* parameters */);
auto grads = derivatives(f, wrt(theta, phi));
double df_dtheta = grads[0];
double df_dphi = grads[1];
```

### 4. **Arena Management**
- All variables share the same arena for efficient memory management
- Arena automatically tracks expression dependencies
- Final arena size: ~33,000 expressions for this complex function

## Mathematical Components

### Legendre Polynomials
- Implemented using recurrence relations
- P_0(x) = 1, P_1(x) = x
- P_{n+1}(x) = ((2n+1)xP_n(x) - nP_{n-1}(x))/(n+1)

### Associated Legendre Polynomials
- P_l^m(x) computed for spherical harmonics
- Handles both positive and negative m values

### Spherical Harmonics (Real Part)
- Y_l^m(θ, φ) = N_lm * P_l^|m|(cos(θ)) * [cos(m·φ) or sin(|m|·φ)]
- Proper normalization factors included

### Ordinary Polynomials
- P(x) = c_0 + c_1·x + c_2·x² + c_3·x³ + c_4·x⁴ + c_5·x⁵
- Random coefficients for each term

## Results

The example computes:
- **Function values** at multiple test points
- **First-order gradients** ∂f/∂θ and ∂f/∂φ
- **Gradient magnitudes** |∇f|

Sample output:
- At (60°, 45°): f = 0.754300, ∂f/∂θ = -0.315411, ∂f/∂φ = -0.057426
- Gradient magnitude: |∇f| = 0.320596

## Usage Notes

1. **Type Safety**: All constants must be explicitly cast to `double` when creating `UnifiedVariable`
2. **Arena Sharing**: Variables must share the same arena for operations
3. **Function Composition**: Complex functions can be built by composing simpler functions
4. **Automatic Differentiation**: The system automatically computes exact derivatives through the expression tree

This example showcases the power and flexibility of the unified reverse mode autodiff system for complex mathematical computations involving spherical harmonics and polynomial compositions.
