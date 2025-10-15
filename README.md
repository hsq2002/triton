# Changes Added

# ### 1. ✅ Added Two Separate Triton Kernels
| Kernel | Purpose | Description |
|---------|----------|-------------|
| `alpha_add_kernel` | αA + B | New kernel that applies a scalar multiplier to A before addition. |
| `element_wise_matrix_multiplication` | A × B | New kernel that performs element-wise vector multiplication. |

Each kernel:
- Uses `@triton.jit` for GPU compilation.
- Implements masking to handle arbitrary input sizes safely.
- Is fully interoperable with PyTorch GPU tensors.

---

### 2. 🧠 Main Function Enhancements
**File:** `main()`  
Reorganized and expanded for clarity and correctness:
- Added an assertion to ensure a CUDA device is available.
- Allocated tensors (`a`, `b`, `c`) directly on the GPU for efficient memory access.
- Introduced a scalar multiplier `alpha = 3.0` for testing the second kernel.
- Added clear sectioned print statements for easier readability.
