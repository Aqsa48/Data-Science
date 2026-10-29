"""
NumPy Basics for Data Science
================================
Covers: array creation, indexing, slicing, broadcasting, aggregations,
        linear algebra, and random number generation.
"""

import numpy as np

# ---------------------------------------------------------
# 1. Creating Arrays
# ---------------------------------------------------------
print("=== Creating Arrays ===")

arr_1d = np.array([1, 2, 3, 4, 5])
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print("1-D array:", arr_1d)
print("2-D array:\n", arr_2d)
print("Shape:", arr_2d.shape, "  ndim:", arr_2d.ndim, "  dtype:", arr_2d.dtype)

# Convenience constructors
print("\nzeros (2x3):\n", np.zeros((2, 3)))
print("ones  (2x3):\n", np.ones((2, 3)))
print("eye   (3x3):\n", np.eye(3))
print("arange(0,10,2):", np.arange(0, 10, 2))
print("linspace(0,1,5):", np.linspace(0, 1, 5))

# ---------------------------------------------------------
# 2. Array Indexing & Slicing
# ---------------------------------------------------------
print("\n=== Indexing & Slicing ===")

matrix = np.arange(1, 13).reshape(3, 4)
print("Matrix:\n", matrix)
print("Row 0:", matrix[0])
print("Col 2:", matrix[:, 2])
print("Sub-matrix [0:2, 1:3]:\n", matrix[0:2, 1:3])

# Boolean masking
evens_mask = matrix % 2 == 0
print("Even elements:", matrix[evens_mask])

# Fancy indexing
rows = [0, 2]
cols = [1, 3]
print("Fancy index [rows, cols]:", matrix[rows, cols])

# ---------------------------------------------------------
# 3. Array Operations & Broadcasting
# ---------------------------------------------------------
print("\n=== Operations & Broadcasting ===")

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("a + b =", a + b)
print("a * b =", a * b)
print("a ** 2 =", a ** 2)
print("dot(a, b) =", np.dot(a, b))

# Broadcasting: add a column vector to each row
col_vec = np.array([[10], [20], [30]])
print("Matrix + col_vec:\n", matrix + col_vec)

# ---------------------------------------------------------
# 4. Aggregations
# ---------------------------------------------------------
print("\n=== Aggregations ===")

data = np.array([[4, 7, 2], [1, 9, 5]])
print("Data:\n", data)
print("sum (all):", data.sum())
print("sum (axis=0 / col):", data.sum(axis=0))
print("sum (axis=1 / row):", data.sum(axis=1))
print("mean:", data.mean())
print("std:", round(data.std(), 4))
print("min:", data.min(), "  max:", data.max())
print("argmin:", data.argmin(), "  argmax:", data.argmax())

# ---------------------------------------------------------
# 5. Reshaping & Stacking
# ---------------------------------------------------------
print("\n=== Reshaping & Stacking ===")

arr = np.arange(12)
print("Original:", arr)
print("Reshaped (3x4):\n", arr.reshape(3, 4))
print("Reshaped (2x2x3):\n", arr.reshape(2, 2, 3))

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
print("hstack:", np.hstack([x, y]))
print("vstack:\n", np.vstack([x, y]))
print("column_stack:\n", np.column_stack([x, y]))

# ---------------------------------------------------------
# 6. Linear Algebra
# ---------------------------------------------------------
print("\n=== Linear Algebra ===")

A = np.array([[2, 1], [5, 3]])
b_vec = np.array([8, 17])

print("Matrix A:\n", A)
print("Det(A):", round(np.linalg.det(A), 4))
print("Inv(A):\n", np.linalg.inv(A))

x_sol = np.linalg.solve(A, b_vec)
print("Solution to Ax=b:", x_sol)

eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:", eigenvalues.round(4))

# ---------------------------------------------------------
# 7. Random Number Generation
# ---------------------------------------------------------
print("\n=== Random Numbers ===")

rng = np.random.default_rng(seed=42)

print("Uniform [0,1) (5):", rng.random(5).round(4))
print("Normal  μ=0,σ=1 (5):", rng.standard_normal(5).round(4))
print("Integers 1-10 (5):", rng.integers(1, 11, size=5))

# Simulate rolling two dice 100k times
dice = rng.integers(1, 7, size=(100_000, 2)).sum(axis=1)
print("Mean of 2-dice rolls (≈7):", dice.mean())

print("\nDone! NumPy basics covered successfully.")
