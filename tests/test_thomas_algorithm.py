import numpy as np
import pytest
from tss.thomas_algorithm import TridiagonalSolver


class TestTridiagonalSolver:
    def test_solve_simple_system(self):
        """Test solving a simple tridiagonal system."""
        solver = TridiagonalSolver()
        solver.set_system([1, 1], [2, 2, 2], [1, 1], [1, 2, 3])
        x = solver.solve()
        expected = np.array([0.5, 0.0, 1.5])  # Analytical solution
        np.testing.assert_allclose(x, expected, atol=1e-14)

    def test_invalid_input_lengths(self):
        """Test that invalid input lengths raise ValueError."""
        solver = TridiagonalSolver()
        with pytest.raises(ValueError):
            solver.set_system([1], [2, 2], [1], [1, 2, 3])  # Wrong lengths

    def test_non_numeric_input(self):
        """Test that non-numeric input raises ValueError."""
        solver = TridiagonalSolver()
        with pytest.raises(ValueError):
            solver.set_system(['a'], [2, 2, 2], [1, 1], [1, 2, 3])

    def test_solve_without_system(self):
        """Test that solving without setting system raises ValueError."""
        solver = TridiagonalSolver()
        with pytest.raises(ValueError):
            solver.solve()

    def test_generate_random_system(self):
        """Test generating a random system."""
        solver = TridiagonalSolver()
        a, b, c, d = solver.generate_random_system(5)
        assert len(a) == 4
        assert len(b) == 5
        assert len(c) == 4
        assert len(d) == 5

    def test_residual_calculation(self):
        """Test residual calculation."""
        solver = TridiagonalSolver()
        solver.set_system([1, 1], [2, 2, 2], [1, 1], [1, 2, 3])
        x = solver.solve()
        residuals = solver.residual(x)
        np.testing.assert_allclose(residuals, np.zeros_like(residuals), atol=1e-14)