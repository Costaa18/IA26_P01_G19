import unittest
from pathlib import Path

from .hill_climbing import hill_climbing
from .model import Problem


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class HillClimbingTests(unittest.TestCase):
    def test_hill_climbing_preserves_feasibility_and_cost_history(self):
        problem = Problem.from_loader(data_dir=DATA_DIR)
        result = hill_climbing(problem, max_iterations=2)

        self.assertEqual(result.cost, problem.cost_function(result.solution))
        self.assertEqual(result.cost_history[0], 1456.19)
        self.assertLessEqual(result.cost_history[-1], result.cost_history[0])
        self.assertGreaterEqual(result.generated_neighbors, 1)
        for vehicle_id, stops in result.solution.items():
            feasible, reason = problem.is_route_feasible(vehicle_id, stops)
            self.assertTrue(feasible, reason)


if __name__ == "__main__":
    unittest.main()