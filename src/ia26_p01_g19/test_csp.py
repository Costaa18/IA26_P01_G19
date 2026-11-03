import unittest
from pathlib import Path

from .csp import solve_csp
from .model import Problem


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class CSPTests(unittest.TestCase):
    def test_csp_finds_a_feasible_complete_assignment(self):
        problem = Problem.from_loader(data_dir=DATA_DIR)
        result = solve_csp(problem)

        self.assertTrue(result.found)
        self.assertIsNotNone(result.solution)
        self.assertEqual(result.cost, problem.cost_function(result.solution))

        delivered = {
            stop.request_id
            for stops in result.solution.values()
            for stop in stops
            if stop.action.value == "DELIVERY"
        }
        self.assertEqual(delivered, set(problem.requests))
        for vehicle_id, stops in result.solution.items():
            feasible, reason = problem.is_route_feasible(vehicle_id, stops)
            self.assertTrue(feasible, reason)


if __name__ == "__main__":
    unittest.main()
