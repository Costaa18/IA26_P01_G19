import unittest
from pathlib import Path
 
from .heuristics import (
    heuristic_next_required_location,
    heuristic_request_chain,
)
from .informed_search import (
    a_star_search,
    expand_state,
    greedy_best_first_search,
    initial_state,
    informed_search_results_to_records,
    run_informed_search_experiments,
)
from .model import Problem
 
 
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
 
 
class InformedSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        complete_problem = Problem.from_loader(data_dir=DATA_DIR)
        # Two requests keep the tests fast and easy to understand.
        request_ids = ("1", "3")
        cls.problem = Problem(
            vehicles=complete_problem.vehicles,
            requests={
                request_id: complete_problem.requests[request_id]
                for request_id in request_ids
            },
            distance_matrix=complete_problem.distance_matrix,
        )
 
    def test_initial_state_contains_all_vehicles(self):
        state = initial_state(self.problem)
 
        self.assertEqual(len(state.routes), len(self.problem.vehicles))
        self.assertEqual(state.current_loads, (0.0, 0.0))
        self.assertEqual(state.picked_by_vehicle, (frozenset(), frozenset()))
        self.assertEqual(state.delivered_requests, frozenset())
        self.assertEqual(state.cost, 0.0)
 
    def test_initial_state_has_feasible_successors(self):
        successors = list(expand_state(self.problem, initial_state(self.problem)))
 
        self.assertGreater(len(successors), 0)
        self.assertTrue(all(successor.cost >= 0 for successor in successors))
 
    def test_greedy_finds_a_valid_solution_with_both_heuristics(self):
        # Greedy should find a route, but it does not promise the best route.
        for heuristic in (
            heuristic_request_chain,
            heuristic_next_required_location,
        ):
            with self.subTest(heuristic=heuristic.__name__):
                result = greedy_best_first_search(
                    self.problem,
                    heuristic,
                    max_expansions=10_000,
                )
                self.assertTrue(result.found)
                self.assertIsNotNone(result.solution)
                self.assertAlmostEqual(
                    result.cost,
                    self.problem.cost_function(result.solution),
                )
                self.assert_solution_is_feasible(result.solution)
 
    def test_a_star_finds_a_valid_solution_with_both_heuristics(self):
        # A* also uses the cost already travelled.
        for heuristic in (
            heuristic_request_chain,
            heuristic_next_required_location,
        ):
            with self.subTest(heuristic=heuristic.__name__):
                result = a_star_search(
                    self.problem,
                    heuristic,
                    max_expansions=10_000,
                )
                self.assertTrue(result.found)
                self.assertIsNotNone(result.solution)
                self.assertAlmostEqual(
                    result.cost,
                    self.problem.cost_function(result.solution),
                )
                self.assert_solution_is_feasible(result.solution)
 
    def test_search_reports_expansion_limit(self):
        # A zero limit must stop before expanding the initial state.
        result = greedy_best_first_search(
            self.problem,
            heuristic_request_chain,
            max_expansions=0,
        )
 
        self.assertFalse(result.found)
        self.assertTrue(result.stopped_by_limit)
        self.assertEqual(result.expanded_nodes, 0)

    def test_experiments_run_all_algorithm_heuristic_combinations(self):
        results = run_informed_search_experiments(
            self.problem,
            max_expansions=10_000,
        )

        self.assertEqual(
            set(results),
            {
                "greedy_request_chain",
                "greedy_next_required_location",
                "a_star_request_chain",
                "a_star_next_required_location",
            },
        )
        self.assertTrue(all(result.found for result in results.values()))

    def test_experiment_results_are_recorded_for_comparison(self):
        results = run_informed_search_experiments(
            self.problem,
            max_expansions=10_000,
        )
        records = informed_search_results_to_records(results)

        self.assertEqual(len(records), 4)
        expected_fields = {
            "experiment",
            "algorithm",
            "heuristic",
            "found",
            "cost_km",
            "elapsed_seconds",
            "expanded_nodes",
            "generated_nodes",
            "stopped_by_limit",
        }
        self.assertTrue(all(set(record) == expected_fields for record in records))
        self.assertTrue(all(record["found"] for record in records))
        self.assertTrue(all(record["cost_km"] is not None for record in records))
 
    def assert_solution_is_feasible(self, solution):
        delivered_request_ids = set()
 
        for vehicle_id, stops in solution.items():
            # The normal model checker validates the final routes.
            feasible, reason = self.problem.is_route_feasible(vehicle_id, stops)
            self.assertTrue(feasible, reason)
            delivered_request_ids.update(
                stop.request_id
                for stop in stops
                if stop.action.value == "DELIVERY"
            )
 
        self.assertEqual(delivered_request_ids, set(self.problem.requests))
 
 
if __name__ == "__main__":
    unittest.main()
 