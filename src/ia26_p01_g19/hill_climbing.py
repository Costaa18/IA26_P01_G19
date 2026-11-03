from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from .informed_search import constructive_greedy_solution
from .model import ActionType, Problem, Stop


Solution = dict[str, list[Stop]]


@dataclass
class HillClimbingResult:
    solution: Solution
    cost: float
    cost_history: list[float]
    iterations: int
    generated_neighbors: int
    elapsed_seconds: float
    converged: bool


def _request_pair(problem: Problem, request_id: str) -> tuple[Stop, Stop]:
    request = problem.requests[request_id]
    return (
        Stop(request_id, ActionType.PICKUP, request.pickup_location_id),
        Stop(request_id, ActionType.DELIVERY, request.delivery_location_id),
    )


def _remove_request(solution: Solution, request_id: str) -> Solution:
    return {
        vehicle_id: [
            stop for stop in stops if stop.request_id != request_id
        ]
        for vehicle_id, stops in solution.items()
    }


def _neighbors(problem: Problem, solution: Solution):
    yield from _relocation_neighbors(problem, solution)
    yield from _swap_neighbors(problem, solution)


def _relocation_neighbors(problem: Problem, solution: Solution):
    request_ids = tuple(problem.requests)
    vehicle_ids = tuple(problem.vehicles)

    for request_id in request_ids:
        base_solution = _remove_request(solution, request_id)
        pickup, delivery = _request_pair(problem, request_id)

        for vehicle_id in vehicle_ids:
            route = base_solution[vehicle_id]
            for pickup_index in range(len(route) + 1):
                for delivery_index in range(pickup_index + 1, len(route) + 2):
                    candidate_route = (
                        route[:pickup_index]
                        + [pickup]
                        + route[pickup_index : delivery_index - 1]
                        + [delivery]
                        + route[delivery_index - 1 :]
                    )
                    candidate = {
                        candidate_vehicle_id: list(candidate_stops)
                        for candidate_vehicle_id, candidate_stops
                        in base_solution.items()
                    }
                    candidate[vehicle_id] = candidate_route

                    feasible = all(
                        problem.is_route_feasible(
                            candidate_vehicle_id,
                            candidate_stops,
                        )[0]
                        for candidate_vehicle_id, candidate_stops
                        in candidate.items()
                    )
                    if feasible:
                        yield candidate


def _swap_neighbors(problem: Problem, solution: Solution):
    """Generate neighbors by swapping complete request identities."""
    request_ids = tuple(problem.requests)
    for first_index, first_request_id in enumerate(request_ids):
        for second_request_id in request_ids[first_index + 1 :]:
            positions = {}
            for vehicle_id, route in solution.items():
                for position, stop in enumerate(route):
                    if stop.request_id in {
                        first_request_id,
                        second_request_id,
                    }:
                        positions.setdefault(stop.request_id, []).append(
                            (vehicle_id, position, stop.action)
                        )

            if set(positions) != {first_request_id, second_request_id}:
                continue

            candidate = {
                vehicle_id: list(route)
                for vehicle_id, route in solution.items()
            }
            for request_id, replacement_id in (
                (first_request_id, second_request_id),
                (second_request_id, first_request_id),
            ):
                for vehicle_id, position, action in positions[request_id]:
                    replacement = problem.requests[replacement_id]
                    location_id = (
                        replacement.pickup_location_id
                        if action is ActionType.PICKUP
                        else replacement.delivery_location_id
                    )
                    candidate[vehicle_id][position] = Stop(
                        replacement_id,
                        action,
                        location_id,
                    )

            feasible = all(
                problem.is_route_feasible(vehicle_id, route)[0]
                for vehicle_id, route in candidate.items()
            )
            if feasible:
                yield candidate


def hill_climbing(
    problem: Problem,
    initial_solution: Solution | None = None,
    *,
    max_iterations: int = 100,
) -> HillClimbingResult:
    """Improve a feasible solution using best-improvement hill climbing."""
    started = perf_counter()
    solution = initial_solution or constructive_greedy_solution(problem)
    if solution is None:
        raise ValueError("Hill-Climbing requires a feasible initial solution.")

    current_cost = problem.cost_function(solution)
    cost_history = [current_cost]
    generated_neighbors = 0
    iterations = 0

    while iterations < max_iterations:
        best_solution: Solution | None = None
        best_cost = current_cost

        for candidate in _neighbors(problem, solution):
            generated_neighbors += 1
            candidate_cost = problem.cost_function(candidate)
            if candidate_cost < best_cost - 1e-9:
                best_solution = candidate
                best_cost = candidate_cost

        if best_solution is None:
            break

        solution = best_solution
        current_cost = best_cost
        cost_history.append(current_cost)
        iterations += 1

    return HillClimbingResult(
        solution=solution,
        cost=problem.cost_function(solution),
        cost_history=cost_history,
        iterations=iterations,
        generated_neighbors=generated_neighbors,
        elapsed_seconds=perf_counter() - started,
        converged=iterations < max_iterations,
    )
