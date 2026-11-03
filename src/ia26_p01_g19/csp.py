from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from .model import ActionType, Problem, Stop


Solution = dict[str, list[Stop]]


@dataclass
class CSPResult:
    solution: Solution | None
    cost: float | None
    backtracks: int
    assignments: int
    elapsed_seconds: float
    found: bool
    stopped_by_limit: bool = False


def _request_pair(problem: Problem, request_id: str) -> tuple[Stop, Stop]:
    request = problem.requests[request_id]
    return (
        Stop(request_id, ActionType.PICKUP, request.pickup_location_id),
        Stop(request_id, ActionType.DELIVERY, request.delivery_location_id),
    )


def _insertions(
    problem: Problem,
    routes: Solution,
    request_id: str,
) -> list[tuple[float, str, list[Stop]]]:
    pickup, delivery = _request_pair(problem, request_id)
    candidates: list[tuple[float, str, list[Stop]]] = []

    for vehicle_id, route in routes.items():
        for pickup_index in range(len(route) + 1):
            for delivery_index in range(pickup_index + 1, len(route) + 2):
                candidate_route = (
                    route[:pickup_index]
                    + [pickup]
                    + route[pickup_index : delivery_index - 1]
                    + [delivery]
                    + route[delivery_index - 1 :]
                )
                feasible, _ = problem.is_route_feasible(
                    vehicle_id,
                    candidate_route,
                )
                if not feasible:
                    continue

                candidate_cost = problem.route_cost(vehicle_id, candidate_route)
                candidates.append((candidate_cost, vehicle_id, candidate_route))

    return sorted(candidates, key=lambda candidate: candidate[0])


def solve_csp(
    problem: Problem,
    *,
    max_backtracks: int | None = None,
) -> CSPResult:
    """Solve the routing assignment with CSP backtracking and MRV.

    Variables are unassigned requests. A domain contains every feasible
    pickup/delivery insertion for the current partial routes. MRV chooses the
    request with the smallest current domain; candidates are ordered by route
    cost as a soft optimization preference.
    """
    started = perf_counter()
    request_ids = frozenset(problem.requests)
    routes: Solution = {vehicle_id: [] for vehicle_id in problem.vehicles}
    backtracks = 0
    assignments = 0
    stopped_by_limit = False

    def search(
        remaining: frozenset[str],
        current_routes: Solution,
    ) -> Solution | None:
        nonlocal backtracks, assignments, stopped_by_limit
        if not remaining:
            return {
                vehicle_id: list(route)
                for vehicle_id, route in current_routes.items()
            }

        domains = {
            request_id: _insertions(problem, current_routes, request_id)
            for request_id in remaining
        }
        available_domains = {
            request_id: domain
            for request_id, domain in domains.items()
            if domain
        }
        if not available_domains:
            request_id = min(
                remaining,
                key=lambda candidate_id: problem.requests[candidate_id].pickup_end,
            )
        else:
            request_id = min(
                available_domains,
                key=lambda candidate_id: (
                    len(available_domains[candidate_id]),
                    problem.requests[candidate_id].pickup_end,
                    candidate_id,
                ),
            )
        domain = domains[request_id]
        if not domain:
            backtracks += 1
            return None

        for _, vehicle_id, candidate_route in domain:
            if max_backtracks is not None and backtracks >= max_backtracks:
                stopped_by_limit = True
                return None

            next_routes = {
                candidate_vehicle_id: list(candidate_stops)
                for candidate_vehicle_id, candidate_stops in current_routes.items()
            }
            next_routes[vehicle_id] = candidate_route
            assignments += 1
            solution = search(
                remaining - {request_id},
                next_routes,
            )
            if solution is not None:
                return solution

            backtracks += 1
            if max_backtracks is not None and backtracks >= max_backtracks:
                stopped_by_limit = True
                return None

        return None

    solution = search(request_ids, routes)
    if solution is None:
        # The constructive builder is not used as a CSP answer; this keeps
        # failure semantics explicit when the CSP limit is reached.
        return CSPResult(
            solution=None,
            cost=None,
            backtracks=backtracks,
            assignments=assignments,
            elapsed_seconds=perf_counter() - started,
            found=False,
            stopped_by_limit=stopped_by_limit,
        )

    return CSPResult(
        solution=solution,
        cost=problem.cost_function(solution),
        backtracks=backtracks,
        assignments=assignments,
        elapsed_seconds=perf_counter() - started,
        found=True,
    )
