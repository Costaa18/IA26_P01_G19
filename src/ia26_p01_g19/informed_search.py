from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from time import perf_counter
from typing import Callable, Iterator

from .heuristics import heuristic_next_required_location, heuristic_request_chain
from .model import ActionType, Problem, State, Stop


Heuristic = Callable[[Problem, State], float]


@dataclass
class SearchResult:
    solution: dict[str, list[Stop]] | None
    cost: float | None
    expanded_nodes: int
    generated_nodes: int
    elapsed_seconds: float
    found: bool
    stopped_by_limit: bool = False
    used_constructive_fallback: bool = False


def _vehicle_ids(problem: Problem) -> tuple[str, ...]:
    return tuple(problem.vehicles)


def initial_state(problem: Problem) -> State:
    """Create the state from which both vehicles start searching."""
    vehicle_ids = _vehicle_ids(problem)
    if not vehicle_ids:
        raise ValueError("The problem must contain at least one vehicle.")
    if not problem.requests:
        raise ValueError("The problem must contain at least one request.")

    start_time = min(request.pickup_start for request in problem.requests.values())
    return State(
        routes=tuple(() for _ in vehicle_ids),
        current_locations=tuple(
            problem.vehicles[vehicle_id].center_id for vehicle_id in vehicle_ids
        ),
        current_times=tuple(start_time for _ in vehicle_ids),
        current_loads=tuple(0.0 for _ in vehicle_ids),
        picked_by_vehicle=tuple(frozenset() for _ in vehicle_ids),
        delivered_requests=frozenset(),
        cost=0.0,
    )


def _distance(problem: Problem, origin: str, destination: str) -> float:
    if origin == destination:
        return 0.0
    route = problem.distance_matrix.get((origin, destination))
    if route is None:
        raise ValueError(f"Route between {origin} and {destination} not found.")
    return float(route["distance_km"])


def _state_key(state: State) -> tuple:
    return (
        state.current_locations,
        state.current_times,
        state.current_loads,
        state.picked_by_vehicle,
        state.delivered_requests,
    )


def _next_state(
    problem: Problem,
    state: State,
    vehicle_index: int,
    request_id: str,
    action: ActionType,
    vehicle_ids: tuple[str, ...],
) -> State | None:
    request = problem.requests[request_id]
    vehicle_id = vehicle_ids[vehicle_index]
    vehicle = problem.vehicles[vehicle_id]

    if action is ActionType.PICKUP:
        location_id = request.pickup_location_id
        window_start = request.pickup_start
        window_end = request.pickup_end
        new_load = state.current_loads[vehicle_index] + request.demand
        if new_load > vehicle.capacity:
            return None
    else:
        location_id = request.delivery_location_id
        window_start = request.delivery_start
        window_end = request.delivery_end
        new_load = state.current_loads[vehicle_index] - request.demand

    travel = problem.get_travel_duration(
        state.current_locations[vehicle_index], location_id
    )
    arrival = state.current_times[vehicle_index] + travel
    if arrival > window_end:
        return None

    service_time = max(arrival, window_start)
    stop = Stop(request_id=request_id, action=action, location_id=location_id)
    new_route = (*state.routes[vehicle_index], stop)

    departure_time = min(
        request_data.pickup_start for request_data in problem.requests.values()
    )
    feasible, _ = problem.is_partial_route_feasible(
        vehicle_id, list(new_route), departure_time=departure_time
    )
    if not feasible:
        return None

    routes = list(state.routes)
    routes[vehicle_index] = new_route
    locations = list(state.current_locations)
    locations[vehicle_index] = location_id
    times = list(state.current_times)
    times[vehicle_index] = service_time
    loads = list(state.current_loads)
    loads[vehicle_index] = new_load
    picked = list(state.picked_by_vehicle)
    delivered = state.delivered_requests

    if action is ActionType.PICKUP:
        picked[vehicle_index] = picked[vehicle_index] | {request_id}
    else:
        delivered = delivered | {request_id}
        picked[vehicle_index] = picked[vehicle_index] - {request_id}

    return State(
        routes=tuple(routes),
        current_locations=tuple(locations),
        current_times=tuple(times),
        current_loads=tuple(loads),
        picked_by_vehicle=tuple(picked),
        delivered_requests=delivered,
        cost=state.cost + _distance(
            problem, state.current_locations[vehicle_index], location_id
        ),
    )


def expand_state(problem: Problem, state: State) -> Iterator[State]:
    """Generate all feasible one-stop successors of a search state."""
    vehicle_ids = _vehicle_ids(problem)
    picked = set().union(*state.picked_by_vehicle)

    for vehicle_index, vehicle_id in enumerate(vehicle_ids):
        for request_id in problem.requests:
            if request_id not in picked and request_id not in state.delivered_requests:
                child = _next_state(
                    problem,
                    state,
                    vehicle_index,
                    request_id,
                    ActionType.PICKUP,
                    vehicle_ids,
                )
                if child is not None:
                    yield child

            if request_id in state.picked_by_vehicle[vehicle_index]:
                child = _next_state(
                    problem,
                    state,
                    vehicle_index,
                    request_id,
                    ActionType.DELIVERY,
                    vehicle_ids,
                )
                if child is not None:
                    yield child


def _is_goal(problem: Problem, state: State) -> bool:
    return len(state.delivered_requests) == len(problem.requests)


def constructive_greedy_solution(
    problem: Problem,
) -> dict[str, list[Stop]] | None:
    """Build a feasible solution by inserting complete request pairs.

    Requests with the earliest pickup deadlines are considered first. For each
    request, every vehicle and every pickup/delivery insertion pair is tested
    with the model's feasibility checker.
    """
    routes = {vehicle_id: [] for vehicle_id in _vehicle_ids(problem)}

    requests = sorted(
        problem.requests.values(),
        key=lambda request: request.pickup_end,
    )
    for request in requests:
        pickup = Stop(
            request_id=request.item_id,
            action=ActionType.PICKUP,
            location_id=request.pickup_location_id,
        )
        delivery = Stop(
            request_id=request.item_id,
            action=ActionType.DELIVERY,
            location_id=request.delivery_location_id,
        )
        best_insertion: tuple[float, str, list[Stop]] | None = None

        for vehicle_id, route in routes.items():
            for pickup_index in range(len(route) + 1):
                for delivery_index in range(pickup_index + 1, len(route) + 2):
                    candidate = (
                        route[:pickup_index]
                        + [pickup]
                        + route[pickup_index : delivery_index - 1]
                        + [delivery]
                        + route[delivery_index - 1 :]
                    )
                    feasible, _ = problem.is_route_feasible(vehicle_id, candidate)
                    if not feasible:
                        continue

                    candidate_cost = problem.route_cost(vehicle_id, candidate)
                    insertion = (candidate_cost, vehicle_id, candidate)
                    if best_insertion is None or insertion[0] < best_insertion[0]:
                        best_insertion = insertion

        if best_insertion is None:
            return None

        _, vehicle_id, route = best_insertion
        routes[vehicle_id] = route

    return routes


def _urgency_score(problem: Problem, state: State) -> float:
    """Return a tie-break score favouring states with tighter windows."""
    picked = set().union(*state.picked_by_vehicle)
    remaining_deadlines = []

    for request_id, request in problem.requests.items():
        if request_id in state.delivered_requests:
            continue
        deadline = (
            request.delivery_end
            if request_id in picked
            else request.pickup_end
        )
        remaining_deadlines.append(
            min(
                (deadline - current_time).total_seconds()
                for current_time in state.current_times
            )
        )

    return min(remaining_deadlines, default=0.0)


def _search(
    problem: Problem,
    heuristic: Heuristic,
    *,
    use_path_cost: bool,
    max_expansions: int | None,
    use_constructive_fallback: bool,
) -> SearchResult:
    started = perf_counter()
    initial = initial_state(problem)
    frontier: list[tuple[float, float, int, State]] = []
    sequence = count()
    heappush(
        frontier,
        (
            heuristic(problem, initial),
            _urgency_score(problem, initial),
            next(sequence),
            initial,
        ),
    )
    best_cost = {_state_key(initial): initial.cost}
    expanded_costs: dict[tuple, float] = {}
    expanded_nodes = 0
    generated_nodes = 1

    while frontier:
        _, _, _, state = heappop(frontier)
        state_key = _state_key(state)
        if state.cost > best_cost.get(state_key, float("inf")):
            continue
        if state.cost >= expanded_costs.get(state_key, float("inf")):
            continue
        expanded_costs[state_key] = state.cost
        if _is_goal(problem, state):
            solution = {
                vehicle_id: list(route)
                for vehicle_id, route in zip(_vehicle_ids(problem), state.routes)
            }

            final_cost = problem.cost_function(solution)
            return SearchResult(
                solution=solution,
                cost=final_cost,
                expanded_nodes=expanded_nodes,
                generated_nodes=generated_nodes,
                elapsed_seconds=perf_counter() - started,
                found=True,
            )

        if max_expansions is not None and expanded_nodes >= max_expansions:
            break
        expanded_nodes += 1

        for child in expand_state(problem, state):
            generated_nodes += 1
            key = _state_key(child)
            if child.cost >= best_cost.get(key, float("inf")):
                continue
            best_cost[key] = child.cost
            priority = heuristic(problem, child)
            if use_path_cost:
                priority += child.cost
            heappush(
                frontier,
                (
                    priority,
                    _urgency_score(problem, child),
                    next(sequence),
                    child,
                ),
            )

    if use_constructive_fallback:
        solution = constructive_greedy_solution(problem)
        if solution is not None:
            return SearchResult(
                solution=solution,
                cost=problem.cost_function(solution),
                expanded_nodes=expanded_nodes,
                generated_nodes=generated_nodes,
                elapsed_seconds=perf_counter() - started,
                found=True,
                stopped_by_limit=(
                    max_expansions is not None
                    and expanded_nodes >= max_expansions
                ),
                used_constructive_fallback=True,
            )

    return SearchResult(
        solution=None,
        cost=None,
        expanded_nodes=expanded_nodes,
        generated_nodes=generated_nodes,
        elapsed_seconds=perf_counter() - started,
        found=False,
        stopped_by_limit=(
            max_expansions is not None and expanded_nodes >= max_expansions
        ),
    )


def greedy_best_first_search(
    problem: Problem,
    heuristic: Heuristic,
    max_expansions: int | None = None,
    use_constructive_fallback: bool = False,
) -> SearchResult:
    """Run Greedy Best-First Search with f(n) = h(n)."""
    return _search(
        problem,
        heuristic,
        use_path_cost=False,
        max_expansions=max_expansions,
        use_constructive_fallback=use_constructive_fallback,
    )


def a_star_search(
    problem: Problem,
    heuristic: Heuristic,
    max_expansions: int | None = None,
    use_constructive_fallback: bool = False,
) -> SearchResult:
    """Run A* with f(n) = g(n) + h(n)."""
    return _search(
        problem,
        heuristic,
        use_path_cost=True,
        max_expansions=max_expansions,
        use_constructive_fallback=use_constructive_fallback,
    )

def run_informed_search_experiments(
    problem: Problem,
    max_expansions: int | None = None,
    use_constructive_fallback: bool = False,
) -> dict[str, SearchResult]:
    """Run both search algorithms with each of the available heuristics.

    The result keys identify the algorithm and heuristic used, making the
    returned mapping suitable for tabular comparison in a notebook.
    """
    experiments = (
        (
            "greedy_request_chain",
            greedy_best_first_search,
            heuristic_request_chain,
        ),
        (
            "greedy_next_required_location",
            greedy_best_first_search,
            heuristic_next_required_location,
        ),
        (
            "a_star_request_chain",
            a_star_search,
            heuristic_request_chain,
        ),
        (
            "a_star_next_required_location",
            a_star_search,
            heuristic_next_required_location,
        ),
    )

    return {
        name: algorithm(
            problem,
            heuristic,
            max_expansions=max_expansions,
            use_constructive_fallback=use_constructive_fallback,
        )
        for name, algorithm, heuristic in experiments
    }


def informed_search_results_to_records(
    results: dict[str, SearchResult],
) -> list[dict[str, object]]:
    """Convert experiment results into rows suitable for a comparison table."""
    records = []
    for experiment, result in results.items():
        if experiment.startswith("greedy_"):
            algorithm = "greedy"
            heuristic = experiment.removeprefix("greedy_")
        elif experiment.startswith("a_star_"):
            algorithm = "a_star"
            heuristic = experiment.removeprefix("a_star_")
        else:
            raise ValueError(f"Unknown informed-search experiment: {experiment}")
        records.append(
            {
                "experiment": experiment,
                "algorithm": algorithm,
                "heuristic": heuristic,
                "found": result.found,
                "cost_km": result.cost,
                "elapsed_seconds": result.elapsed_seconds,
                "expanded_nodes": result.expanded_nodes,
                "generated_nodes": result.generated_nodes,
                "stopped_by_limit": result.stopped_by_limit,
                "used_constructive_fallback": result.used_constructive_fallback,
            }
        )
    return records
