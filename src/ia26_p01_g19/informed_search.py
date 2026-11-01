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


def _search(
    problem: Problem,
    heuristic: Heuristic,
    *,
    use_path_cost: bool,
    max_expansions: int | None,
) -> SearchResult:
    started = perf_counter()
    initial = initial_state(problem)
    frontier: list[tuple[float, int, State]] = []
    sequence = count()
    heappush(frontier, (heuristic(problem, initial), next(sequence), initial))
    best_cost = {_state_key(initial): initial.cost}
    expanded_nodes = 0
    generated_nodes = 1

    while frontier:
        _, _, state = heappop(frontier)
        if _is_goal(problem, state):
            solution = {
                vehicle_id: list(route)
                for vehicle_id, route in zip(_vehicle_ids(problem), state.routes)
            }
            return SearchResult(
                solution=solution,
                cost=state.cost,
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
            heappush(frontier, (priority, next(sequence), child))

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
) -> SearchResult:
    """Run Greedy Best-First Search with f(n) = h(n)."""
    return _search(
        problem,
        heuristic,
        use_path_cost=False,
        max_expansions=max_expansions,
    )


def a_star_search(
    problem: Problem,
    heuristic: Heuristic,
    max_expansions: int | None = None,
) -> SearchResult:
    """Run A* with f(n) = g(n) + h(n)."""
    return _search(
        problem,
        heuristic,
        use_path_cost=True,
        max_expansions=max_expansions,
    )

