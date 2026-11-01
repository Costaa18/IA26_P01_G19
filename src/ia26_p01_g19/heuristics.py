from .model import Problem, State


def _distance(problem: Problem, origin: str, destination: str) -> float:
    if origin == destination:
        return 0.0
    route = problem.distance_matrix.get((origin, destination))
    if route is None:
        raise ValueError(f"Route between {origin} and {destination} not found.")
    return float(route["distance_km"])


def heuristic_request_chain(problem: Problem, state: State) -> float:
    """Admissible lower bound using pickup-to-delivery chains."""
    vehicle_count = len(problem.vehicles)
    picked = set().union(*state.picked_by_vehicle)
    lower_bounds: list[float] = []

    for request_id, request in problem.requests.items():
        if request_id in state.delivered_requests:
            continue

        candidates = []
        for vehicle_index in range(vehicle_count):
            location = state.current_locations[vehicle_index]
            if request_id in state.picked_by_vehicle[vehicle_index]:
                candidates.append(
                    _distance(problem, location, request.delivery_location_id)
                )
            elif request_id not in picked:
                candidates.append(
                    _distance(problem, location, request.pickup_location_id)
                    + _distance(
                        problem,
                        request.pickup_location_id,
                        request.delivery_location_id,
                    )
                )

        if candidates:
            lower_bounds.append(min(candidates))

    return max(lower_bounds, default=0.0)


def heuristic_next_required_location(problem: Problem, state: State) -> float:
    """Admissible lower bound using only the next required location."""
    picked = set().union(*state.picked_by_vehicle)
    lower_bounds: list[float] = []

    for request_id, request in problem.requests.items():
        if request_id in state.delivered_requests:
            continue

        target = (
            request.delivery_location_id
            if request_id in picked
            else request.pickup_location_id
        )
        lower_bounds.append(
            min(
                _distance(problem, location, target)
                for location in state.current_locations
            )
        )

    return max(lower_bounds, default=0.0)
