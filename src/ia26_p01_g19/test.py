from datetime import datetime
from pathlib import Path

from .model import ActionType, Problem, Stop


def route_tests():
    print("Loading Problem")
    data_dir = Path(__file__).resolve().parents[1] / "data"
    prob = Problem.from_loader(data_dir=data_dir)

    # Example data from the dataset
    if not prob.vehicles or not prob.requests:
        print(
            "No vehicles or requests found in the dataset. Please ensure the data is loaded correctly."
        )
        return

    vehicle_id = list(prob.vehicles.keys())[0]
    vehicle = prob.vehicles[vehicle_id]

    # Select two requests for testing
    request_ids = list(prob.requests.keys())
    req1 = prob.requests[request_ids[0]]
    req2 = (
        prob.requests[request_ids[1]]
        if len(request_ids) > 1
        else prob.requests[request_ids[0]]
    )

    print(f"Test Vehicle: {vehicle_id} (Capacity: {vehicle.capacity})")
    print(f"Request 1: {req1.item_id} | Demand: {req1.demand}")
    print(f"Request 2: {req2.item_id} | Demand: {req2.demand}")
    print("=" * 50)

    # Case 1
    # Feasible route: Pickup and then Delivery for req1
    print("\n Test 1: Feasible Route")
    valid_stops = [
        Stop(
            request_id=req1.item_id,
            action=ActionType.PICKUP,
            location_id=req1.pickup_location_id,
        ),
        Stop(
            request_id=req1.item_id,
            action=ActionType.DELIVERY,
            location_id=req1.delivery_location_id,
        ),
    ]

    is_ok, reason = prob.is_route_feasible(
        vehicle_id, valid_stops, departure_time=req1.pickup_start
    )
    cost = prob.cost_function({vehicle_id: valid_stops})
    print(f"Expected Result: Valid (True)")
    print(f"Result:   {is_ok} | Reason: {reason} | Cost: {cost:.2f} km")

    # Case 2
    # Violation of precedence: Delivery before Pickup
    print("\n Test 2: Violation of Precedence (Delivery before Pickup)")
    invalid_precedence = [
        Stop(
            request_id=req1.item_id,
            action=ActionType.DELIVERY,
            location_id=req1.delivery_location_id,
        ),
        Stop(
            request_id=req1.item_id,
            action=ActionType.PICKUP,
            location_id=req1.pickup_location_id,
        ),
    ]

    is_ok, reason = prob.is_route_feasible(
        vehicle_id, invalid_precedence, departure_time=req1.pickup_start
    )
    print(f"Expected Result: Invalid (False)")
    print(f"Result:   {is_ok} | Reason: {reason}")

    # case 3: Violation of Time Window (Late Arrival)
    print("\n Test 3: Violation of Time Window (Late Arrival)")

    # Departure time is set after the pickup window of req1 1 year
    late_departure = datetime(2030, 1, 1, 12, 0, 0)
    is_ok, reason = prob.is_route_feasible(
        vehicle_id, valid_stops, departure_time=late_departure
    )
    print(f"Expected Result: Invalid (False - late arrival/time window closed)")
    print(f"Result:   {is_ok} | Reason: {reason}")

    # Case 4: Vehicle Capacity Violation
    print("\n Test 4: Vehicle Capacity Violation")

    original_capacity = vehicle.capacity
    vehicle.capacity = 0.001

    is_ok, reason = prob.is_route_feasible(
        vehicle_id, valid_stops, departure_time=req1.pickup_start
    )
    print(f"Expected Result: Invalid (False - capacity exceeded)")
    print(f"Result:   {is_ok} | Reason: {reason}")

    vehicle.capacity = original_capacity

    # Case 5: Incomplete Route (Picked up but not delivered)
    print("\n Test 5: Incomplete Route (Picked up but not delivered)")
    incomplete_stops = [
        Stop(
            request_id=req1.item_id,
            action=ActionType.PICKUP,
            location_id=req1.pickup_location_id,
        )
    ]

    is_ok, reason = prob.is_route_feasible(
        vehicle_id, incomplete_stops, departure_time=req1.pickup_start
    )
    print(f"Expected Result: Invalid (False - pending items)")
    print(f"Result:   {is_ok} | Reason: {reason}")


if __name__ == "__main__":
    route_tests()
