from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from .data_loader import DataLoader
from enum import Enum


@dataclass(frozen=True)
class Request:
    item_id: str
    order_id: str
    pickup_location_id: str
    delivery_location_id: str
    pickup_start: datetime
    pickup_end: datetime
    delivery_start: datetime
    delivery_end: datetime
    demand: float


@dataclass
class Vehicle:
    vehicle_id: str
    center_id: str
    capacity: float


@dataclass
class State:
    routes: dict[str, list[str]]
    current_time: dict[str, datetime]
    current_load: dict[str, float]
    pending_requests: set[str]


class ActionType(Enum):
    PICKUP = "PICKUP"
    DELIVERY = "DELIVERY"


@dataclass
class Stop:
    request_id: str  # OrderItem ID 
    action: ActionType  # PICKUP or DELIVERY
    location_id: str  # to look up in the distance matrix


class Problem:

    def __init__(
        self,
        vehicles: dict[str, Vehicle],
        requests: dict[str, Request],
        distance_matrix: dict[tuple[str, str], dict],
    ):
        self.vehicles = vehicles
        self.requests = requests
        self.distance_matrix = distance_matrix

    @classmethod
    def from_loader(
        cls, loader: DataLoader | None = None, data_dir: Path | str = "data"
    ) -> "Problem":
        """Initialize the problem from the DataLoader."""
        if loader is None:
            loader = DataLoader(data_dir)
        vehicles_df = loader.load_vehicles()
        order_items_df = loader.load_order_items()
        dist_df = loader.load_distance_matrix()

        # Create Vehicle for each one
        vehicles = {
            str(row["vehicle_id"]): Vehicle(
                vehicle_id=str(row["vehicle_id"]),
                center_id=str(row["center_id"]),
                capacity=float(row["capacity_kg"]),
            )
            for _, row in vehicles_df.iterrows()
        }

        # Create Request for each order item
        requests = {
            str(row["order_item_id"]): Request(
                item_id=str(row["order_item_id"]),
                order_id=str(row["order_id"]),
                pickup_location_id=str(row.get("producer_id", "")),
                delivery_location_id=str(row.get("delivery_location_id", "")),
                pickup_start=row["availability_start"],
                pickup_end=row["availability_end"],
                delivery_start=row["delivery_start"],
                delivery_end=row["delivery_end"],
                demand=float(row.get("weight", 1.0)),
            )
            for _, row in order_items_df.iterrows()
        }

        # Create a distance lookup dictionary for quick access
        dist_lookup = dist_df.set_index(["origin", "destination"])[
            ["distance_km", "duration_min"]
        ].to_dict(orient="index")

        return cls(
            vehicles=vehicles,
            requests=requests,
            distance_matrix=dist_lookup,
        )

    def get_travel_duration(self, origin_id: str, destination_id: str) -> timedelta:
        """Get the travel duration between two points as a timedelta.

        If the origin is the same as the destination, the duration is 0.
        """
        if origin_id == destination_id:
            return timedelta(minutes=0)

        route_info = self.distance_matrix.get((origin_id, destination_id))
        if not route_info:
            # If the route is not found, raise an error
            raise ValueError(
                f"Route between {origin_id} and {destination_id} not found."
            )

        duration_minutes = float(route_info["duration_min"])
        return timedelta(minutes=duration_minutes)

    def is_route_feasible(
        self, vehicle_id: str, stops: list[Stop], departure_time: datetime | None = None
    ) -> tuple[bool, str]:
        """
        Validate if a sequence of stops for a vehicle is feasible.
        Returns (True, "OK") if feasible, or (False, "reason") if it violates rules.
        """
        vehicle = self.vehicles.get(vehicle_id)
        if not vehicle:
            return False, f"Vehicle {vehicle_id} does not exist."

        # No stops is considered feasible 
        # vehicle doesn't leave the depot
        if not stops:
            return True, "No stops."

        # Start Point, warehouse of the vehicle
        current_location = vehicle.center_id
        current_load = 0.0

        # If departure_time is not provided, assume the availability time of the first item
        if departure_time is None:
            first_request = self.requests[stops[0].request_id]
            current_time = first_request.pickup_start
        else:
            current_time = departure_time

        # Establish sets to track picked up and delivered items
        picked_up_items = set()
        delivered_items = set()

        for idx, stop in enumerate(stops):
            req = self.requests.get(stop.request_id)
            if not req:
                return False, f"Stop {idx}: Request {stop.request_id} not found."

            # Calculate travel time from current location to the stop
            try:
                travel_duration = self.get_travel_duration(
                    current_location, stop.location_id
                )
            except ValueError as err:
                return False, f"Stop {idx}: Error in distance matrix - {err}"

            # Time of arrival at the stop
            arrival_time = current_time + travel_duration

            #  Process the stop based on its action type
            if stop.action == ActionType.PICKUP:
                # Cannot pick up the same item twice
                if stop.request_id in picked_up_items:
                    return (
                        False,
                        f"Stop {idx}: The item {stop.request_id} has already been picked up.",
                    )

                # Max Capacity
                if current_load + req.demand > vehicle.capacity:
                    return (
                        False,
                        f"Stop {idx}: Capacity exceeded. "
                        f"Load: {current_load + req.demand} > Capacity: {vehicle.capacity}",
                    )

                # Time Window for Pickup
                #If arrival_time is after the pickup window, it's infeasible
                if arrival_time > req.pickup_end:
                    return (
                        False,
                        f"Stop {idx}: Arrived at {arrival_time}, after the pickup window ({req.pickup_end}).",
                    )

                # If arrival_time is before the pickup window, wait until the start of the window
                start_service_time = max(arrival_time, req.pickup_start)

                # Update state 
                current_load += req.demand
                picked_up_items.add(stop.request_id)
                current_time = start_service_time

            elif stop.action == ActionType.DELIVERY:
                # Check if the item was picked up before attempting delivery
                if stop.request_id not in picked_up_items:
                    return (
                        False,
                        f"Stop {idx}: Attempt to deliver {stop.request_id} without previous pickup.",
                    )

                # Cannot deliver the same item twice
                if stop.request_id in delivered_items:
                    return (
                        False,
                        f"Stop {idx}: The item {stop.request_id} has already been delivered.",
                    )

                # Time Window for Delivery
                if arrival_time > req.delivery_end:
                    return (
                        False,
                        f"Stop {idx}: Arrived at {arrival_time}, after the delivery window ({req.delivery_end}).",
                    )

                start_service_time = max(arrival_time, req.delivery_start)

                # Update state
                current_load -= req.demand
                delivered_items.add(stop.request_id)
                current_time = start_service_time

            # Update the current location to the stop's location for the next iteration
            current_location = stop.location_id

        # After processing all stops, check if all picked up items have been delivered
        if picked_up_items != delivered_items:
            uncompleted = picked_up_items - delivered_items
            return False, f"Stop {idx}: The route ended without delivering the items: {uncompleted}"

        return True, "Route is valid."

    def cost_function(self, solution: dict[str, list[Stop]], return_to_depot: bool = False) -> float:
        """
        Calculate the total cost of a complete solution.
        
        Solution: dict {vehicle_id: [Stop1, Stop2, ...]}
        Returns: sum of the costs of all the routes.
        """
        total_solution_cost = 0.0

        for vehicle_id, stops in solution.items():
            total_solution_cost += self.route_cost(vehicle_id, stops, return_to_depot=return_to_depot)

        return total_solution_cost
