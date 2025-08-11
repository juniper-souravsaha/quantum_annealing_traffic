# Road network configuration with travel time (minutes) and capacity (vehicles)
# This will be used for shortest path and congestion simulation.

ROAD_NETWORK = [
    ("A", "B", {"time": 5, "capacity": 100}),
    ("B", "C", {"time": 7, "capacity": 80}),
    ("C", "D", {"time": 4, "capacity": 120}),
    ("A", "D", {"time": 10, "capacity": 150}),
    ("B", "D", {"time": 3, "capacity": 60}),
    ("A", "C", {"time": 6, "capacity": 90}),
]