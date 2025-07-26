SUPER_CLASS_MAP = {
    "vehicle": ["LongVehicle", "Car", "PoliceCar"],
    "pedestrian": ["Child", "RoadWorker", "Pedestrian", "Scooter",
                   "ScooterRider", "Motorcycle", "MotorcyleRider",
                   "BicycleRider"],
    "truck": ["Truck", "Van", "TrashCan", "ConcreteTruck", "Bus"],
}     # this is used by V2X-Real dataset to map the class names to super classes