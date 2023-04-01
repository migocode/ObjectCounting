from Threshold import Threshold, Point
from typing import List


class TrackedObject:
    def __init__(self, identifier: int, classification_id: int, point: Point):
        self.identifier: int = identifier
        self.classification_id: int = classification_id
        self.direction: int = 0
        self.center: Point = point
        self.age: int = 0

    def increment_age(self):
        self.age += 1


class ObjectCounter:
    def __init__(self, thresholds: List[Threshold], max_age: int = 30):
        self.max_age: int = max_age
        self.thresholds: List[Threshold] = thresholds
        self.tracked_objects: dict[int, TrackedObject] = {}

    def check(self, identifier: int, classification_id: int, current_object_center: Point) -> int:
        current_object: TrackedObject = TrackedObject(identifier, classification_id, current_object_center)

        if (identifier in self.tracked_objects) is False:
            self.tracked_objects[identifier] = current_object
            return 0

        tracked_object = self.tracked_objects.pop(identifier)

        direction_metric = self.__check_thresholds(tracked_object, current_object)
        current_object.direction = direction_metric
        current_object.age = 0

        self.tracked_objects[identifier] = current_object
        return direction_metric

    def __check_thresholds(self, tracked_object: TrackedObject, current_object: TrackedObject) -> int:
        # If ANY thresholds are crossed, return direction information
        for threshold in self.thresholds:
            return threshold.check(tracked_object.center, current_object.center)
        return 0

    def increment_ages(self):
        for identifier in list(self.tracked_objects):
            tracked_object = self.tracked_objects.get(identifier)
            if tracked_object.age > self.max_age:
                self.tracked_objects.pop(identifier)
                continue

            tracked_object.increment_age()
