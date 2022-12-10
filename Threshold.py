class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __str__(self):
        return f"{self.x}, {self.y}"

    def get_tuple(self) -> (int, int):
        return self.x, self.y


class Vector2d:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y


class Line:
    def __init__(self, point_start: Point, point_end: Point):
        self.point_start = point_start
        self.point_end = point_end


class TrackedObject:
    def __init__(self, identifier: int, classification_id: int, point: Point):
        self.identifier = identifier
        self.classification_id = classification_id
        self.point = point
        self.age = 0

    def increment_age(self):
        self.age += 1


class Threshold:
    def __init__(self, threshold_line: Line, max_age: int, outside_direction: int = 1):
        self.threshold: Line = threshold_line
        self.max_age = max_age
        self.outside_direction: int = outside_direction / abs(outside_direction) if outside_direction != 0 else 1

        self.tracked_objects: dict[int, TrackedObject] = {}

    @staticmethod
    def get_box_center(xy, wh) -> Point:
        x = int(((wh[0] - xy[0]) / 2) + xy[0])
        y = int(((wh[1] - xy[1]) / 2) + xy[1])
        return Point(x, y)

    @staticmethod
    def __vectorize(point1: Point, point2: Point) -> Vector2d:
        return Vector2d(point1.x - point2.x, point1.y - point2.y)

    @staticmethod
    def __cross_product(vector1: Vector2d, vector2: Vector2d) -> int:
        return vector1.x * vector2.y - vector1.y * vector2.x

    def check_threshold(self, identifier: int, classification_id: int, current_object_center: Point) -> (bool, int, Point, Point):
        tracked_object: TrackedObject = self.tracked_objects.get(identifier)

        if tracked_object is None:
            self.tracked_objects[identifier] = TrackedObject(identifier, classification_id, current_object_center)
            return False, 0, None, current_object_center

        path_of_object = Line(tracked_object.point, current_object_center)

        direction_factor = self.__is_outside(current_object_center)
        has_crossed = self.__intersect(self.threshold, path_of_object)

        tracked_object.point = current_object_center
        return has_crossed, direction_factor, tracked_object.point, current_object_center

    # Source: https://www.geeksforgeeks.org/direction-point-line-segment/
    def __is_outside(self, point: Point) -> int:
        direction = self.__direction(self.threshold.point_start, self.threshold.point_end, point)
        # return positive or negative 1 depending on the cross product, modified by the outside direction setting
        return (direction / abs(direction) if direction != 0 else 1) * self.outside_direction

    # checks if p lies on the segment p1p2
    @staticmethod
    def __lies_on_line(point: Point, line: Line):
        return min(line.point_start.x, line.point_end.x) <= point.x <= max(line.point_start.x, line.point_end.x) \
               and min(line.point_start.y, line.point_end.y) <= point.y <= max(line.point_start.y, line.point_end.y)

    def __direction(self, point1: Point, point2: Point, point3: Point) -> int:
        return self.__cross_product(self.__vectorize(point1, point2), self.__vectorize(point3, point2))

    # https://algorithmtutor.com/Computational-Geometry/Check-if-two-line-segment-intersect/
    def __intersect(self, threshold: Line, path: Line) -> bool:
        direction1 = self.__direction(threshold.point_start, threshold.point_end, path.point_start)
        direction2 = self.__direction(threshold.point_start, threshold.point_end, path.point_end)
        direction3 = self.__direction(path.point_start, path.point_end, threshold.point_start)
        direction4 = self.__direction(path.point_start, path.point_end, threshold.point_end)

        if ((direction1 > 0 > direction2) or (direction1 < 0 < direction2)) and \
                ((direction3 > 0 > direction4) or (direction3 < 0 < direction4)):
            return True

        elif direction1 == 0 and self.__lies_on_line(path.point_start, threshold):
            return True
        elif direction2 == 0 and self.__lies_on_line(path.point_end, threshold):
            return True
        elif direction3 == 0 and self.__lies_on_line(threshold.point_start, path):
            return True
        elif direction4 == 0 and self.__lies_on_line(threshold.point_end, path):
            return True
        else:
            return False

    def __remove_identifier(self, identifier):
        if identifier in self.tracked_objects:
            self.tracked_objects.pop(identifier)

    def increment_ages(self):
        for identifier in list(self.tracked_objects):
            tracked_object = self.tracked_objects.get(identifier)
            if tracked_object.age > self.max_age:
                self.tracked_objects.pop(identifier)
                continue

            tracked_object.increment_age()
