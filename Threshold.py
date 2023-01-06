class Point:
    def __init__(self, x: int, y: int):
        self.x: int = x
        self.y: int = y

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __eq__(self, other) -> bool:
        return self.x == other.x and self.y == other.y

    def __ne__(self, other) -> bool:
        return self.x != other.x or self.y != other.y

    def get_tuple(self) -> (int, int):
        return self.x, self.y


class Vector2d:
    def __init__(self, x: int, y: int):
        self.x: int = x
        self.y: int = y


class Line:
    def __init__(self, point_start: Point, point_end: Point):
        self.point_start: Point = point_start
        self.point_end: Point = point_end


class Threshold:
    def __init__(self, threshold_line: Line, outside_direction: int = 1):
        self.threshold: Line = threshold_line
        self.outside_direction: int = outside_direction / abs(outside_direction) if outside_direction != 0 else 1

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

    def check(self, current_object_center: Point, tracked_object_center: Point) -> int:
        # If no movement has occurred, nothing needs to be calculated
        if current_object_center == tracked_object_center:
            return 0

        path_of_object = Line(tracked_object_center, current_object_center)
        has_crossed = self.__get_crossing_info(self.threshold, path_of_object)
        return has_crossed * self.outside_direction

    # Source: https://www.geeksforgeeks.org/direction-point-line-segment/
    def __is_outside(self, point: Point) -> int:
        direction = self.__direction(self.threshold.point_start, self.threshold.point_end, point)
        # return positive or negative 1 depending on the cross product, modified by the outside direction setting
        return (direction / abs(direction) if direction != 0 else 1) * self.outside_direction

    # checks if p lies on the segment line
    @staticmethod
    def __is_within_line_bounds(point: Point, line: Line):
        return min(line.point_start.x, line.point_end.x) <= point.x <= max(line.point_start.x, line.point_end.x) \
               and min(line.point_start.y, line.point_end.y) <= point.y <= max(line.point_start.y, line.point_end.y)

    def __direction(self, point1: Point, point2: Point, point3: Point) -> int:
        return self.__cross_product(self.__vectorize(point1, point2), self.__vectorize(point3, point2))

    # https://algorithmtutor.com/Computational-Geometry/Check-if-two-line-segment-intersect/
    # 0 if not crossed, positive or negative if crossed depending on direction of crossing
    # (according to rules of the cross product)
    def __get_crossing_info(self, threshold: Line, path: Line) -> int:
        relation_threshold_to_path_start = self.__direction(threshold.point_start,
                                                            threshold.point_end,
                                                            path.point_start)
        relation_threshold_to_path_end = self.__direction(threshold.point_start,
                                                          threshold.point_end,
                                                          path.point_end)
        relation_path_to_threshold_start = self.__direction(path.point_start,
                                                            path.point_end,
                                                            threshold.point_start)
        relation_path_to_threshold_end = self.__direction(path.point_start,
                                                          path.point_end,
                                                          threshold.point_end)

        # If any direction is 0, that means the points involved in its calculation
        # are collinear (are located on the same line)
        if ((relation_threshold_to_path_start > 0 > relation_threshold_to_path_end)
            or (relation_threshold_to_path_start < 0 < relation_threshold_to_path_end)) \
                and ((relation_path_to_threshold_start > 0 > relation_path_to_threshold_end)
                     or (relation_path_to_threshold_start < 0 < relation_path_to_threshold_end)):
            return relation_threshold_to_path_end
        # For the crossing to occur, the start point of the path has to be positioned ON the threshold line
        # therefore WITHIN the threshold line boundaries (start point and end point)
        elif relation_threshold_to_path_start == 0 and self.__is_within_line_bounds(path.point_start, threshold):
            return relation_threshold_to_path_end
        elif relation_threshold_to_path_end == 0 and self.__is_within_line_bounds(path.point_end, threshold):
            return relation_threshold_to_path_end
        elif relation_path_to_threshold_start == 0 and self.__is_within_line_bounds(threshold.point_start, path):
            return relation_threshold_to_path_end
        elif relation_path_to_threshold_end == 0 and self.__is_within_line_bounds(threshold.point_end, path):
            return relation_threshold_to_path_end
        else:
            return 0
