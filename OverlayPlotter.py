from numpy import random
import cv2
from Threshold import Point


class OverlayPlotter:
    def __init__(self, line_thickness: int, center_thickness: int, class_names: []):
        self.line_thickness = line_thickness
        self.center_thickness = center_thickness
        self.class_names = class_names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]

    def plot_one_box(self, box, center, img, class_id: int, id_detection: int, confidence_score):
        # Plots one bounding box on image
        color = self.colors[class_id]
        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))

        # Plot box
        cv2.rectangle(img, c1, c2, color, thickness=self.line_thickness, lineType=cv2.LINE_AA)

        # Plot center
        cv2.circle(img, center, radius=self.center_thickness, color=color, thickness=self.center_thickness)

        # Plot label
        label = f'{id_detection} {self.class_names[class_id]} {confidence_score:.2f}'
        font_thickness = max(self.line_thickness - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=self.line_thickness / 3, thickness=font_thickness)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled

        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, self.line_thickness / 3, [225, 255, 255],
                    thickness=font_thickness, lineType=cv2.LINE_AA)

    def plot_fps(self, image, fps):
        font = cv2.FONT_HERSHEY_SIMPLEX
        # org
        org = (0, 20)
        # fontScale
        font_scale = 0.5
        # Blue color in BGR
        color = (255, 255, 255)
        # Line thickness of 2 px
        thickness = 1
        cv2.putText(image, str(fps), org, font, font_scale, color, thickness, cv2.LINE_AA)

    def plot_line(self, start_point: (int, int), end_point: (int, int), img, color=(0, 69, 255)):
        cv2.line(img, start_point, end_point, color, self.line_thickness)
