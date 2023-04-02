from numpy import random
import cv2
from Threshold import Point


class OverlayPlotter:
    def __init__(self, line_thickness: int, center_thickness: int, class_names: []):
        self.main_font = cv2.FONT_HERSHEY_SIMPLEX
        self.metrics_font_scale = 0.5
        self.fps_position = (0, 15)
        self.metrics_position = (0, 30)
        self.metrics_font_color = (255, 255, 255)
        self.metrics_border_color = (0, 0, 0)
        self.metrics_font_thickness = 1
        self.line_thickness = line_thickness
        self.threshold_count_thickness = line_thickness + 2
        self.threshold_color = (255, 0, 0)
        self.threshold_count_embark_color = (0, 128, 0)
        self.threshold_count_disembark_color = (0, 0, 255)
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
        # Line thickness of 2 px
        thickness = 1
        # Text boarder in black
        cv2.putText(image, "fps: " + str(fps),
                    self.fps_position,
                    self.main_font,
                    self.metrics_font_scale,
                    self.metrics_border_color,
                    self.metrics_font_thickness + 1,
                    cv2.LINE_AA)
        cv2.putText(image, "fps: " + str(fps),
                    self.fps_position,
                    self.main_font,
                    self.metrics_font_scale,
                    self.metrics_font_color,
                    self.metrics_font_thickness,
                    cv2.LINE_AA)

    def plot_detection_metrics(self, image, frames_processed, frames_with_detection):
        # Text boarder in black
        cv2.putText(image, "frames processed: " + str(frames_processed),
                    self.metrics_position,
                    self.main_font,
                    self.metrics_font_scale,
                    self.metrics_border_color,
                    self.metrics_font_thickness + 1,
                    cv2.LINE_AA)
        cv2.putText(image, "frames processed: " + str(frames_processed),
                    self.metrics_position,
                    self.main_font,
                    self.metrics_font_scale,
                    self.metrics_font_color,
                    self.metrics_font_thickness,
                    cv2.LINE_AA)

        # Text boarder in black
        cv2.putText(image, "frames missed detection: " + str(frames_with_detection),
                    (self.metrics_position[0], self.metrics_position[1] + 15),
                    self.main_font,
                    self.metrics_font_scale,
                    self.metrics_border_color,
                    self.metrics_font_thickness + 1,
                    cv2.LINE_AA)
        cv2.putText(image, "frames missed detection: " + str(frames_with_detection),
                    (self.metrics_position[0], self.metrics_position[1] + 15),
                    self.main_font,
                    self.metrics_font_scale,
                    self.metrics_font_color,
                    self.metrics_font_thickness,
                    cv2.LINE_AA)

    def plot_threshold(self, start_point: (int, int), end_point: (int, int), img, direction=0):
        thickness = self.line_thickness
        color = self.threshold_color

        if direction > 0:
            thickness = self.threshold_count_thickness
            color = self.threshold_count_embark_color

        if direction < 0:
            thickness = self.threshold_count_thickness
            color = self.threshold_count_disembark_color

        cv2.line(img, start_point, end_point, color, thickness)
