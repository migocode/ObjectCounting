from numpy import random
import cv2


class OverlayPlotter:
    def __init__(self, line_thickness: int, class_names: []):
        self.line_thickness = line_thickness
        self.class_names = class_names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]

    def plot_one_box(self, box, center, img, class_id: int, id_detection: int, confidence_score):
        # Plots one bounding box on image
        color = self.colors[class_id]
        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))

        # Plot box
        cv2.rectangle(img, c1, c2, color, thickness=self.line_thickness, lineType=cv2.LINE_AA)

        # Plot center
        cv2.circle(img, center, radius=self.line_thickness, color=color, thickness=self.line_thickness)

        # Plot label
        label = f'{id_detection} {self.class_names[class_id]} {confidence_score:.2f}'
        font_thickness = max(self.line_thickness - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=self.line_thickness / 3, thickness=font_thickness)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled

        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, self.line_thickness / 3, [225, 255, 255],
                    thickness=font_thickness, lineType=cv2.LINE_AA)

    def plot_threshold(self, start_point, end_point, img, color=(0, 69, 255)):
        cv2.line(img, start_point, end_point, color, self.line_thickness)
