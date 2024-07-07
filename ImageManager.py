import cv2
import math
import numpy as np
import pydicom as dicom
from skimage import exposure

font = cv2.FONT_HERSHEY_SIMPLEX


# Method used to show an image in a resizable window
def show_image(window_name, image):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)


# Method used to update the image in a windows that has associated an event handler
def update_image_window(points_list, counter, image_draw):
    if len(points_list) > counter:
        counter = len(points_list)
        cv2.circle(image_draw, (points_list[-1][0], points_list[-1][1]), 3, [0, 0, 255], -1)
        labelClickedPixel(image_draw, points_list[-1][0], points_list[-1][1])
    return counter


# Method used to handle the click event in an image
def labelClickedPixel(target_image, x, y):
    cv2.putText(target_image, str(x) + ',' +
                str(y), (x, y), font,
                1, (255, 0, 0), 2)


# Method used to compute the angle between two lines
def angle(dx1, dy1, dx2, dy2):
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180 / math.pi)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180 / math.pi)

    if angle1 * angle2 >= 0:
        inside_angle = abs(angle1 - angle2)
    else:
        inside_angle = abs(angle1) + abs(angle2)
        if inside_angle > 180:
            inside_angle = 360 - inside_angle
    inside_angle = inside_angle % 180
    return inside_angle


# Method used to define the axis of the femur
def axis(ex1, ey1, ex2, ey2, maxy):
    m = ((ey2 - ey1) / (ex2 - ex1) + 1e-8)
    b = int(ey1 - (m * ex1))
    x1 = int((0 - b) / m)
    x2 = int((maxy - b) / m)
    return x1, 0, x2, maxy, m, b


# Method used to identify the femur head and the circle that best describes it
def femur_head(img):
    chosen_x = 0
    chosen_y = 0
    chosen_r = 0
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 150,
                               param1=50, param2=55, minRadius=130, maxRadius=300)

    if circles is None:
        print("No circles where found to define femur head")
        return chosen_x, chosen_y, chosen_r, None, None

    circles = np.uint16(np.around(circles))
    # Choose the circle with shortest radio
    for i in circles[0, :]:
        if chosen_r < i[2]:
            chosen_x = i[0]
            chosen_y = i[1]
            chosen_r = i[2]
        cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 2)
        cv2.circle(img, (i[0], i[1]), 2, (255, 0, 0), 22)
    h, w = img.shape
    eje_x = chosen_x
    eje_y = chosen_y
    eje_x2 = None
    eje_y2 = None

    foreground = np.ones((h, w), np.uint8)
    final_foreground = np.ones((h, w), np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    edges = cv2.Canny(opening, 30, 120, apertureSize=5)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=40, maxLineGap=9)

    if lines is None:
        print("No lines were found to define femur radius")
        return

    for line in lines:
        x1, y1, x2, y2 = line[0]
        pend = ((y2 - y1) / ((x2 - x1) + 1e-8))
        cv2.line(foreground, (x1, y1), (x2, y2), (255, 0, 0), 1, cv2.LINE_AA)
        if pend < -1:
            if x1 < chosen_x and y1 > chosen_y:
                if x1 < eje_x:
                    eje_x = x1
                    eje_y = y1
                    eje_x2 = x2
                    eje_y2 = y2

    cv2.line(final_foreground, (eje_x, eje_y), (eje_x2, eje_y2), (255, 0, 0), 1, cv2.LINE_AA)
    axis_slope = ((eje_y2 - eje_y) / ((eje_x2 - eje_x) + 1e-8))
    axis_b_cut = int(eje_y - (eje_x2 * axis_slope))

    return chosen_x, chosen_y, chosen_r, axis_slope, axis_b_cut


# Class used to process all the actions that can be executed in the program
class ImageManager:

    # Init method, used to clear all the class attributes
    def __init__(self):
        self.image = None
        self.height_2 = None
        self.seed_points = []
        self.image_type = None
        self.shape_org_X = None
        self.shape_org_Y = None
        self.pixel_spacing = None
        self.right_oriented = False
        self.bounding_box_points = []

    # Method used to load the image found in the image_path parameter, the method takes into account if the image
    # that is going to be loaded is a DICOM image or a normal image
    def set_image(self, image_path):
        if "dcm" in image_path:
            dicom_img = dicom.dcmread(image_path)
            pixel_array = dicom_img.pixel_array
            pixel_array = exposure.equalize_adapthist(pixel_array)
            self.pixel_spacing = getattr(dicom_img, 'PixelSpacing')
            self.image = pixel_array
            self.image_type = "dicom"
        else:
            self.image = cv2.imread(image_path)
            self.image_type = "standard"
        if self.image is not None:
            print("Image loaded successfully")
        else:
            print("Image was not loaded correctly, please check image name and try again")

    # Method used to transform the image loaded in a grayscale image
    def create_gray_image(self):
        gray_image = None
        if self.image_type == 'dicom':
            gray_image = np.uint8(self.image * 255)
        elif self.image_type == 'standard':
            if len(self.image.shape) == 3:
                gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = self.image

        return gray_image

    # Method used to identify the lines that best describe the bone
    def find_femur_lines_and_angle(self, img, rad_x, rad_y, pendiente_eje, bias_eje, res_image):
        h, w = img.shape
        true_height = self.shape_org_Y + self.height_2
        kernel = np.ones((5, 5), np.uint8)

        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        edges = cv2.Canny(opening, 30, 140, apertureSize=5)
        lines = cv2.HoughLinesP(edges, 1.7, np.pi / 180, 100, minLineLength=30, maxLineGap=15)

        if lines is None:
            print("No lines where found")
            return

        foreground = np.ones((h, w), np.uint8)
        background = np.ones((h, w), np.uint8)
        f_x1, f_y1 = 99999, 99999
        axis_slope = None
        for line in lines:
            x1, y1, x2, y2 = line[0]
            pend = ((y2 - y1) / (x2 - x1) + 1e-8)
            if pend > 0:
                if x1 < f_x1 and y1 < f_y1:
                    f_x1, f_y1, f_x2, f_y2 = x1, y1, x2, y2
                    axis_slope = pend

        if axis_slope is None:
            print("No lines were found with the specific slope")
            return

        obtb = int(f_y1 - (axis_slope * f_x1))
        best_candidate = 0
        min_dist_face = 20
        max_dist_face = 90

        int_axe_femur_x = None
        int_axe_femur_y = None
        int_axe_femur_x2 = None
        int_axe_femur_y2 = None

        for candidate in lines:
            x1, y1, x2, y2 = candidate[0]
            pend = ((y2 - y1) / (x2 - x1) + 1e-8)
            efx1 = int((y1 - obtb)/axis_slope)
            dist = x1 - efx1
            if pend > 0:
                if min_dist_face < dist < max_dist_face:
                    if dist > best_candidate:
                        int_axe_femur_x = x1
                        int_axe_femur_y = y1
                        int_axe_femur_x2 = x2
                        int_axe_femur_y2 = y2
                    if x1 < int_axe_femur_x:
                        f_x1, f_y1, f_x2, f_y2 = x1, y1, x2, y2
                        axis_slope = pend

        lines_dist = cv2.HoughLinesP(edges, 1.7, np.pi / 180, 100, minLineLength=40, maxLineGap=16)

        if lines_dist is None:
            print("No lines where founded to define edges")
            return

        # average of human femur transverse diameter is 32.5 +- 2.3, lower 30,2
        close_point = int(30.2 / self.pixel_spacing[0])
        max_dist = 999
        min_distance = None
        y_open = None
        x_open = None
        min_dist_x = None
        min_dist_y = None
        for candidate in lines_dist:
            x1, y1, x2, y2 = candidate[0]
            pend = ((y2 - y1) / ((x2 - x1) + 1e-8))
            efx1 = int((y1 - obtb)/axis_slope)
            dist = x1 - efx1
            if pend < 0:
                if min_dist_face < dist:
                    if dist < close_point:
                        close_point = dist
                        min_dist_x = x1
                        min_dist_y = y1
                        min_distance = dist

            # According to the femur arc, we can consider that the upper section is going to have the maximum
            # intramedular diameter, meanwhile the lower section is going to have the minimum intramedular diameter
            if y1 < (h/2):
                if max_dist_face < dist:
                    cv2.line(foreground, (x1, y1), (x2, y2), (255, 0, 0), 1, cv2.LINE_AA)
                    if dist < max_dist:
                        max_dist = dist
                        y_open = y1
                        x_open = x1
        fe_y2 = int(axis_slope * (f_x1 + 330) + obtb)
        cv2.line(background, (f_x1, f_y1), ((f_x1 + 330), fe_y2), (255, 0, 0), 1, cv2.LINE_AA)
        cv2.line(res_image,
                 (f_x1 + self.shape_org_X, f_y1 + self.shape_org_Y),
                 (((f_x1 + 330) + self.shape_org_X), fe_y2 + self.shape_org_Y),
                 (255, 0, 0), 5, cv2.LINE_AA)

        # Validate that all parameters required to compute the femur axis where found
        if int_axe_femur_x is None or int_axe_femur_y is None or int_axe_femur_x2 is None or int_axe_femur_y2 is None:
            print("No enough parameters to compute femur axis")
            return

        dx1, dy1, dx2, dy2, dm, db = axis(int_axe_femur_x, int_axe_femur_y, int_axe_femur_x2, int_axe_femur_y2, w)

        cv2.line(res_image,
                 (dx1 + self.shape_org_X, dy1 + true_height),
                 ((dx2 + self.shape_org_X), dy2 + true_height),
                 (255, 0, 0), 5, cv2.LINE_AA)

        # compute maximum distance
        x_cut = int((y_open - db) / dm)
        cv2.line(res_image,
                 (x_cut + self.shape_org_X, y_open + true_height),
                 ((x_open + self.shape_org_X), y_open + true_height),
                 (255, 0, 0), 5, cv2.LINE_AA)

        if min_distance is None:
            print("No minimum distance was found in the femur")
            return

        # compute minimum distance
        print("Maximum intrafemoral distance: ", max_dist * self.pixel_spacing[0])
        print("Minimum intrafemoral distance: ", min_distance * self.pixel_spacing[0])

        if min_dist_y is None or min_dist_x is None:
            print("No minimum and maximum distances where found")
            return

        x_cut = int((min_dist_y - db) / dm)
        cv2.line(res_image, (min_dist_x + self.shape_org_X, min_dist_y + true_height),
                 ((x_cut + self.shape_org_X),
                  min_dist_y + true_height),
                 (255, 0, 0), 5, cv2.LINE_AA)

        cv2.line(background, (min_dist_x, min_dist_y), ((f_x1 + 10), min_dist_y), (255, 0, 0), 1, cv2.LINE_AA)

        eje_neck_y = rad_y + self.shape_org_Y
        eje_neck_x = int((true_height - pendiente_eje) / bias_eje)

        cv2.line(res_image,
                 (eje_neck_x, eje_neck_y + self.height_2),
                 ((rad_x + self.shape_org_X), eje_neck_y),
                 (255, 0, 0), 5, cv2.LINE_AA)

        ax1 = eje_neck_x - (dx2 + self.shape_org_X)
        ay1 = (eje_neck_y + self.height_2) - eje_neck_y
        ax2 = (dx1 + self.shape_org_X) - (dx2 + self.shape_org_X)
        ay2 = (dy1 + true_height) - (dy2 + true_height)

        # Find the angle between radius axis and femur axis
        femur_angle = angle(ax1, ay1, ax2, ay2)
        print("Angle between radius axis and femur axis: ", femur_angle)

    # Method used to process the image that is loaded in the program
    def process_image(self):
        if self.image is not None:
            cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
            gray_image = self.create_gray_image()
            image_draw = np.copy(gray_image)
            cv2.setMouseCallback("Image", self.process_click_event, image_draw)

            self.seed_points = []
            print("Please choose two points to crop the interest region")

            # First of all, crop image in the interest region
            # Ask for the start and end points for the rectangle
            bounding_box_counter = 0
            while len(self.bounding_box_points) < 2:
                cv2.imshow("Image", image_draw)
                cv2.waitKey(1)
                bounding_box_counter = update_image_window(self.bounding_box_points, bounding_box_counter, image_draw)

            labelClickedPixel(image_draw, self.seed_points[-1][0], self.seed_points[-1][1])
            cv2.rectangle(image_draw, self.bounding_box_points[0], self.bounding_box_points[1], (255, 0, 0), 5)

            # Crop the image to the rectangle defined
            x, y = self.bounding_box_points[0]
            width = self.bounding_box_points[1][0] - x
            height = self.bounding_box_points[1][1] - y

            # Crop for split head from femur(body)
            height_2 = int(height / 2)

            self.shape_org_X = self.bounding_box_points[0][0]
            self.shape_org_Y = self.bounding_box_points[0][1]
            self.height_2 = height_2

            cropped_image = gray_image[y: y + height + 1, x: x + width + 1]

            cropped_image_head = gray_image[y: y + height_2 + 1, x: x + width + 1]
            cropped_image_femur = cropped_image[height_2: height, 0: width - 1]

            self.bounding_box_points = []
            self.seed_points = []

            cv2.destroyWindow("Image")

            show_image("cropped_image_femur", cropped_image_femur)
            show_image("cropped_image_head", cropped_image_head)

            print("Processing image.....")

            # Apply morphological transforms
            threshold = 5
            kernel = np.ones((5, 5), np.uint8)

            # First of all, apply opening to interest region
            opening_head = cv2.morphologyEx(cropped_image_head, cv2.MORPH_OPEN, kernel)
            opening_femur = cv2.morphologyEx(cropped_image_femur, cv2.MORPH_OPEN, kernel)

            # Then, compute the image gradient to detect femur borders
            gradient_head = cv2.morphologyEx(opening_head, cv2.MORPH_GRADIENT, kernel)
            gradient_femur = cv2.morphologyEx(opening_femur, cv2.MORPH_GRADIENT, kernel)

            # Threshold
            # Binary gradient image using otsu
            thresh_head = cv2.threshold(gradient_head, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            thresh_femur = cv2.threshold(gradient_femur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            # Reduce noise in binary gradient
            noise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            de_noise_head = cv2.morphologyEx(thresh_head, cv2.MORPH_OPEN, noise_kernel, iterations=1)
            de_noise_femur = cv2.morphologyEx(thresh_femur, cv2.MORPH_OPEN, noise_kernel, iterations=1)

            # Then, find contours
            cont_head = cv2.findContours(de_noise_head, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cont_femur = cv2.findContours(de_noise_femur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cont_femur = cont_femur[0] if len(cont_femur) == 2 else cont_femur[1]
            cont_head = cont_head[0] if len(cont_head) == 2 else cont_head[1]

            # Show borders that accomplish the threshold criteria
            for c in cont_head:
                area = cv2.contourArea(c)
                if area < threshold:
                    cv2.drawContours(de_noise_head, [c], -1, 0, -1)

            for c in cont_femur:
                area = cv2.contourArea(c)
                if area < threshold:
                    cv2.drawContours(de_noise_femur, [c], -1, 0, -1)

            # Invert results
            de_noise_femur = 255 - de_noise_femur
            de_noise_head = 255 - de_noise_head

            res_image = np.copy(gray_image)

            rad_x, rad_y, rad_r, axis_slope, axis_bias = femur_head(de_noise_head)

            if axis_slope is not None and axis_bias is not None:
                cv2.circle(res_image, (rad_x + self.shape_org_X, rad_y + self.shape_org_Y), rad_r, (255, 0, 0), 2)
                cv2.circle(res_image, (rad_x + self.shape_org_X, rad_y + self.shape_org_Y), 10, (255, 0, 0), 22)

                self.find_femur_lines_and_angle(de_noise_femur, rad_x, rad_y, axis_slope, axis_bias, res_image)
                show_image("Results", res_image)
            else:
                print("The program could not process the femur correctly")

            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Image is empty. Please, first load an image")

    # Method used to delete the image loaded in memory
    def unload_image(self):
        if self.image is not None:
            self.image = None
            self.image_type = None

        else:
            print("Current image is empty")

    # Method to handle a click event in the image window
    def process_click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.seed_points.append((x, y))
            self.bounding_box_points.append((x, y))
