import cv2

import src.waveguide_entrance_detection.waveguide_entrance_detector as WaveguideEntranceDetector


def main(image_path):
    # Load and preprocess the image

    image = cv2.imread(image_path)

    image_gu = WaveguideEntranceDetector.prepare_image(image)

    horizontal_edge_chip, vertical_edge_chip = WaveguideEntranceDetector.detect_edges(image_gu)

    ROI_image, ROI_bbox, ROI_corners, intersection_edge_point = WaveguideEntranceDetector.get_ROI(horizontal_edge_chip,
                                                                                                  vertical_edge_chip,
                                                                                                  image_gu)

    left_vertical_line, right_vertical_line = WaveguideEntranceDetector.detect_waveguide_lines(ROI_image)

    left_OS_vertical_line, right_OS_vertical_line, waveguide_entrance_coordinates = WaveguideEntranceDetector.get_waveguide_entrance(
        horizontal_edge_chip, left_vertical_line, right_vertical_line, ROI_bbox)

    image_with_drawings = WaveguideEntranceDetector.draw_all_on_image(image_gu, horizontal_edge_chip,
                                                                      vertical_edge_chip, intersection_edge_point,
                                                                      ROI_corners,
                                                                      [left_OS_vertical_line, right_OS_vertical_line],
                                                                      waveguide_entrance_coordinates)

    WaveguideEntranceDetector.visualize_image(image_with_drawings)


if __name__ == '__main__':
    image_path = "../data/foto1.png"
    main(image_path)
