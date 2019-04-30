import sys
import math
import cv2 as cv
import numpy as np

def main(argv):
    main_images_path = '../images/'
    #efault_file = main_images_path + 'Empty1.jpg'
    default_file = main_images_path + 'Empty2.jpg'
    #efault_file = main_images_path + 'Empty3.png'
    #efault_file = main_images_path + 'empty_lot.jpg'
    #default_file = main_images_path + 'parking_lot.png'

    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)
    # Check if image is loaded fine
    if src is None:
        print('Error opening image!')
        print('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1

    dst = cv.Canny(src, 250, 350, None, 3)
    # here i can change the value of the min and max gradients depending on the brightness of the image
    cv.imshow ("canny", dst)
    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)


    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 6, cv.LINE_AA)
            #print(l[0], "  ", l[1], "  ", l[2], "  ", l[3])
# cv.LINE_AA >> gives anti-aliased line which looks great for curves.
    cv.imshow("Source", src)
    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

    cv.waitKey()
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])