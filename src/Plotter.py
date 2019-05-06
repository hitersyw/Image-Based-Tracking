import cv2
import numpy as np

class Plotter:
    """!
    Helper class to prepare plots of keypoints and matches.
    """

    def plot_segmentation_results(self, image, labeled_image, color_contour, thickness):
        """!
        Visualizes the segmentation result by drawing the contour of the instruments on the image.

        @param image The original image, will be used to display the contours and masks.
        @param labeled_image The resulting image after the segmentation.
        @param color_contours Color Triple (blue, green, red) with values from 0 to 255. The contours will be drawn in this color.
        @param thickness The thickness of the contours.
        @return The orignal image with the conoturs drawn onto.
        """

        if (not isinstance(color_contour, tuple)) or (len(color_contour) != 3) or (not all(isinstance(x, int) for x in color_contour)) or (not all(x >= 0 and x <= 255 for x in color_contour)):
            raise ValueError('Color_contour needs to be of format (b, g, r) with integer values from 0 to 255.')

        label = cv2.cvtColor(labeled_image, cv2.COLOR_BGR2GRAY)

        # depending on your OpenCV version this function call differs
        if cv2.__version__.startswith('4'):
            contours, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:
            _, contours, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for i in range(len(contours)):
            cv2.drawContours(image, contours[i], -1, color_contour, thickness=thickness)

        return image

    def plot_keypoints(self, image, keypoints, radius, color):
        """!
        Draws the given keypoints onto the given image in the given color.

        @param image: The image to draw the keypoints on.
        @param keypoints: The keypoints to draw.
        @param radius: The radius of the keypoints.
        @param color: Color Triple (blue, green, red) with values from 0 to 255. The keypoints will be drawn in this color.

        @return image_with_keypoints: returns the given image with the keypoints drawn on it
        """
        if image is None:
            raise ValueError('The first argument needs to be an image.')


        if (not isinstance(color, tuple)) or (len(color) != 3) or (not all(isinstance(x, int) for x in color)) or (not all(x >= 0 and x <= 255 for x in color)):
            raise ValueError('Color needs to be of format (b, g, r) with integer values from 0 to 255.')

        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # does not work currently since OpenCV 4.0.1 seems to have an issue with it, should be fixed with ne next release however
        # colorImage = cv2.drawKeypoints(colorImage, keypoints, colorImage)
        for kp in keypoints:
          x, y = kp.pt
          cv2.circle(image, (int(x), int(y)), radius, color)
        return image

    def plot_matches(self, image1, keypoints1, image2, keypoints2, matches, color_matches, color_keypoints, mask=None, color_outliers=None):
        """!
        Draws the given keypoints and matches onto the given images.
        An optional mask determines what keypoints to draw and if color_outliers is set additionaly the mask will be treated as inliers and outliers and will be drawn in different colors.

        @param image1 The first image
        @param keypoints1 The keypoints of the first image
        @param image2 The second image
        @param keypoints2 The keypoints of the second image.
        @param matches The matches to plot
        @param color_matches: Color Triple (blue, green, red) with values from 0 to 255. Matches will be drawn in this color.
        @param color_keypoints Color Triple (blue, green, red) with values from 0 to 255. Keypoints will be drawn in this color.
        @param mask optional paremeter. Mask determining which matches are drawn. Has to be of same size as matches.
        @param color_outliers optional parameter. Color Triple (blue, green, red) with values from 0 to 255. Requires mask to be set as well. The masks values will be handled as inliers and will be drawn in color_matches and the outliers will be drawn in color_outliers.

        @return image_with_matches: returns image of size image1.width + image2.width with keypoints and matches drawn according to parameters
        """
        if image1 is None:
            raise ValueError('Image1 is not a valid image.')

        if image2 is None:
            raise ValueError('Image2 is not a valid image.')

        if (not isinstance(color_matches, tuple)) or (len(color_matches) != 3) or (not all(isinstance(x, int) for x in color_matches)) or (not all(x >= 0 and x <= 255 for x in color_matches)):
            raise ValueError('Color_matches needs to be of format (b, g, r) with integer values from 0 to 255.')

        if (not isinstance(color_keypoints, tuple)) or (len(color_keypoints) != 3) or (not all(isinstance(x, int) for x in color_keypoints)) or (not all(x >= 0 and x <= 255 for x in color_keypoints)):
            raise ValueError('Color_keypoints needs to be of format (b, g, r) with integer values from 0 to 255.')

        if mask is None and color_outliers is not None:
            raise ValueError('If color_outliers is set, mask needs to be set as well.')

        # Plot all matches
        if mask is None and color_outliers is None:
            return cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, color_matches, color_keypoints)

        if len(mask) != len(matches):
            raise ValueError('Mask and matches need to be of same length.')

        if mask is not None and color_outliers is None:
            return cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, color_matches, color_keypoints, mask)

        if (not isinstance(color_outliers, tuple)) or (len(color_outliers) != 3) or (not all(isinstance(x, int) for x in color_outliers)) or (not all(x >= 0 and x <= 255 for x in color_outliers)):
            raise ValueError('Color_outliers needs to be of format (b, g, r) with integer values from 0 to 255.')
        # Plot inliers and outliers in different colors
        inliers = mask.ravel().tolist()
        outliers = inliers.copy()
        for i in range(len(outliers)):
            if outliers[i] == 0:
                outliers[i] = 1
            else:
                outliers[i] = 0
        out = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, color_matches, color_keypoints, inliers)
        return cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, out, color_outliers, color_keypoints, outliers, 1)


    def plot_matches_one_image(self, image, keypoints1, keypoints2, matches, mask, color_keypoints, color_inliers, color_outliers=None):
        """!
        Draws the given keypoints and matches onto one image.

        @param image The image to draw onto
        @param keypoints1 The keypoints of the first image
        @param keypoints2 The keypoints of the second image.
        @param matches The matches to plot
        @param mask  Mask determining which matches are drawn. Has to be of same size as matches.
        @param color_keypoints Color Triple (blue, green, red) with values from 0 to 255. Keypoints will be drawn in this color.
        @param color_inliers Color Triple (blue, green, red) with values from 0 to 255. Inliers will be drawn in this color.
        @param color_outliers optional paramter Color Triple (blue, green, red) with values from 0 to 255. Outliers will be drawn in this color if set. Otherwise the outliers are not plotted.

        @return image_with_matches: returns the image with keypoints and matches drawn according to parameters
        """
        if image is None:
            raise ValueError('Image is not a valid image.')

        if (not isinstance(color_inliers, tuple)) or (len(color_inliers) != 3) or (not all(isinstance(x, int) for x in color_inliers)) or (not all(x >= 0 and x <= 255 for x in color_inliers)):
            raise ValueError('Color_matches needs to be of format (b, g, r) with integer values from 0 to 255.')

        if color_outliers is not None:
            if  (not isinstance(color_outliers, tuple)) or (len(color_outliers) != 3) or (not all(isinstance(x, int) for x in color_outliers)) or (not all(x >= 0 and x <= 255 for x in color_outliers)):
                raise ValueError('Color_matches needs to be of format (b, g, r) with integer values from 0 to 255.')

        if (not isinstance(color_keypoints, tuple)) or (len(color_keypoints) != 3) or (not all(isinstance(x, int) for x in color_keypoints)) or (not all(x >= 0 and x <= 255 for x in color_keypoints)):
            raise ValueError('Color_keypoints needs to be of format (b, g, r) with integer values from 0 to 255.')

        if mask is None:
            raise ValueError('Mask may not be None')

        for i in range(len(matches)):
            m = matches[i]
            (x1, y1) = keypoints1[m.queryIdx].pt
            x1 = int(x1)
            y1 = int(y1)
            (x2, y2) = keypoints2[m.trainIdx].pt
            x2 = int(x2)
            y2 = int(y2)
            cv2.circle(image, (x2, y2), 5, color_keypoints)
            if mask[i]:
                cv2.line(image, (x1, y1), (x2, y2), color_inliers)
            else:
                if color_outliers is not None:
                    cv2.line(image, (x1, y1), (x2, y2), color_outliers)

        return image
