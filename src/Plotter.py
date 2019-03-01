import cv2
import numpy as np

class Plotter:
    """!
    Helper class to prepare plots of keypoints and matches.
    """

    def plot_segmentation_results(self, image, labeled_image, color_contour, color_mask):
        """!
        Visualizes the segmentation result by drawing the contour of the instruments on the image,
        as well as the outline of the geretaed mask of points to exclude from the feature search.

        @param image The original image, will be used to display the contours and masks.
        @param labeled_image The resulting image after the segmentation.
        @param color_contours Color Triple (blue, green, red) with values from 0 to 255. The conoturs will be drawn in this color.
        @param color_mask Color Triple (blue, green, red) with values from 0 to 255. The masks will be drawn in this color.
        @return The orignal image with the conoturs and the masks drawn onto.
        """

        if (not isinstance(color_contour, tuple)) or (len(color_contour) != 3) or (not all(isinstance(x, int) for x in color_contour)) or (not all(x >= 0 and x <= 255 for x in color_contour)):
            raise ValueError('Color_contour needs to be of format (b, g, r) with integer values from 0 to 255.')

        if (not isinstance(color_mask, tuple)) or (len(color_mask) != 3) or (not all(isinstance(x, int) for x in color_mask)) or (not all(x >= 0 and x <= 255 for x in color_mask)):
            raise ValueError('Color_contour needs to be of format (b, g, r) with integer values from 0 to 255.')


        edges = cv2.Canny(labeled_image, 10, 100)
        kernel = np.ones((7,7),np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        bounding_boxes = []
        for i in range(len(contours)):
            box = cv2.minAreaRect(contours[i])
            bounding_boxes.append(box)
            vertices = cv2.boxPoints(box)
            vertices = np.int0(vertices)
            cv2.drawContours(image, contours[i], -1, color_contour)
            cv2.drawContours(image, [vertices], 0, color_mask, 2)

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
