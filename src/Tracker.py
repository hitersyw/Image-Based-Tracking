import cv2
import numpy as np
import warnings

from .Segmentation import Predictor


class Tracker:
    """!
    This class can be used to track a scene by providing two images of the scene.
    It uses the OpenCV implementation of ORB (Oriented FAST and Rotated BRIEF) for keypoint and descriptor extraction
    and the OpenCV implementation of RANSAC (Random Sample Consensus) for computing the transformation.
    """

    def preprocess(self, image):
        """!
        Preprocesses the given image and returns it.

        @param image The image to preprocess.
        @return The preprocessed image.
        """
        # convert images to greyscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # erode with (5, 5) kernel images to remove lighting artefacts
        kernel = (5, 5)
        image = cv2.erode(image, kernel)

        # histogramm equalization
        image = cv2.equalizeHist(image)
        return image

    def semantic_segmentation(self, image):
        """!
        Semantically segments the given image.
        Currently only segmentation for surgical instruments is supported.
        If you have a different model trained you have to change the parameters in the Predictor.predict() call.
        We use googles DeepLabV3+ as model.
        If you want to change the net see @link Predictor @endlink for further information.

        @param image The image to segment. Must be a color image.
        """
        pred = Predictor.Predictor()
        label = pred.predict(image, 'DeepLabV3_plus', './src/Segmentation/trained_net/latest_model_DeepLabV3_plus_Surgery.ckpt')
        return label


    def __extract_mask(self, image):
        """!
        Generates the mask for the feature detector. The mask determines what area(s) to exclude from the feature search.
        The mask is based on the semantic segmentation of the image which labels surgical instruments.
        For other non rigid tracking scenes out of surgeries, this method can be adjusted or replaced.

        @param image The image the mask should be extracted for.
        """
        label = self.semantic_segmentation(image)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        kernel = np.ones((7,7),np.uint8)

        edges = cv2.Canny(label, 10, 100)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # depending on your OpenCV version this function call differs
        if cv2.__version__.startswith('4'):
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:
            _, contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        bounding_boxes = []
        for i in range(len(contours)):
            box = cv2.minAreaRect(contours[i])
            bounding_boxes.append(box)
            vertices = cv2.boxPoints(box)
            vertices = np.int0(vertices)
            mask = cv2.fillConvexPoly(mask, vertices, 255)

        return cv2.bitwise_not(mask)


    def extract_and_match(self, reference_image, comparison_image, segmentation=False):
        """!
        Extracts keypoints and their descriptors in the given images and tries to find matches between these keypoints.
        The images need to be color images and need to be of same size.
        If no keypoints were found in either one or both images, keypoints_reference_image, keypoints_comparison_image and matches will be empty.
        If no matches were found, matches is empty.

        @param reference_image The first image.
        @param comparison_image The second image.
        @param segmentation optional parameter. If set to true, the image is preprocessed to exclude parts of the image from the keypoint search. Currenty only supports semantic segmentation for surgical instruments.

        @return keypoints_reference_image A list of the extracted keypoints in the reference image of OpenCV type <a href="https://docs.opencv.org/4.0.1/d2/d29/classcv_1_1KeyPoint.html">KeyPoint</a>
        @return keypoints_comparison_image A list of the extracted keypoints in the comparison image of OpenCV type <a href="https://docs.opencv.org/4.0.1/d2/d29/classcv_1_1KeyPoint.html">KeyPoint</a>
        @return matches A list of the found matches of OpenCV type <a href="https://docs.opencv.org/4.0.1/d4/de0/classcv_1_1DMatch.html">DMatch</a>
        """
        orb = cv2.ORB_create()
        mask_reference = None
        mask_comparison = None
        if segmentation:
            mask_reference = self.__extract_mask(reference_image)
            mask_comparison = self.__extract_mask(comparison_image)

        reference_image = self.preprocess(reference_image)
        comparison_image = self.preprocess(comparison_image)

        keypoints1, descriptors1 = orb.detectAndCompute(reference_image, mask_reference)
        keypoints2, descriptors2 = orb.detectAndCompute(comparison_image, mask_comparison)

        if not keypoints1:
            warnings.warn('No keypoints found in the reference image, try passing an image with more distinct features.')
            return ([], [], [])

        if not keypoints2:
            warnings.warn('No keypoints found in the comparison image, try passing an image with more distinct features.')
            return ([], [], [])

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        matches = bf.match(descriptors1, descriptors2)

        if not matches:
            warnings.warn('No matches found.')

        return (keypoints1, keypoints2, matches)

    def compute_affine_transform(self, keypoints_reference_image, keypoints_comparison_image, matches):
        """!
        Compute the affine transform between the first image and the second image.

        @param keypoints_reference_image The keypoints found in the first image. Must be of OpenCV type <a href="https://docs.opencv.org/4.0.1/d2/d29/classcv_1_1KeyPoint.html">KeyPoint</a>.
        @param keypoints_comparison_image The keypoints found in the second image. Must be of OpenCV type <a href="https://docs.opencv.org/4.0.1/d2/d29/classcv_1_1KeyPoint.html">KeyPoint</a>.
        @param matches The found matches between the two keypoint sets. Must be of OpenCV type <a href="https://docs.opencv.org/4.0.1/d4/de0/classcv_1_1DMatch.html">DMatch</a>.

        @return model The computed affine transformation from the first image to the second image.
        @return mask Binary mask where 1 indicates an inlier to the found model.
        """
        matches_reference = np.zeros((len(matches), 2), dtype=np.float32)
        matches_comparison = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
          matches_reference[i, :] = keypoints_reference_image[match.queryIdx].pt
          matches_comparison[i, :] = keypoints_comparison_image[match.trainIdx].pt
        model, mask	= cv2.estimateAffine2D(matches_reference, matches_comparison, None, method=cv2.RANSAC, ransacReprojThreshold = 10)
        return (model, mask)

    def track(self, reference_image, comparison_image, segmentation=False):
        """!
        Method combining all tracking steps. Use this if you only care for the transformation matrix and not the steps on the way.
        The images need to be color images and need to be of same size.

        @param reference_image The first image.
        @param comparison_image The second image.
        @param segmentation optional parameter. If set to true, the image is preprocessed to exclude parts of the image from the keypoint search. Currenty only supports semantic segmentation for surgical instruments.

        @return model The computed affine transformation from the first image to the second image.
        @return mask Binary mask where 1 indicates an inlier to the found model.
        """
        k1, k2, m = self.extract_and_match(reference_image, comparison_image, segmentation)
        return self.calculate_affine_transform(k1, k2, m)
