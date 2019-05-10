import cv2
import numpy as np
from skimage import measure
from skimage import transform as tf
import warnings

from .Segmentation import Predictor

class Tracker:
    """!
    This class can be used to track a scene by providing two images of the scene.
    It uses the OpenCV implementation of ORB (Oriented FAST and Rotated BRIEF) for keypoint and descriptor extraction
    and the scikit-image implementation of RANSAC (Random Sample Consensus) for computing the transformation.
    """

    def __init__(self, segmentation):
        """!
        Initializes the Tracker. Segmentation option determines whether to load the semantic segmentation model.
        Currently only segmentation for surgical instruments is supported.
        If you have a different model trained you have to change the parameters in the Predictor.predict() call.
        We use Googles DeepLabV3+ as model.
        If you want to change the trained network see @link Predictor @endlink for further information.

        @param segmentation True, if the tracker should consider labels from semantic segmentations during the tracking process.
        """
        self.__pred = None
        if segmentation:
            # change this if you want to use a different neural net
            # first argument is the name of the used model (see the Segmantation Suite repo for available models)
            # provide the path to the trained net in the second argument
            self.__pred = Predictor.Predictor('DeepLabV3_plus', './src/Segmentation/trained_net/latest_model_DeepLabV3_plus_Surgery.ckpt')
        self.__segmentation = segmentation


    def preprocess(self, image):
        """!
        Preprocesses the given image and returns a mask with lighting dependent artifacts marked. If the class is called with segmentation=True
        all found surgical instruments are marked as well. The input image needs to be a color image.

        @param image The image to preprocess.
        @return The black and white mask for the given image.
        """
        if image is None:
            raise ValueError("The image may not be None.")
        # detect lighting dependent artifacts
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        height, width, _ = image_hsv.shape
        mask = np.zeros((height, width), np.uint8)

        highest_intensity = np.amax(image_hsv[2])
        lowest_saturation = np.amax(image_hsv[1])

        upper_intensity_thresh = 0.8 * highest_intensity
        lower_saturation_thresh = 0.2 * lowest_saturation

        mask[np.where(image_hsv[:, :, 2] > upper_intensity_thresh) and np.where(image_hsv[:, :, 1] < lower_saturation_thresh)] = 255

        if self.__segmentation:
            label = self.semantic_segmentation(image)
            label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
            label[np.nonzero(label)] = 255
            mask = mask | label

        kernel = np.ones((5, 5))
        mask = cv2.dilate(mask, kernel)
        return cv2.bitwise_not(mask)

    def semantic_segmentation(self, image):
        """!
        Semantically segments the given image.
        Currently only segmentation for surgical instruments is supported.
        If you have a different model trained you have to change the parameters in the Predictor.predict() call.
        We use Googles DeepLabV3+ as model.
        If you want to change the trained network see @link Predictor @endlink for further information.

        @param image The image to segment. Must be a color image.
        """
        if image is None:
            raise ValueError("The image may not be None.")
        # ensure the neural net is initialized
        if not self.__pred:
            self.__pred = Predictor.Predictor('DeepLabV3_plus', './src/Segmentation/trained_net/latest_model_DeepLabV3_plus_Surgery.ckpt')
        label = self.__pred.predict(image)
        return label


    def extract_and_match(self, reference_image, comparison_image):
        """!
        Extracts keypoints and their descriptors in the given images and tries to find matches between these keypoints.
        The images need to be color images and need to be of same size.
        If no keypoints were found in either one or both images, keypoints_reference_image, keypoints_comparison_image and matches will be empty.
        If no matches were found, matches is empty.

        @param reference_image The first image.
        @param comparison_image The second image.
        @return keypoints_reference_image A list of the extracted keypoints in the reference image of OpenCV type <a href="https://docs.opencv.org/4.0.1/d2/d29/classcv_1_1KeyPoint.html">KeyPoint</a>
        @return keypoints_comparison_image A list of the extracted keypoints in the comparison image of OpenCV type <a href="https://docs.opencv.org/4.0.1/d2/d29/classcv_1_1KeyPoint.html">KeyPoint</a>
        @return matches A list of the found matches of OpenCV type <a href="https://docs.opencv.org/4.0.1/d4/de0/classcv_1_1DMatch.html">DMatch</a>
        """
        if reference_image is None:
            raise ValueError('reference_image is not a valid image.')

        if comparison_image is None:
            raise ValueError('comparison_image is not a valid image.')

        orb = cv2.ORB_create(2000)
        mask_reference = self.preprocess(reference_image)
        mask_comparison = self.preprocess(comparison_image)

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
        Compute the affine transform between the first image and the second image given the keypoints and the found matches.

        @param keypoints_reference_image The keypoints found in the first image. Must be of OpenCV type <a href="https://docs.opencv.org/4.0.1/d2/d29/classcv_1_1KeyPoint.html">KeyPoint</a>. May not be empty.
        @param keypoints_comparison_image The keypoints found in the second image. Must be of OpenCV type <a href="https://docs.opencv.org/4.0.1/d2/d29/classcv_1_1KeyPoint.html">KeyPoint</a>. May not be empty.
        @param matches The found matches between the two keypoint sets. Must be of OpenCV type <a href="https://docs.opencv.org/4.0.1/d4/de0/classcv_1_1DMatch.html">DMatch</a>. May not be empty.

        @return model The computed affine transformation matrix.
        @return mask Binary mask where True indicates an inlier to the found model.
        """
        if not keypoints_reference_image:
            raise ValueError('keypoints_reference_image may not be empyt.')

        if not keypoints_comparison_image:
            raise ValueError('keypoints_comparison_image may not be empyt.')

        if not matches:
            raise ValueError('matches may not be empty.')

        matches_reference = np.zeros((len(matches), 2), dtype=np.float32)
        matches_comparison = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
          matches_reference[i, :] = keypoints_reference_image[match.queryIdx].pt
          matches_comparison[i, :] = keypoints_comparison_image[match.trainIdx].pt

        model, mask = measure.ransac((matches_reference, matches_comparison), tf.AffineTransform, min_samples=3,residual_threshold=10, max_trials=600)
        return model.params, mask

    def track(self, reference_image, comparison_image):
        """!
        Method combining all tracking steps. Use this if you only care for the transformation matrix and not the steps on the way.
        The images need to be color images and need to be of same size.

        @param reference_image The first image.
        @param comparison_image The second image.

        @return model The computed affine transformation from the first image to the second image.
        @return mask Binary mask where 1 indicates an inlier to the found model.
        """
        if reference_image is None:
            raise ValueError('reference_image is not a valid image.')

        if comparison_image is None:
            raise ValueError('comparison_image is not a valid image.')

        k1, k2, m = self.extract_and_match(reference_image, comparison_image)
        return self.compute_affine_transform(k1, k2, m)
