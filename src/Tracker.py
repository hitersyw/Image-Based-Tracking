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

    def __init__(self, reference_image, comparison_image):
        """!
        The constructor. Takes two images of same size and same number of channels.

        @param reference_image The image used as reference image in the comparison.
        @param comparison_image The image used as comparison image.
        """
        assert reference_image.shape == comparison_image.shape
        if reference_image is None:
            raise ValueError('reference_image is not an image')
        if comparison_image is None:
            raise ValueError('new_image is not an image')
        self.reference_image = reference_image
        self.comparison_image = comparison_image

    def preprocess(self):
        """!
        Preprocesses the given images and returns them.

        @return reference_image: the preprocessed reference image
        @return comparison_image: the preprocessed comparison image
        """
        # convert images to greyscale
        self.reference_image = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2GRAY)
        self.comparison_image = cv2.cvtColor(self.comparison_image, cv2.COLOR_BGR2GRAY)

        # erode with (5, 5) kernel images to remove lighting artefacts
        kernel = (5, 5)
        self.reference_image = cv2.erode(self.reference_image, kernel)
        self.comparison_image = cv2.erode(self.comparison_image, kernel)

        # histogramm equalization
        self.reference_image = cv2.equalizeHist(self.reference_image)
        self.comparison_image = cv2.equalizeHist(self.comparison_image)
        return (self.reference_image, self.comparison_image)

    def semantic_segmentation(self, image):
        """!
        Semantically segments the given image. Currently only segmentation for surgical instruments is supported.
        If you have a different model trained you have to change the parameters in the Predictor.predict() call.
        We use googles DeepLabV3+ as model.
        If you want to change the net see @link Predictor @endlink for further information.

        @param image The image to segment.
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
        # Get label from semantic segmentation
        label = self.semantic_segmentation(image)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        kernel = np.ones((7,7),np.uint8)

        edges = cv2.Canny(label, 10, 100)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        bounding_boxes = []
        for i in range(len(contours)):
            box = cv2.minAreaRect(contours[i])
            bounding_boxes.append(box)
            vertices = cv2.boxPoints(box)
            vertices = np.int0(vertices)
            mask = cv2.fillConvexPoly(mask, vertices, 255)

        return cv2.bitwise_not(mask)


    def extract_and_match(self, segmentation=False):
        """!
        Extracts keypoints and their descriptors in both images and tries to find matches.
        If no keypoints were extracted in either one or both images, keypoints_reference_image, keypoints_comparison_image and matches will be empty.
        If no matches were found, matches is empty.

        @param segmentation optional parameter. If set to true, the image is preprocessed to exclude parts of the image from the keypoint search. Currenty only supports semantic segmentation for surgical instruments.
        @return keypoints_reference_image: A list of the extracted keypoints in the reference image of OpenCV type <a href="https://docs.opencv.org/4.0.1/d2/d29/classcv_1_1KeyPoint.html">KeyPoint</a>
        @return keypoints_comparison_image: A list of the extracted keypoints in the comparison image of OpenCV type <a href="https://docs.opencv.org/4.0.1/d2/d29/classcv_1_1KeyPoint.html">KeyPoint</a>
        @return matches: A list of the found matches of OpenCV type <a href="https://docs.opencv.org/4.0.1/d4/de0/classcv_1_1DMatch.html">DMatch</a>
        """
        orb = cv2.ORB_create()
        mask_reference = None
        mask_comparison = None
        if segmentation:
            mask_reference = self.__extract_mask(self.reference_image)
            mask_comparison = self.__extract_mask(self.comparison_image)
        keypoints1, descriptors1 = orb.detectAndCompute(self.reference_image, mask_reference)
        keypoints2, descriptors2 = orb.detectAndCompute(self.comparison_image, mask_comparison)

        if not keypoints1:
            warnings.warn('No keypoints found in the reference image, try passing an image with more distinct features.')
            return ([], [], [])

        if not keypoints2:
            warnings.warn('No keypoints found in the comparison image, try passing an image with more distinct features.')
            return ([], [], [])

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.matches = bf.match(descriptors1, descriptors2)

        if not self.matches:
            warnings.warn('No matches found.')

        self.__matches_reference = np.zeros((len(self.matches), 2), dtype=np.float32)
        self.__matches_comparison = np.zeros((len(self.matches), 2), dtype=np.float32)

        for i, match in enumerate(self.matches):
          self.__matches_reference[i, :] = keypoints1[match.queryIdx].pt
          self.__matches_comparison[i, :] = keypoints2[match.trainIdx].pt

        return (keypoints1, keypoints2, self.matches)

    def compute_affine_transform(self):
        """!
        Compute the affine transform between the first image and the second image.

        @return model: The computet affine transformation from reference_image to comparison_image.
        @return mask: Binary mask where 1 indicates an inlier to the found model.
        """
        retval, inliers	= cv2.estimateAffine2D(self.__matches_reference, self.__matches_comparison, None, method=cv2.RANSAC, ransacReprojThreshold = 10)
        return (retval, inliers)

    def track(self, segmentation=False):
        """!
        Method combining all tracking steps. Use this if you only care for the transformation matrix and not the steps on the way.

        @param segmentation optional parameter. If set to true, the image is preprocessed to exclude parts of the image from the keypoint search. Currenty only supports semantic segmentation for surgical instruments.
        @return model: The computet affine transformation from reference_image to comparison_image.
        @return mask: Binary mask where 1 indicates an inlier to the found model.
        """
        self.preprocess()
        self.extract_and_match(segmentation)
        return self.calculate_affine_transform()

    """def find_models(self):
        #berechne homography aus allen matches
        #entferne alle inlier aus den matches
        #checke ob noch genug matches vorhanden sind (> 50)
        #wenn nicht: gebe [(matches, model, mask)] zurück
        retval = []
        while len(self.matches) > 50:
            model, mask = self.calculate_homography()
            if len(model) == 0:
                break
            matches = self.matches
            count_inliers = np.count_nonzero(mask)
            # TODO: What value makes sense?
            if count_inliers >= 10:
                retval.append((matches, model, mask))

            # update internal matches such that only outliers of the calculated model are left
            matches_temp = self.matches[0:1]
            matches_ref_temp = self.matches_reference[0:1]
            matches_new_temp = self.matches_new[0:1]
            for i in range(len(mask)):
                if mask[i] == 0:
                    matches_temp.append(self.matches[i])
                    matches_ref_temp = np.concatenate((matches_ref_temp, [self.matches_reference[i]]))
                    matches_new_temp = np.concatenate((matches_new_temp, [self.matches_new[i]]))
            self.matches = matches_temp[2:]
            self.matches_reference = matches_ref_temp[2:]
            self.matches_new = matches_new_temp[2:]

        return retval"""


    """def extract_and_match_seperated(self, fac):
        # width and height sind vertauscht
        width, height = self.reference.shape
        factor_width = width // fac
        factor_height = height // fac

        ref = self.reference[0:factor_width, 0:factor_height]
        new = self.new[0:factor_width, 0:factor_height]
        orb = cv2.ORB_create()
        keypoints1, descriptors1 = orb.detectAndCompute(ref, None)
        keypoints2, descriptors2 = orb.detectAndCompute(new, None)
        kp1 = keypoints1
        dp1 = descriptors1
        kp2 = keypoints2
        dp2 = descriptors2

        for i in range(fac):
            for j in range(fac):
                mask = np.zeros((height, width), dtype=np.uint8)
                rec = cv2.rectangle(mask, (j * factor_width, i * factor_height), (j * factor_width + factor_width, i * factor_height + factor_height), (255), thickness = -1)
                orb = cv2.ORB_create()
                keypoints1 = orb.detect(self.reference, mask)
                keypoints2 = orb.detect(self.new, mask)
                kp1.extend(keypoints1)
                kp2.extend(keypoints2)
                _, descriptors1 = orb.compute(self.reference, keypoints1)
                # this seems to run into an issue, havent figured out what yet
                dp1 = np.concatenate((dp1, descriptors1))
                _, descriptors2 = orb.compute(self.new, keypoints2)
                dp2 = np.concatenate((dp2, descriptors2))

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(dp1, dp2)

        self.matches_reference = np.zeros((len(matches), 2), dtype=np.float32)
        self.matches_new = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
          self.matches_reference[i, :] = kp1[match.queryIdx].pt
          self.matches_new[i, :] = kp2[match.trainIdx].pt

        return (kp1, dp1, kp2, dp2, matches)"""
