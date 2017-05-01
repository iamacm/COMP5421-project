#include "stdafx.h"
#include <windows.h>
#include <algorithm> 
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "ThreeDimReconstruction.h"

void ThreeDimReconstruction::FeatureDetectors::nonMaxSuppression(const Img src, Img& dst) {
	// find pixels that are equal to the local neighborhood not maximum (including 'plateaus')
	cv::dilate(src.mat, dst.mat, Mat());
	//cv::compare(src.mat, dst.mat, dst.mat, cv::CMP_GE);
}
vector<Point2d> ThreeDimReconstruction::FeatureDetectors::detectHarrisCorner(const Img src, bool showResult) {
	/// Result
	vector<Point2d> cornerPoints;
	/// Detector parameters
	const int blockSize = 4;
	const int apertureSize = 3;
	const float thresholdVal = 0.00005;
	const double k = 0.04;

	Img dst, dstNorm, srcGray, thresholdedBinary, thresholded, thresholdedDilated, thresholdedDilatedBinary, localMaxBinary;
	cvtColor(src.mat, srcGray.mat, CV_BGR2GRAY);

	const Size imageSize = srcGray.mat.size();

	//Mat dst_norm_scaled;
	dst.mat = Mat::zeros(imageSize, CV_32FC1);	// Matrix of 32-bit float
	thresholdedBinary.mat = Mat::zeros(imageSize, CV_8UC1);
	thresholded.mat = Mat::zeros(imageSize, CV_32FC1);	// Matrix of 32-bit float
	thresholdedDilated.mat = Mat::zeros(imageSize, CV_32FC1);	// Matrix of 32-bit float
	thresholdedDilatedBinary.mat = Mat::zeros(imageSize, CV_8UC1);
	localMaxBinary.mat = Mat::zeros(imageSize, CV_8UC3);	// BGR


															/// Detecting corners
	cornerHarris(srcGray.mat, dst.mat, blockSize, apertureSize, k, BORDER_DEFAULT);
	/// Normalizing for display
	normalize(dst.mat, dstNorm.mat, 0, 1, NORM_MINMAX, CV_32FC1, Mat());
	/// Thresholding to binary image
	//threshold(dst.mat, thresholdedBinary.mat, thresholdVal, 255, THRESH_BINARY);
	//threshold(dst.mat, thresholded.mat, thresholdVal, 255, THRESH_TOZERO);


	for (int i = 0; i < thresholdedBinary.mat.rows; ++i) {
		for (int j = 0; j < thresholdedBinary.mat.cols; ++j) {
			if (dst.mat.at<float>(i, j) >= thresholdVal) {
				thresholdedBinary.mat.at<uint8_t>(i, j) = 255;
				thresholded.mat.at<float>(i, j) = dst.mat.at<float>(i, j);
			}
		}
	}
	// Non max suppression
	cv::dilate(thresholded.mat, thresholdedDilated.mat, Mat());

	for (int i = 0; i < thresholdedBinary.mat.rows; ++i) {
		for (int j = 0; j < thresholdedBinary.mat.cols; ++j) {
			thresholdedDilatedBinary.mat.at<uint8_t>(i, j) = thresholdedDilated.mat.at<float>(i, j) > 0 ? 255 : 0;
			if (thresholdedBinary.mat.at<uint8_t>(i, j)) {
				bool isLocalMaxima = thresholded.mat.at<float>(i, j) >= thresholdedDilated.mat.at<float>(i, j);
				localMaxBinary.mat.at<Vec3b>(i, j)[1] = isLocalMaxima ? 255 : 0;
				localMaxBinary.mat.at<Vec3b>(i, j)[2] = isLocalMaxima ? 0 : 100;

				if (isLocalMaxima) {
					cornerPoints.push_back(Point2d(j, i));
				}

			}
		}
	}
	//ThreeDimReconstruction::FeatureDetectors::nonMaxSuppression(thresholded, localMaxBinary);
	//bitwise_and(thresholdedBinary.mat, localMaxBinary.mat, res);



	/// Drawing a circle around corners
	for (int j = 0; j < dstNorm.mat.rows; j++)
	{
		cout.precision(27);
		for (int i = 0; i < dstNorm.mat.cols; i++)
		{
			float val = dst.mat.at<float>(j, i);

		}
	}
	/// Showing the result
	if (showResult) {
		dstNorm.name = "dstNorm of image " + src.name;
		//dstNorm.show();

		thresholdedBinary.name = "Thresholded mask of image " + src.name;
		//thresholdedBinary.show();

		thresholdedDilatedBinary.name = "Thresholded dilated binary of image " + src.name;
		//thresholdedDilatedBinary.show();
		thresholdedBinary.showWith(thresholdedDilatedBinary);

		localMaxBinary.name = "localMax of image " + src.name;
		//localMaxBinary.show();
	}

	printf("%d corner(s) detected!\n", (int) cornerPoints.size());
	//namedWindow("corners_window", CV_WINDOW_AUTOSIZE);
	//imshow("corners_window", dst_norm_scaled);

	//dstNorm.mat = Mat(dst_norm_scaled);
	//waitKey(0);
	return cornerPoints;
}

vector<SIFTFeature> ThreeDimReconstruction::FeatureDetectors::detectSIFT(const Img src, bool showResult) {

	vector<SIFTFeature> features; // SIFT features to be returned

	Mat grayImg;
	cv::cvtColor(src.mat, grayImg, CV_BGR2GRAY);

	SiftFeatureDetector detector(
		0, // nFeatures
		4, // nOctaveLayers
		0.10, // contrastThreshold 0.04
		20, //edgeThreshold 10
		1.6 //sigma
	);
	SiftDescriptorExtractor extractor;

	vector<KeyPoint> keypoints;
	Mat descriptors;

	detector.detect(grayImg, keypoints);
	extractor.compute(grayImg, keypoints, descriptors);

	for (int i = 0; i < keypoints.size(); ++i) {
		KeyPoint keypoint = keypoints[i];
		Mat descriptor = descriptors.row(i);
		features.push_back(SIFTFeature(keypoint, descriptor));
	}

	return features;
}