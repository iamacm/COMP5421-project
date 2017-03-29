#include "stdafx.h"
#include <windows.h>
#include <algorithm> 
#include "opencv2/imgproc/imgproc.hpp"
#include "ThreeDimReconstruction.h"


#define	SCREEN_WIDTH				GetSystemMetrics(SM_CXSCREEN)
#define	SCREEN_HEIGHT				GetSystemMetrics(SM_CYSCREEN)

// Constructor of Img
ThreeDimReconstruction::Img::Img(void) {

}

ThreeDimReconstruction::Img::Img(string path) {
	// Store the image path
	this->path = path;
	// Read the file
	this->mat = imread(this->path, IMREAD_COLOR);
	this->mat = imread(this->path, IMREAD_COLOR);
	if (this->mat.empty()) {
		throw new Exception();
	}

	// Name the image
	this->name = "Image " + this->path;

}

// Display the image of ID with aspect ratio kept
void ThreeDimReconstruction::Img::show(float resizeRatio) const {
	const int MAX_WIDTH = SCREEN_WIDTH * resizeRatio;
	const int MAX_HEIGHT = SCREEN_HEIGHT * resizeRatio;

	int width = this->mat.cols;
	int height = this->mat.rows;
	if (width > MAX_WIDTH || height > MAX_HEIGHT) {
		const double resize_ratio = min((double)MAX_WIDTH / width, (double)MAX_HEIGHT / height);
		width *= resize_ratio;
		height *= resize_ratio;
	}

	namedWindow(this->name, WINDOW_KEEPRATIO); // Create a window for display.
	resizeWindow(this->name, width, height);		// Resize the image
	imshow(this->name, this->mat); // Show our image inside it.

	printf("\"%s\" displayed\n", this->name.c_str());
}

// Display an image with another image together in the same window, horizontally merged
void ThreeDimReconstruction::Img::showWith(ThreeDimReconstruction::Img anotherImg, float resizeRatio) const {
	Img bothImg;
	hconcat(this->mat, anotherImg.mat, bothImg.mat);
	bothImg.name = this->name + " with " + anotherImg.name;
	bothImg.show(resizeRatio);
}

ThreeDimReconstruction::ThreeDimReconstruction(char* imgPath1, char* imgPath2)
{
	printf("Screen resolution: %d * %d\n", SCREEN_WIDTH, SCREEN_HEIGHT);

	this->img.push_back(ThreeDimReconstruction::Img(imgPath1));
	this->img.push_back(ThreeDimReconstruction::Img(imgPath2));

	
	printf("Loaded files successfully!\n");
	for (int i = 0; i < this->img.size(); ++i) {
		printf("img[%d]: %d * %d\n", i, this->img[i].mat.cols, this->img[i].mat.rows);
	}
	
}




void ThreeDimReconstruction::showOriginalImg(void) const {
	for (const Img img : this->img) {
		img.show();
	}
	this->wait();
}

void ThreeDimReconstruction::wait(void) const {
	waitKey(0); // Wait for a keystroke in the window
}

void ThreeDimReconstruction::process(void) {
	/*
	Img imgFiltered;
	imgFiltered.name = "Image blurred";
	int kernel_size = 100;
	Mat kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size*kernel_size);
	filter2D(this->img[0].mat, imgFiltered.mat, -1, kernel);
	this->show(imgFiltered);
	this->wait();
	*/
	
	//this->wait();
	const int imgCount = this->img.size();
	vector<Img> dst(imgCount), dstNorm(imgCount);
	

	for (int i = 0; i < imgCount; ++i) {
		ThreeDimReconstruction::FeatureDetector::detectHarrisCorner(this->img[i], dst[i], dstNorm[i]);
		//dstNorm[i].name = "dstNorm of image " + this->img[i].name;
		//dstNorm[i].show();
	}

	this->wait();
}

void ThreeDimReconstruction::FeatureDetector::nonMaxSuppression(const Img src, Img& dst) {
	// find pixels that are equal to the local neighborhood not maximum (including 'plateaus')
	cv::dilate(src.mat, dst.mat, Mat());
	//cv::compare(src.mat, dst.mat, dst.mat, cv::CMP_GE);
}
vector<Point2d> ThreeDimReconstruction::FeatureDetector::detectHarrisCorner(const Img src, Img& dst, Img& dstNorm, bool showResult) {
	/// Result
	vector<Point2d> cornerPoints;
	/// Detector parameters
	const int blockSize = 4;
	const int apertureSize = 3;
	const float thresholdVal = 0.00005;
	const double k = 0.04;

	Img srcGray, thresholdedBinary, thresholded, thresholdedDilated, thresholdedDilatedBinary, localMaxBinary;
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
					cornerPoints.push_back(Point2d(i, j));
				}

			}
		}
	}
	//ThreeDimReconstruction::FeatureDetector::nonMaxSuppression(thresholded, localMaxBinary);
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
		dstNorm.show();

		thresholdedBinary.name = "Thresholded mask of image " + src.name;
		//thresholdedBinary.show();

		thresholdedDilatedBinary.name = "Thresholded dilated binary of image " + src.name;
		//thresholdedDilatedBinary.show();
		thresholdedBinary.showWith(thresholdedDilatedBinary);

		localMaxBinary.name = "localMax of image " + src.name;
		localMaxBinary.show();
	}

	printf("%d corner(s) detected!\n", cornerPoints.size());
	//namedWindow("corners_window", CV_WINDOW_AUTOSIZE);
	//imshow("corners_window", dst_norm_scaled);

	//dstNorm.mat = Mat(dst_norm_scaled);
	//waitKey(0);
	return cornerPoints;
}