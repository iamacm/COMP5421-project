#include "stdafx.h"
#include <windows.h>
#include <algorithm> 
#include <stdint.h>
#include <thread>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "ThreeDimReconstruction.h"
#include "SIFTFeature.h"

#define	SCREEN_WIDTH				GetSystemMetrics(SM_CXSCREEN)
#define	SCREEN_HEIGHT				GetSystemMetrics(SM_CYSCREEN)

// Constructor of Img
ThreeDimReconstruction::Img::Img(void) {

}

ThreeDimReconstruction::Img::Img(const ThreeDimReconstruction::Img& img) {
	this->mat = img.mat.clone();
	this->name = img.name;
	this->path = img.path;
}

ThreeDimReconstruction::Img::Img(string path) {
	// Store the image path
	this->path = path;
	// Read the file
	this->mat = imread(this->path, IMREAD_COLOR);
	if (this->mat.empty()) {
		throw new Exception();
	}

	// Name the image
	this->name = "Image " + this->path;

}

ThreeDimReconstruction::Img ThreeDimReconstruction::Img::clone() const {
	return ThreeDimReconstruction::Img(*this);
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

	namedWindow(this->name, CV_WINDOW_NORMAL); // Create a window for display.
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

	this->images.push_back(ThreeDimReconstruction::Img(imgPath1));
	this->images.push_back(ThreeDimReconstruction::Img(imgPath2));

	
	printf("Loaded files successfully!\n");
	for (int i = 0; i < this->images.size(); ++i) {
		printf("img[%d]: %d * %d\n", i, this->images[i].mat.cols, this->images[i].mat.rows);
	}
	
}




void ThreeDimReconstruction::showOriginalImg(void) const {
	for (const Img& img : this->images) {
		img.show();
	}
	this->wait();
}

void ThreeDimReconstruction::wait(void) const {
	waitKey(0); // Wait for a keystroke in the window
}

void ThreeDimReconstruction::processHarrisCorner(void) {
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
	const int imgCount = this->images.size();
	vector<vector<Point>> imgCorners;
	vector<Img> imgCornersCircled(this->images.size());

	for (int i = 0; i < imgCount; ++i) {
		vector<Point2d> corners = ThreeDimReconstruction::FeatureDetectors::detectHarrisCorner(this->images[i], false);
		imgCornersCircled[i].mat = this->images[i].mat.clone();
		for (const Point pt : corners) {
			circle(imgCornersCircled[i].mat, pt, 15, Scalar(0, 0, 255), 5);
		}
		imgCornersCircled[i].name = this->images[i].name + " with cornered circled";
		imgCornersCircled[i].show();
		
	}

	this->wait();
}


void ThreeDimReconstruction::visualizeFeatures(const Img& img, const vector<SIFTFeature>& features) const {
	Img imgWithFeatures = img.clone();

	for (const auto& feature : features) {
		circle(imgWithFeatures.mat, feature.keypoint.pt, cvRound(feature.keypoint.size*1.0), Scalar(0, 255, 0), 3);
	}

	imgWithFeatures.name += " SIFT features";
	imgWithFeatures.show();
	//circle(matchingImage, kp.pt + Point2f(newColorImage.size().width, newColorImage.size().height), cvRound(kp.size*1.0), Scalar(0, 255, 0), 5, 8, 0);
}


void ThreeDimReconstruction::process(void) {
	vector<thread> threads;
	vector<SIFTFeature> features;
	for (const Img& img : this->images) {

		vector<SIFTFeature> features = FeatureDetectors::detectSIFT(img);

		printf("%d SIFT feature(s) found in %s\n", features.size(), img.name);

		visualizeFeatures(img, features);
	}

	this->wait();

}

