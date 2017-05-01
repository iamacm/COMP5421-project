#include "stdafx.h"
#include <windows.h>
#include <algorithm> 
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
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
	const int imgCount = this->img.size();
	vector<vector<Point>> imgCorners;
	vector<Img> imgCornersCircled(this->img.size());

	for (int i = 0; i < imgCount; ++i) {
		vector<Point2d> corners = ThreeDimReconstruction::FeatureDetector::detectHarrisCorner(this->img[i], false);
		imgCornersCircled[i].mat = this->img[i].mat.clone();
		for (const Point pt : corners) {
			circle(imgCornersCircled[i].mat, pt, 15, Scalar(0, 0, 255), 5);
		}
		imgCornersCircled[i].name = this->img[i].name + " with cornered circled";
		imgCornersCircled[i].show();
		
	}

	this->wait();
}

void ThreeDimReconstruction::process(void) {

}

