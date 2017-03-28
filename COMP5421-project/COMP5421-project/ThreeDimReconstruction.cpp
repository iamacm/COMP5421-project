#include "stdafx.h"
#include <windows.h>
#include <algorithm> 
#include "opencv2/imgproc/imgproc.hpp"
#include "ThreeDimReconstruction.h"

#define RESIZE_RATIO		0.50		// Relative to the screen size
#define	SCREEN_WIDTH		GetSystemMetrics(SM_CXSCREEN)
#define	SCREEN_HEIGHT		GetSystemMetrics(SM_CYSCREEN)

// Constructor of Img

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
void ThreeDimReconstruction::Img::show(void) {
	const int MAX_WIDTH = SCREEN_WIDTH * RESIZE_RATIO;
	const int MAX_HEIGHT = SCREEN_HEIGHT * RESIZE_RATIO;

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




void ThreeDimReconstruction::showOriginalImg(void) {
	for (int i = 0; i < 2; ++i) {
		this->img[i].show();
	}
	this->wait();
}

void ThreeDimReconstruction::wait(void) {
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
}

void ThreeDimReconstruction::FeatureDetection::detectHarrisCorner(Img src, Mat dst, bool output) {
	
}