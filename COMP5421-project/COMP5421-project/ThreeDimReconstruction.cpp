#include "stdafx.h"
#include <windows.h>
#include <algorithm> 
#include "ThreeDimReconstruction.h"

#define RESIZE_RATIO		0.45		// Relative to the screen size
#define	SCREEN_WIDTH		GetSystemMetrics(SM_CXSCREEN)
#define	SCREEN_HEIGHT		GetSystemMetrics(SM_CYSCREEN)
ThreeDimReconstruction::ThreeDimReconstruction(char* imgPath1, char* imgPath2)
{
	// Store the image path
	this->img[0].path = imgPath1;
	this->img[1].path = imgPath2;

	// Read the file
	this->img[0].mat = imread(this->img[0].path, IMREAD_COLOR);
	this->img[1].mat = imread(this->img[1].path, IMREAD_COLOR);
	if (this->img[0].mat.empty() || this->img[1].mat.empty()) {
		throw new Exception();
	}

	// Name the image
	this->img[0].name = "Image " + this->img[0].path;
	this->img[1].name = "Image " + this->img[1].path;
	if (this->img[0].name == this->img[1].name) {
		this->img[1].name += " (1)";
	}

	printf("Screen resolution: %d * %d\n", SCREEN_WIDTH, SCREEN_HEIGHT);
	printf("Loaded files successfully!\n");
	for (int i = 0; i < 2; ++i) {
		printf("img[%d]: %d * %d\n", i, this->img[i].mat.cols, this->img[i].mat.rows);
	}
	
}


// Display the image of ID with aspect ratio kept
void ThreeDimReconstruction::show(int id) {
	const int MAX_WIDTH = SCREEN_WIDTH * RESIZE_RATIO;
	const int MAX_HEIGHT = SCREEN_HEIGHT * RESIZE_RATIO;

	Img thisImg = this->img[id];
	int width = thisImg.mat.cols;
	int height = thisImg.mat.rows;
	if (width > MAX_WIDTH || height > MAX_HEIGHT) {
		const double resize_ratio = min((double)MAX_WIDTH / width, (double)MAX_HEIGHT/ height);
		width *= resize_ratio;
		height *= resize_ratio;
	}
	
	namedWindow(this->img[id].name, WINDOW_KEEPRATIO); // Create a window for display.
	resizeWindow(this->img[id].name, width, height);		// Resize the image
	imshow(this->img[id].name, this->img[id].mat); // Show our image inside it.
}

void ThreeDimReconstruction::showAll(void) {
	for (int i = 0; i < 2; ++i) {
		this->show(i);
	}
}

void ThreeDimReconstruction::wait(void) {
	waitKey(0); // Wait for a keystroke in the window
}
