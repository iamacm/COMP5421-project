#include "stdafx.h"
#include "ThreeDimReconstruction.h"


ThreeDimReconstruction::ThreeDimReconstruction(char* imgPath1, char* imgPath2) {
	this->imgPath[0] = imgPath1;
	this->imgPath[1] = imgPath2;

	// Read the file
	img[0] = imread(this->imgPath[0], IMREAD_COLOR);
	img[1] = imread(this->imgPath[1], IMREAD_COLOR);
	if (img[0].empty() || img[1].empty()) {
		throw new Exception();
	}

	printf("Loaded files successfully!\n");
	printf("img[0]: %d * %d\n", img[0].cols, img[0].rows);
	printf("img[1]: %d * %d\n", img[1].cols, img[1].rows);
}

void ThreeDimReconstruction::show(void) {
	
	namedWindow("Image " + this->imgPath[0], WINDOW_NORMAL | WINDOW_KEEPRATIO); // Create a window for display.
	imshow("Image " + this->imgPath[0], img[0]); // Show our image inside it.

	namedWindow("Image " + this->imgPath[1], WINDOW_NORMAL | WINDOW_KEEPRATIO); // Create a window for display.
	imshow("Image " + this->imgPath[1], img[1]); // Show our image inside it.
}
