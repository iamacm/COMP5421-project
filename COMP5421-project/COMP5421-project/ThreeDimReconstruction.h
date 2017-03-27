#pragma once
#ifndef THREEDMINRECONSTRUCTION_H
#define THREEDMINRECONSTRUCTION_H

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

class ThreeDimReconstruction {

public:
	ThreeDimReconstruction(char* imgPath1, char* imgPath2);
	void show(void);

private:
	int x;
	Mat img[2];	// The two images for 3D reconstruction
	string imgPath[2];
};




#endif