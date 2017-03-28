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
	void show(int id);	// Show image of id
	void showAll(void);	// Show all images
	void wait(void);
	struct Img {
		Mat mat;
		string path, name;
	};
private:
	ThreeDimReconstruction::Img img[2];
};




#endif