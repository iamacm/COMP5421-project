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
	// Sub-class Img
	class Img {
	public:
		Img(string path);
		void show(void);
		Mat mat;
		string path, name;
	};

	class FeatureDetection {
		static void detectHarrisCorner(Img src, Mat dst, bool output);
	};
public:
	ThreeDimReconstruction(char* imgPath1, char* imgPath2);
	//void show(Img img);	// Show image of id
	void showOriginalImg(void);	// Show all images
	void process(void);
	void wait(void);
	
	
private:
	vector<ThreeDimReconstruction::Img> img;	// ARRAY of Img*
};




#endif