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
		// Constructors
		Img(void);
		Img(string path);
		// Methods
		void show(float resizeRatio = 0.50);
		void showWith(Img img, float resizeRatio = 0.95);
		// Properties
		Mat mat;
		string path, name;
	};

	// Sub-class FeatureDetection
	class FeatureDetector {
		static void detectHarrisCorner(Img src, Img dst, bool output);
	};
public:
	// Constructors
	ThreeDimReconstruction(char* imgPath1, char* imgPath2);
	// Methods
	void showOriginalImg(void);	// Show all images
	void process(void);
	void wait(void);
	
	
private:
	// Properties
	vector<ThreeDimReconstruction::Img> img;	// ARRAY of Img*
};




#endif