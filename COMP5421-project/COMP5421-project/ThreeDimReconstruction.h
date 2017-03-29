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
		void show(float resizeRatio = 0.50) const;
		void showWith(Img img, float resizeRatio = 0.95) const;
		// Properties
		Mat mat;
		string path, name;
	};

	// Sub-class FeatureDetection
	class FeatureDetector {
	public:
		static void nonMaxSuppression(const Img src, Img& dst);
		static vector<Point2d> detectHarrisCorner(const Img src, bool showResult = true);
	};
public:
	// Constructors
	ThreeDimReconstruction(char* imgPath1, char* imgPath2);
	// Methods
	void showOriginalImg(void) const;	// Show all images
	void process(void);
	void wait(void) const;
	
	
private:
	// Properties
	vector<ThreeDimReconstruction::Img> img;	// ARRAY of Img*
};




#endif