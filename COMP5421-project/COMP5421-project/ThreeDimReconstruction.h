#pragma once
#ifndef THREEDMINRECONSTRUCTION_H
#define THREEDMINRECONSTRUCTION_H

#include <opencv2/core.hpp>
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
		Img(const ThreeDimReconstruction::Img& img);	// Deep copy
		Img(string path);
		// Methods
		ThreeDimReconstruction::Img clone() const;
		void show(float resizeRatio = 0.50) const;
		void showWith(Img img, float resizeRatio = 0.95) const;
		// Properties
		Mat mat;
		string path, name;
	};

	// Sub-class FeatureDetection
	class FeatureDetectors {
	public:
		static void nonMaxSuppression(const Img src, Img& dst);
		static vector<Point2d> detectHarrisCorner(const Img src, bool showResult = true);
		static vector<pair<KeyPoint, Mat>> detectSIFT(const Img src, bool showResult = true);
	};
public:
	// Constructors
	ThreeDimReconstruction(char* imgPath1, char* imgPath2);
	// Methods
	void showOriginalImg(void) const;	// Show all images
	void processHarrisCorner(void);
	void process(void);
	void wait(void) const;
	
	
private:
	// Properties
	vector<ThreeDimReconstruction::Img> images;	// ARRAY of Img*
};




#endif