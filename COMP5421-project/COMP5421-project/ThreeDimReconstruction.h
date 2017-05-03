#pragma once
#ifndef THREEDMINRECONSTRUCTION_H
#define THREEDMINRECONSTRUCTION_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include "SIFTFeature.h"

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
		static vector<SIFTFeature> detectSIFT(const Img& src);
		static void detectSIFT(vector<SIFTFeature>& features, const Img& src);
	};
public:
	// Constructors
	ThreeDimReconstruction(char* imgPath1, char* imgPath2);
	// Methods
	void showOriginalImg(void) const;	// Show all images
	void processHarrisCorner(void);

	// Visualization methods
	Img visualizeFeatures(const Img& img, const vector<SIFTFeature>& features) const;
	Img visualizeMatchings(const Img& img1, const Img& img2, const vector<pair<SIFTFeature, SIFTFeature>>& matchings);
	Img visualizeMatchingWithEpipolarLines(const Img& img1, const Img& img2, const vector<pair<SIFTFeature, SIFTFeature>>& matchings, const Mat& F);
	vector<pair<SIFTFeature, SIFTFeature>> SIFTFeatureMatching(const Img& img1, const vector<SIFTFeature> features1, const Img& img2, const vector<SIFTFeature> features2);
	Mat eightPointAlgorithm(const vector<pair<SIFTFeature, SIFTFeature>>& matchings, const int N = 8);
	void process(void);
	void wait(void) const;
	
	
private:
	// Properties
	vector<ThreeDimReconstruction::Img> images;	// ARRAY of Img*
};




#endif