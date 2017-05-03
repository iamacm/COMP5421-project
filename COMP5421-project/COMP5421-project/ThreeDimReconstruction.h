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


class Matching: public pair<SIFTFeature, SIFTFeature> {
public:
	Matching() {};
	Matching(SIFTFeature x, SIFTFeature y) {
		this->first = x;
		this->second = y;
	};
	double descriptorDistSquared, imgDist;
};

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
		void show(double resizeRatio = 0.50) const;
		void showWith(Img img, double resizeRatio = 0.95) const;
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
	ThreeDimReconstruction(const char* const * imgPaths, int count);
	// Methods
	void showOriginalImg(void) const;	// Show all images
	Img processHarrisCorner(const Img& img);

	// Visualization methods
	Img visualizeFeatures(const Img& img, const vector<SIFTFeature>& features) const;
	Img visualizeMatchings(const Img& img1, const Img& img2, const vector<Matching>& matchings);
	Img visualizeMatchingWithEpipolarLines(const Img& img1, const Img& img2, const vector<Matching>& matchings, const Mat& F);
	vector<Matching> SIFTFeatureMatching(const Img& img1, const vector<SIFTFeature> features1, const Img& img2, const vector<SIFTFeature> features2);
	Mat computeFundamentalMatrix(const vector<Matching>& matchings, const int N = 8);
	Mat twoViewTriangulation(const vector<Matching>& matchings, const Mat& F,
		Mat& outputR, Mat& outputT,
		const Mat* prevR = NULL, const Mat* prevT = NULL);
	void writePly(const string& file, const Mat& points3D);
	void writeFundamentalMatrix(const string& file, const Mat& F);
	void process(void);
	void wait(void) const;
	
	
private:
	// Properties
	vector<ThreeDimReconstruction::Img> images;	// ARRAY of Img*

};




#endif