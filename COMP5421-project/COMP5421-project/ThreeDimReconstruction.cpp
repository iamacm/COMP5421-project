#include "stdafx.h"
#include <windows.h>
#include <algorithm> 
#include <stdint.h>
#include <thread>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "ThreeDimReconstruction.h"
#include "SIFTFeature.h"

#define	SCREEN_WIDTH				GetSystemMetrics(SM_CXSCREEN)
#define	SCREEN_HEIGHT				GetSystemMetrics(SM_CYSCREEN)

// Global function
double euclideanDistanceSquared(const Mat& mat1, const Mat& mat2) {
	if (mat1.cols != mat2.cols) {
		throw Exception();
	}
	double sum = 0;
	for (int i = 0; i < mat1.cols; ++i) {
		const int diff = mat1.at<uchar>(0, i) - mat2.at<uchar>(0, i);
		sum += diff * diff;
	}
	return sum;
}

// Find the nearest neighbor of feature1 out of the list of features
// Return the id of the nearest neighbor feature
int nearestNeighbor(const SIFTFeature& feature1, const vector<SIFTFeature>& features) {
	int id = -1;
	double minDistance = INFINITY;
	for (int i = 0; i < features.size(); ++i) {
		const double distance = euclideanDistanceSquared(feature1.descriptor, features[i].descriptor);
		if (minDistance > distance) {
			id = i;
			minDistance = distance;
		}
	}
	return id;
}

// Constructor of Img
ThreeDimReconstruction::Img::Img(void) {

}

ThreeDimReconstruction::Img::Img(const ThreeDimReconstruction::Img& img) {
	this->mat = img.mat.clone();
	this->name = img.name;
	this->path = img.path;
}

ThreeDimReconstruction::Img::Img(string path) {
	// Store the image path
	this->path = path;
	// Read the file
	this->mat = imread(this->path, IMREAD_COLOR);
	if (this->mat.empty()) {
		throw new Exception();
	}

	// Name the image
	this->name = "Image " + this->path;

}

ThreeDimReconstruction::Img ThreeDimReconstruction::Img::clone() const {
	return ThreeDimReconstruction::Img(*this);
}

// Display the image of ID with aspect ratio kept
void ThreeDimReconstruction::Img::show(float resizeRatio) const {
	const int MAX_WIDTH = SCREEN_WIDTH * resizeRatio;
	const int MAX_HEIGHT = SCREEN_HEIGHT * resizeRatio;

	int width = this->mat.cols;
	int height = this->mat.rows;
	if (width > MAX_WIDTH || height > MAX_HEIGHT) {
		const double resize_ratio = min((double)MAX_WIDTH / width, (double)MAX_HEIGHT / height);
		width *= resize_ratio;
		height *= resize_ratio;
	}

	namedWindow(this->name, CV_WINDOW_NORMAL); // Create a window for display.
	resizeWindow(this->name, width, height);		// Resize the image
	imshow(this->name, this->mat); // Show our image inside it.

	printf("\"%s\" displayed\n", this->name.c_str());
}

// Display an image with another image together in the same window, horizontally merged
void ThreeDimReconstruction::Img::showWith(ThreeDimReconstruction::Img anotherImg, float resizeRatio) const {
	Img bothImg;
	hconcat(this->mat, anotherImg.mat, bothImg.mat);
	bothImg.name = this->name + " with " + anotherImg.name;
	bothImg.show(resizeRatio);
}

ThreeDimReconstruction::ThreeDimReconstruction(char* imgPath1, char* imgPath2)
{
	printf("Screen resolution: %d * %d\n", SCREEN_WIDTH, SCREEN_HEIGHT);

	this->images.push_back(ThreeDimReconstruction::Img(imgPath1));
	this->images.push_back(ThreeDimReconstruction::Img(imgPath2));

	
	printf("Loaded files successfully!\n");
	for (int i = 0; i < this->images.size(); ++i) {
		printf("img[%d]: %d * %d\n", i, this->images[i].mat.cols, this->images[i].mat.rows);
	}
	
}




void ThreeDimReconstruction::showOriginalImg(void) const {
	for (const Img& img : this->images) {
		img.show();
	}
	this->wait();
}

void ThreeDimReconstruction::wait(void) const {
	waitKey(0); // Wait for a keystroke in the window
}

void ThreeDimReconstruction::processHarrisCorner(void) {
	/*
	Img imgFiltered;
	imgFiltered.name = "Image blurred";
	int kernel_size = 100;
	Mat kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size*kernel_size);
	filter2D(this->img[0].mat, imgFiltered.mat, -1, kernel);
	this->show(imgFiltered);
	this->wait();
	*/
	
	//this->wait();
	const int imgCount = this->images.size();
	vector<vector<Point>> imgCorners;
	vector<Img> imgCornersCircled(this->images.size());

	for (int i = 0; i < imgCount; ++i) {
		vector<Point2d> corners = ThreeDimReconstruction::FeatureDetectors::detectHarrisCorner(this->images[i], false);
		imgCornersCircled[i].mat = this->images[i].mat.clone();
		for (const Point pt : corners) {
			circle(imgCornersCircled[i].mat, pt, 15, Scalar(0, 0, 255), 5);
		}
		imgCornersCircled[i].name = this->images[i].name + " with cornered circled";
		imgCornersCircled[i].show();
		
	}

	this->wait();
}


void ThreeDimReconstruction::visualizeFeatures(const Img& img, const vector<SIFTFeature>& features) const {
	Img imgWithFeatures = img.clone();

	for (const auto& feature : features) {
		circle(imgWithFeatures.mat, feature.keypoint.pt, cvRound(feature.keypoint.size*1.0), Scalar(0, 255, 0), 3);
	}

	imgWithFeatures.name += " SIFT features";
	imgWithFeatures.show();
	//circle(matchingImage, kp.pt + Point2f(newColorImage.size().width, newColorImage.size().height), cvRound(kp.size*1.0), Scalar(0, 255, 0), 5, 8, 0);
}

void ThreeDimReconstruction::visualizeMatchings(const Img& img1, const Img& img2, const vector<pair<SIFTFeature, SIFTFeature>> matchings) {
	Img matchingImg;
	matchingImg.name = "Matching of " + img1.name + " & " + img2.name;
	Size matchingImgsize(img1.mat.size().width + img2.mat.size().width, max(img1.mat.size().height, img2.mat.size().height));
	matchingImg.mat = Mat::zeros(matchingImgsize, CV_8UC3);	// Creating image of size

	// Copy images to the matching image
	img1.mat.copyTo(matchingImg.mat(Rect(0, 0, img1.mat.size().width, img1.mat.size().height)));
	img2.mat.copyTo(matchingImg.mat(Rect(img1.mat.size().width, 0, img2.mat.size().width, img2.mat.size().height)));

	// Circle keypoints and draw lines
	for (const pair<SIFTFeature, SIFTFeature>& matching : matchings) {
		const SIFTFeature& feature1 = matching.first;
		const SIFTFeature& feature2 = matching.second;
		// Circle feature1
		circle(matchingImg.mat, feature1.keypoint.pt, cvRound(feature1.keypoint.size*1.0), Scalar(0, 255, 0), 3);
		// Circle feature2
		circle(matchingImg.mat, feature2.keypoint.pt + Point2f(img1.mat.size().width, 0), cvRound(feature2.keypoint.size*1.0), Scalar(0, 255, 0), 3);
		// Draw line between the two features
		line(matchingImg.mat, feature1.keypoint.pt, feature2.keypoint.pt + Point2f(img1.mat.size().width, 0), Scalar(0, 0, 255), 3);
	}

	matchingImg.show();
}

vector<pair<SIFTFeature, SIFTFeature>> ThreeDimReconstruction::SIFTFeatureMatching(const Img& img1, const vector<SIFTFeature> features1, const Img& img2, const vector<SIFTFeature> features2) {
	// For each feature of mat1, find the best feature to be matched with mat2
	vector<pair<SIFTFeature, SIFTFeature>> matchings;

	for (int i = 0; i < features1.size(); ++i) {
		int img2MatchedFeatureId = nearestNeighbor(features1[i], features2);
		printf("%d\t%d\n", i, img2MatchedFeatureId);
		const SIFTFeature& img2MatchedFeature = features2[i];

		// Left-right check (check if this img1 feature is also the best matched of img2)
		if (nearestNeighbor(img2MatchedFeature, features1) == i) {
			matchings.push_back(make_pair(features1[i], img2MatchedFeature));
		}

		//matchings.push_back(make_pair(features1[i], img2MatchedFeature));

		
		
	}

	return matchings;
}

void ThreeDimReconstruction::process(void) {
	vector<vector<SIFTFeature>> featuresOfImages;

	for (const Img& img : this->images) {
		SIFTFeature feature;
		vector<SIFTFeature> features = FeatureDetectors::detectSIFT(img);

		printf("%d SIFT feature(s) found in %s\n", features.size(), img.name);

		//visualizeFeatures(img, features);

		featuresOfImages.push_back(features);
	}

	if (this->images.size() >= 2) {
		vector<pair<SIFTFeature, SIFTFeature>> matchings = SIFTFeatureMatching(this->images[0], featuresOfImages[0], this->images[1], featuresOfImages[1]);
		visualizeFeatures(this->images[0], featuresOfImages[0]);
		visualizeFeatures(this->images[1], featuresOfImages[1]);
		printf("%d matchings found\n", matchings.size());
		visualizeMatchings(this->images[0], this->images[1], matchings);
	}

	this->wait();
	

}

