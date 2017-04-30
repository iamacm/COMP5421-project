#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;
using namespace cv;

const double THRESHOLD = 500;
const double KEYPOINT_MIN = 1.0;
/**
* Calculate euclid distance
*/
double euclidDistance(Mat& vec1, Mat& vec2) {
	double sum = 0.0;
	int dim = vec1.cols;
	for (int i = 0; i < dim; i++) {
		sum += (vec1.at<uchar>(0, i) - vec2.at<uchar>(0, i)) * (vec1.at<uchar>(0, i) - vec2.at<uchar>(0, i));
	}
	return sqrt(sum);
}

/**
* Find the index of nearest neighbor point from keypoints.
*/
int nearestNeighbor(const KeyPoint keypoint, Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors) {
	int neighbor = -1;
	double minDist = INFINITY;

	for (int i = 0; i < descriptors.rows; i++) {
		KeyPoint pt = keypoints[i];
		if (pt.size < KEYPOINT_MIN) { continue; }
		Mat v = descriptors.row(i);
		double d = euclidDistance(vec, v);
		double pixel_d = norm(pt.pt - keypoint.pt);
		//printf("%d %f\n", v.cols, d);
		double obj = d + pixel_d * 0.2;
		if (obj < minDist) {
			minDist = d;
			neighbor = i;
		}
	}

	if (minDist < THRESHOLD) {
		return neighbor;
	}

	return -1;
}

/**
* Find pairs of points with the smallest distace between them
*/
void findPairs(vector<KeyPoint>& keypoints1, Mat& descriptors1,
	vector<KeyPoint>& keypoints2, Mat& descriptors2,
	vector<Point2f>& srcPoints, vector<Point2f>& dstPoints) {
	for (int i = 0; i < descriptors1.rows; i++) {
		KeyPoint pt1 = keypoints1[i];
		if (pt1.size < KEYPOINT_MIN) { continue; }
		Mat desc1 = descriptors1.row(i);
		int nn = nearestNeighbor(pt1, desc1, keypoints2, descriptors2);
		if (nn >= 0) {
			KeyPoint pt2 = keypoints2[nn];
			srcPoints.push_back(pt1.pt);
			dstPoints.push_back(pt2.pt);
		}
	}
}

int main(int argc, char** argv) {
	const char* filename[2];
	if (argc < 3) {
		filename[0] = "0006.png";
		filename[1] = "0005.png";
	}
	else {
		filename[0] = argv[1];
		filename[1] = argv[2];
	}

	

	printf("load file:%s and %s\n", filename[0], filename[1]);
	

	// initialize detector and extractor
	FeatureDetector* detector;
	detector = new SiftFeatureDetector(
		0, // nFeatures
		4, // nOctaveLayers
		0.10, // contrastThreshold 0.04
		20, //edgeThreshold 10
		1.6 //sigma
	);

	DescriptorExtractor* extractor;
	extractor = new SiftDescriptorExtractor();

	// Compute keypoints and descriptor from the source image in advance
	vector<KeyPoint> keypoints2;
	Mat descriptors2;

	
	Mat originalGrayImage;
	Mat originalColorImage = imread(filename[0], CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
	if (!originalColorImage.data) {
		cerr << "color image open error" << endl;
		return -1;
	}
	cv::cvtColor(originalColorImage, originalGrayImage, CV_BGR2GRAY);

	Mat newGrayImage;
	if (!originalGrayImage.data) {
		cerr << "gray image load error" << endl;
		return -1;
	}
	Mat newColorImage = imread(filename[1], CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
	if (!originalColorImage.data) {
		cerr << "color image open error" << endl;
		return -1;
	}

	namedWindow("mywindow", CV_WINDOW_NORMAL);
	imshow("mywindow", originalColorImage);

	



	// Create a image for displaying mathing keypoints
	Size sz = Size(newColorImage.size().width + originalColorImage.size().width, newColorImage.size().height + originalColorImage.size().height);
	Mat matchingImage = Mat::zeros(sz, CV_8UC3);

	// Draw camera framehaha
	Mat roi1 = Mat(matchingImage, Rect(0, 0, newColorImage.size().width, newColorImage.size().height));
	newColorImage.copyTo(roi1);
	// Draw original image
	Mat roi2 = Mat(matchingImage, Rect(newColorImage.size().width, newColorImage.size().height, originalColorImage.size().width, originalColorImage.size().height));
	originalColorImage.copyTo(roi2);

	vector<KeyPoint> keypoints1;
	Mat descriptors1;
	vector<DMatch> matches;

	// Detect keypoints


	detector->detect(originalGrayImage, keypoints2);
	extractor->compute(originalGrayImage, keypoints2, descriptors2);
	printf("original image:%d keypoints are found.\n", (int)keypoints2.size());


	detector->detect(newGrayImage, keypoints1);
	extractor->compute(newGrayImage, keypoints1, descriptors1);

	printf("image1:%zd keypoints are found.\n", keypoints1.size());


	for (int i = 0; i<keypoints1.size(); i++) {
		KeyPoint kp = keypoints1[i];
		cout << kp.size << endl;
		circle(matchingImage, kp.pt, cvRound(kp.size*1.0), Scalar(0, 255, 0), 5, 8, 0);
	}

	for (int i = 0; i<keypoints2.size(); i++) {
		KeyPoint kp = keypoints2[i];
		circle(matchingImage, kp.pt + Point2f(newColorImage.size().width, newColorImage.size().height), cvRound(kp.size*1.0), Scalar(0, 255, 0), 5, 8, 0);
	}

	// Find nearest neighbor pairs
	vector<Point2f> srcPoints;
	vector<Point2f> dstPoints;
	findPairs(keypoints1, descriptors1, keypoints2, descriptors2, srcPoints, dstPoints);
	printf("%zd keypoints are matched.\n", srcPoints.size());

	char text[256];
	sprintf(text, "%zd/%zd keypoints matched.", srcPoints.size(), keypoints2.size());
	//putText(matchingImage, text, Point(0, cvRound(newColorImage.size().height + 30)), FONT_HERSHEY_SCRIPT_SIMPLEX, 1, Scalar(0, 0, 255));

	// Draw line between nearest neighbor pairs
	for (int i = 0; i < (int)srcPoints.size(); ++i) {
		Point2f pt1 = srcPoints[i];
		Point2f pt2 = dstPoints[i];
		Point2f from = pt1;
		Point2f to = Point(newColorImage.size().width + pt2.x, newColorImage.size().height + pt2.y);
		line(matchingImage, from, to, Scalar(0, 0, 255), 5);
	}
	

	namedWindow("matching Image", CV_WINDOW_NORMAL);
	// Display mathing image
	imshow("matching Image", matchingImage);
	imwrite("matching.jpg", matchingImage);


	waitKey(0);

	return 0;
}