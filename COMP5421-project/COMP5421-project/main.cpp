// COMP5421-project.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>
#include "ThreeDimReconstruction.h"
#include <thread>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;


const double THRESHOLD = 400;

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
int nearestNeighbor(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors) {
	int neighbor = -1;
	double minDist = 1e6;

	for (int i = 0; i < descriptors.rows; i++) {
		KeyPoint pt = keypoints[i];
		Mat v = descriptors.row(i);
		double d = euclidDistance(vec, v);
		//printf("%d %f\n", v.cols, d);
		if (d < minDist) {
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
		Mat desc1 = descriptors1.row(i);
		int nn = nearestNeighbor(desc1, keypoints2, descriptors2);
		if (nn >= 0) {
			KeyPoint pt2 = keypoints2[nn];
			srcPoints.push_back(pt1.pt);
			dstPoints.push_back(pt2.pt);
		}
	}
}



int maintest(int argc, char** argv)
{
	if (argc != 2)
	{
		cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
		return -1; 
	}

	Mat image;
	image = imread(argv[1], CV_LOAD_IMAGE_COLOR); // Read the file

	if (image.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image " << argv[1] << std::endl;
		return -1;
	}

	namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window", image); // Show our image inside it.

	waitKey(0); // Wait for a keystroke in the window
	return 0;
}

int main(int argc, char** argv)
{
	ThreeDimReconstruction* threeDimRec = NULL;
	if (argc != 3)
	{
		threeDimRec = new ThreeDimReconstruction("0005.png", "0006.png");
		//cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
		//return -1;
		//delete argv;
		//argv = new char*[3];
		//argv[1] = "0005.png";
		//argv[2] = "0006.png";
	}
	else {
		threeDimRec = new ThreeDimReconstruction(argv[1], argv[2]);
	}

	thread t1(&ThreeDimReconstruction::showOriginalImg, threeDimRec);
	thread t2(&ThreeDimReconstruction::process, threeDimRec);
	//thread t3(&ThreeDimReconstruction::wait, threeDimRec);

	t1.join();
	t2.join();

	
	return 0;
}

