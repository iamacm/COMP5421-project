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

int main2(int argc, char** argv)
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

