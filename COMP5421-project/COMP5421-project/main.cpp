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
		cout << "Could not open or find the image " << argv[1] << endl;
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
	if (argc < 3)
	{
		char* files[] = { 
			"0000.png", 
			"0001.png", 
			"0002.png", 
			"0003.png", 
			"0004.png", 
			"0005.png", 
			"0006.png", 
			"0007.png", 
			"0008.png", 
			"0009.png", 
			"0010.png" 
		};

		threeDimRec = new ThreeDimReconstruction(files, sizeof(files) / sizeof(char*));
	}
	else {
		const int count = argc - 1;
		threeDimRec = new ThreeDimReconstruction(&argv[1], count);
	}

	thread t1(&ThreeDimReconstruction::showOriginalImg, threeDimRec);
	thread t2(&ThreeDimReconstruction::process, threeDimRec);
	//thread t3(&ThreeDimReconstruction::wait, threeDimRec);

	t1.join();
	t2.join();

	
	return 0;
}

