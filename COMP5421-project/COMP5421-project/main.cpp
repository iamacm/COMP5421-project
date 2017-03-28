// COMP5421-project.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "ThreeDimReconstruction.h"

using namespace cv;
using namespace std;

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
	threeDimRec->showAll();
	threeDimRec->wait();

	
	return 0;
}

