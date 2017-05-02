#pragma once


#include <opencv2/core.hpp>
#include <opencv2/nonfree/nonfree.hpp>

namespace cv {
	class SIFTFeature {
	public:
		SIFTFeature();
		SIFTFeature(KeyPoint keypoint, Mat descriptor);
		// Properties
		KeyPoint keypoint;
		Mat descriptor;	// A row of matrix of uchar type
	};
}