#include "stdafx.h"
#include "SIFTFeature.h"

namespace cv {
	SIFTFeature::SIFTFeature(cv::KeyPoint keypoint, cv::Mat descriptor) :
		keypoint(keypoint), descriptor(descriptor) {

	}
}

