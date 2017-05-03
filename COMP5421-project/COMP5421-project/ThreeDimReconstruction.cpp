#include "stdafx.h"
#include <windows.h>
#include <algorithm> 
#include <stdint.h>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "ThreeDimReconstruction.h"
#include "SIFTFeature.h"

#define	SCREEN_WIDTH				GetSystemMetrics(SM_CXSCREEN)
#define	SCREEN_HEIGHT				GetSystemMetrics(SM_CYSCREEN)
#define IMAGE_WRITE_FOLDER			"results/"
#define PLY_WRITE_FOLDER			"ply/"
// Global function
int euclideanDistanceSquared(const Mat& mat1, const Mat& mat2) {
	if (mat1.cols != mat2.cols) {
		throw Exception();
	}
	int sum = 0;
	for (int i = 0; i < mat1.cols; ++i) {
		const int diff = mat1.at<uchar>(0, i) - mat2.at<uchar>(0, i);
		sum += (diff * diff);
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

// Calculate the vector of the epipolar line based on F
Mat calculateEpipolarLine(const Mat& F, Point2f pt) {
	// l' = Fu is the epipolar line, where u is (x, y, 1)
	Mat u(3, 1, CV_64FC1);
	u.at<double>(0, 0) = pt.x;
	u.at<double>(1, 0) = pt.y;
	u.at<double>(2, 0) = 1;

	return F * u;
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
	this->name = "Image " + this->path.substr(this->path.length() - 8);

}

ThreeDimReconstruction::Img ThreeDimReconstruction::Img::clone() const {
	return ThreeDimReconstruction::Img(*this);
}

// Display the image of ID with aspect ratio kept
void ThreeDimReconstruction::Img::show(double resizeRatio) const {
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
void ThreeDimReconstruction::Img::showWith(ThreeDimReconstruction::Img anotherImg, double resizeRatio) const {
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

ThreeDimReconstruction::ThreeDimReconstruction(const char* const * imgPaths, int count) {
	for (int i = 0; i < count; ++i) {
		this->images.push_back(ThreeDimReconstruction::Img(imgPaths[i]));
	}
	printf("Loaded %d files successfully!\n", count);
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

ThreeDimReconstruction::Img ThreeDimReconstruction::processHarrisCorner(const Img& img) {
	/*
	Img imgFiltered;
	imgFiltered.name = "Image blurred";
	int kernel_size = 100;
	Mat kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (double)(kernel_size*kernel_size);
	filter2D(this->img[0].mat, imgFiltered.mat, -1, kernel);
	this->show(imgFiltered);
	this->wait();
	*/
	
	//this->wait();
	//const int imgCount = this->images.size();
	//vector<Point> imgCorners;
	Img imgCornersCircled;


	vector<Point2d> corners = ThreeDimReconstruction::FeatureDetectors::detectHarrisCorner(img, false);
	imgCornersCircled.mat = img.mat.clone();
	for (const Point pt : corners) {
		circle(imgCornersCircled.mat, pt, 15, Scalar(0, 0, 255), 5);
	}
	imgCornersCircled.name = img.name + " with corners circled";

	return imgCornersCircled;
}


ThreeDimReconstruction::Img ThreeDimReconstruction::visualizeFeatures(const Img& img, const vector<SIFTFeature>& features) const {
	Img imgWithFeatures = img.clone();

	for (const auto& feature : features) {
		circle(imgWithFeatures.mat, feature.keypoint.pt, cvRound(feature.keypoint.size*1.0), Scalar(0, 255, 0), 3);
	}

	imgWithFeatures.name += " SIFT features";
	imgWithFeatures.show();
	//circle(matchingImage, kp.pt + Point2f(newColorImage.size().width, newColorImage.size().height), cvRound(kp.size*1.0), Scalar(0, 255, 0), 5, 8, 0);
	return imgWithFeatures;
}



ThreeDimReconstruction::Img ThreeDimReconstruction::visualizeMatchings(const Img& img1, const Img& img2, const vector<pair<SIFTFeature, SIFTFeature>>& matchings) {
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

	return matchingImg;
}


ThreeDimReconstruction::Img ThreeDimReconstruction::visualizeMatchingWithEpipolarLines(const Img& img1, const Img& img2, const vector<pair<SIFTFeature, SIFTFeature>>& matchings, const Mat& F) {
	const Mat& Ft = F.t();
	Img matchingImg;
	matchingImg.name = "Matching of " + img1.name + " & " + img2.name + " with epipolar lines";
	Size matchingImgsize(img1.mat.size().width + img2.mat.size().width, max(img1.mat.size().height, img2.mat.size().height));
	matchingImg.mat = Mat::zeros(matchingImgsize, CV_8UC3);	// Creating image of size

	// Copy images to the matching image
	img1.mat.copyTo(matchingImg.mat(Rect(0, 0, img1.mat.cols, img1.mat.rows)));
	img2.mat.copyTo(matchingImg.mat(Rect(img1.mat.cols, 0, img2.mat.cols, img2.mat.rows)));

	// Circle keypoints and draw lines
	int count = 0;

	for (const pair<SIFTFeature, SIFTFeature>& matching : matchings) {
		const SIFTFeature& feature1 = matching.first;
		const SIFTFeature& feature2 = matching.second;
		const Scalar color(128*(count & 0x4) + 127, 128 *(count & 0x2) + 127, 128 * (count & 0x1) + 127);

		// Circle feature1
		circle(matchingImg.mat, feature1.keypoint.pt, 20, color, -1);
		// Circle feature2
		circle(matchingImg.mat, feature2.keypoint.pt + Point2f(img1.mat.cols, 0), 20, color, -1);
		// Draw line between the two features
		//line(matchingImg.mat, feature1.keypoint.pt, feature2.keypoint.pt + Point2f(img1.mat.size().width, 0), Scalar(0, 0, 255), 3);

		// Draw epipolar lines
		// l' = Fu = (a', b', c')t
		const Mat lp = calculateEpipolarLine(F, feature1.keypoint.pt);
		// i.e., a'x + b'y + c = 0 is the line l'
		// Then, draw the line from point (0, -c'/b') to (cols, -(a'cols+c')/b'
		double ap = lp.at<double>(0, 0), bp = lp.at<double>(1, 0), cp = lp.at<double>(2, 0);
		line(matchingImg.mat, Point(0 + img1.mat.cols, -cp / bp), Point(img2.mat.cols + img1.mat.cols, -(ap * img1.mat.cols + cp) / bp), color, 3);

		//printf("Haha test: u'l' = %f\n", feature2.keypoint.pt.x * ap + feature2.keypoint.pt.y * bp + 1 * cp);
		// Draw epipolar lines
		const Mat l = calculateEpipolarLine(Ft, feature2.keypoint.pt);	// l = Ftu', simiular to l'
		double a = l.at<double>(0, 0), b = l.at<double>(1, 0), c = l.at<double>(2, 0);
		line(matchingImg.mat, Point(0, -c / b), Point(img1.mat.cols, -(a * img1.mat.cols + c) / b), color, 3);

		++count;
	}

	return matchingImg;
}

vector<pair<SIFTFeature, SIFTFeature>> ThreeDimReconstruction::SIFTFeatureMatching(const Img& img1, const vector<SIFTFeature> features1, const Img& img2, const vector<SIFTFeature> features2) {
	// For each feature of mat1, find the best feature to be matched with mat2
	vector<pair<SIFTFeature, SIFTFeature>> matchings;
	const double RATIO_REQUIRED = 0.75;	// For ratio test


	for (int feature1Id = 0; feature1Id < features1.size(); ++feature1Id) {
		const SIFTFeature& feature = features1[feature1Id];
		const int img2MatchedFeatureId = nearestNeighbor(feature, features2);
		const SIFTFeature& img2MatchedFeature = features2[img2MatchedFeatureId];

		double ratio = feature.keypoint.size / img2MatchedFeature.keypoint.size;

		if (nearestNeighbor(img2MatchedFeature, features1) == feature1Id && 
			ratio >= RATIO_REQUIRED && ratio <= 1.0 / RATIO_REQUIRED	// Ratio test passed
			) {
			//printf("Ratio test: %f\t%f\n", feature.keypoint.size, img2MatchedFeature.keypoint.size);
			matchings.push_back(make_pair(feature, img2MatchedFeature));
		}


	}

	// Outlier check by fundamental matrix ransac 
	vector<Point2f> points1(matchings.size());
	vector<Point2f> points2(matchings.size());
	for (int i = 0; i < matchings.size(); ++i) {
		points1[i] = matchings[i].first.keypoint.pt;
		points2[i] = matchings[i].second.keypoint.pt;
	}
	vector<uchar> ransacMask;
	findFundamentalMat(points1, points2, ransacMask, FM_RANSAC);
	
	// Delete outliers
	for (int i = matchings.size() - 1; i >= 0; --i) {
		if (ransacMask[i] != 1) {
			//matchings.erase(matchings.begin() + i);
		}
	}


	// Sort by distance diff
	sort(matchings.begin(), matchings.end(), [](const pair<SIFTFeature, SIFTFeature>& prev, const pair<SIFTFeature, SIFTFeature>& next) {
		int prevDistance = euclideanDistanceSquared(prev.first.descriptor, prev.second.descriptor);
		int nextDistance = euclideanDistanceSquared(next.first.descriptor, next.second.descriptor);
		return prevDistance < nextDistance;
	});

	return matchings;
}



// Output the fundamental matrix F
// If N == 7, seven-point algorithm is used
// If N >= 8, eight-point algorithm is used
Mat ThreeDimReconstruction::computeFundamentalMatrix(const vector<pair<SIFTFeature, SIFTFeature>>& matchings, const int N) {
	Mat fundamentalMatrix(3, 3, CV_64FC1);

	if (N < 7 || matchings.size() < N) {
		throw Exception();
	}

	Mat A(N, 9, CV_64FC1);

	for (int n = 0; n < N; ++n) {
		const Point2f& point1 = matchings[n].first.keypoint.pt;
		const Point2f& point2 = matchings[n].second.keypoint.pt;
		// Epipolar constraints u1Fu2 = 0, where u = (x, y, 1) and up = (xp, yp, 1)
		// Eqivalent to AF = 0, where
		// A(i) = (xpx, xpy, xp, ypx, ypy, yp, x, y, 1)
		const double u[3] = { point1.x, point1.y, 1 };
		const double up[3] = { point2.x, point2.y, 1 };
		

		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				A.at<double>(n, i * 3 + j) = up[i] * u[j];
			}
		}
	}

	cout << "A:" << A << endl;

	// Apply SVD: A = UDVt
	Mat U, D, Vt;
	SVD::compute(A, D, U, Vt);

	
	// Get the column of the the smallest singular value of V, i.e. the row of least v of Vt
	// In fact, it is the last row of Vt
	cout << "U:" << endl << U << endl;
	cout << "D:" << endl << D << endl;
	cout << "Vt:" << endl << Vt << endl;

	if (N == 7) {
		// 7-point algorithm
		// Get the TWO singular vectors fa, fb of the smallest singular value
		Mat faTmp = Vt.row(Vt.rows - 1).t();
		Mat fbTmp = Vt.row(Vt.rows - 2).t();
		Mat fa(3, 3, CV_64FC1), fb(3, 3, CV_64FC1);;

		// Remake fa and fb from 1 x 9 back to 3 x 3
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				fa.at<double>(i, j) = faTmp.at<double>(i * 3 + j, 0);
				fb.at<double>(i, j) = fbTmp.at<double>(i * 3 + j, 0);
			}
		}

		// Consider the cubic equation ax^3 + bx^2t + cxt^2 + dt^3 = 0
		// by det(x fa + t fb) = 0
		double a = 0, b = 0, c = 0, d = 0;	// Coefficient of the cubic equation
		// a: Determinant of all elements from fa
		a = determinant(fa);	
		// b: Determinant of TWO elements from fa + ONE element from fb
		for (int i = 0; i < 3; ++i) {
			Mat faTmp = fa.clone();

			// Get one row from fb
			for (int p = 0; p < 3; ++p) {
				faTmp.at<double>(i, p) = fb.at<double>(i, p);
			}

			b += determinant(faTmp);
		}

		// c: Determinant of TWO elements from fb + ONE element from fa
		for (int i = 0; i < 3; ++i) {
			Mat fbTmp = fa.clone();
			// Get one row from fb
			for (int p = 0; p < 3; ++p) {
				fbTmp.at<double>(i, p) = fa.at<double>(i, p);
			}
			c += determinant(fbTmp);
		}

		// Determinant of all elements from fb
		d = determinant(fb);	

		double coefficients[4] = { a, b, c, d };
		double roots[3];
		solveCubic(Mat(1, 4, CV_64F, coefficients), Mat(1, 3, CV_64F, roots));

		// The root is x/t
		cout << "fa: " << fa << endl;
		cout << "fb: " << fb << endl;
		cout << "roots: " << roots[0] << "\t" << roots[1] << "\t" << roots[2] << endl;
		// F = x fa + t fb
		// <=> F' = (x/t) fa + fb, to be normalized
		fundamentalMatrix = fa.mul(roots[0]) + fb;

	} else if (N >= 8) {
		// 8-point algorithm
		// F' = the last column (i.e., with least singular value) of V
		Mat FpTmp = Vt.row(Vt.rows - 1).t();
		// Remake Fp to be from 1 x 9 back to 3 x 3
		Mat Fp(3, 3, CV_64FC1);

		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				Fp.at<double>(i, j) = FpTmp.at<double>(i * 3 + j, 0);
			}
		}

		cout << "Fp:" << endl << Fp << endl;

		// Apply SVD again: Fp = UpDpVpt
		Mat Up, Dp, Vpt;
		SVD::compute(Fp, Dp, Up, Vpt, SVD::FULL_UV);

		Mat DpTmp = Mat::zeros(3, 3, CV_64FC1);
		DpTmp.at<double>(0, 0) = Dp.at<double>(0, 0);
		DpTmp.at<double>(1, 1) = Dp.at<double>(1, 0);
		// Set the value of the least singular value to be 0 (i.e., the last element)
		//DpTmp.at<double>(2, 2) = Dp.at<double>(2, 0);

		cout << "Up: " << Up << endl;
		cout << "Dp: " << DpTmp << endl;
		cout << "Vpt: " << Vpt << endl;

		// F = U'D''V't
		fundamentalMatrix = Up * DpTmp * Vpt;

		
		cout << "F" << fundamentalMatrix << endl;
		//cout << "Ftmp: " << fundamentalMatrixTmp << endl;

	}

	// Normalize F such that the last element must be 1
	double normalizationFactor = 1.0 / fundamentalMatrix.at<double>(2, 2);
	fundamentalMatrix.mul(normalizationFactor);

	return fundamentalMatrix;
}


// Rows of 3D points
Mat ThreeDimReconstruction::twoViewTriangulation(const vector<pair<SIFTFeature, SIFTFeature>>& matchings, const Mat& F) {
	if (matchings.size() < 5) {
		throw Exception();
	}
	// Intrinsic parameter K and Kp
	Mat K = Mat::zeros(3, 3, CV_64FC1);
	K.at<double>(0, 0) = 2759.48;
	K.at<double>(1, 1) = 2764.16;
	K.at<double>(0, 2) = 1520.69;
	K.at<double>(1, 2) = 1006.81;
	K.at<double>(2, 2) = 1;
	Mat Kp = K.clone();

	// Compute the inverse matrix
	Mat KInv = K.inv();
	Mat KpInv = Kp.inv();

	cout << "K: " << K << endl;
	cout << "KInv: " << KInv << endl;

	// Essential matrix
	// E = K't F K
	Mat E = Kp.t() * F * K;

	cout << "E: " << E << endl;
	Mat U, Wdiag, Vt;
	// SVD: E = UWVt
	SVD::compute(E, Wdiag, U, Vt, SVD::FULL_UV);

	Mat W = Mat::zeros(3, 3, CV_64FC1);
	W.at<double>(0, 0) = Wdiag.at<double>(0, 0);
	W.at<double>(1, 1) = Wdiag.at<double>(1, 0);
	W.at<double>(2, 2) = Wdiag.at<double>(2, 0);

	//cout << "U: " << U << endl;
	//cout << "W: " << W << endl;
	//cout << "Vt: " << Vt << endl;

	/*
	Mat Diag = Mat::zeros(3, 3, CV_64F);
	// Diag = (1 0 0; 0 1 0; 0 0 0)
	Diag.at<double>(0, 0) = 1;
	Diag.at<double>(1, 1) = 1;

	E = U * Diag * Vt;
	*/
	SVD::compute(E, Wdiag, U, Vt, SVD::FULL_UV);
	cout << "U: " << U << endl;
	cout << "Wdiag: " << Wdiag << endl;
	cout << "Vt: " << Vt << endl;

	// D = (0 1 0; -1 0 0; 0 0 1)
	Mat D = Mat::zeros(3, 3, CV_64F);
	D.at<double>(0, 1) = 1;
	D.at<double>(1, 0) = -1;
	D.at<double>(2, 2) = 1;

	// TWO possible values of R: R = UDVt or UDtVt
	Mat R[2] = { U * D * Vt, U * D.t() * Vt };
	// TWO possible values of t: +/- the last column vector of U
	Mat t[2] = { U.col(U.cols - 1), U.col(U.cols - 1).mul(-1) };

	// Choose the R and t such that both 3D points are positive
	Mat testUp(3, 1, CV_64F);
	testUp.at<double>(0, 0) = matchings[0].second.keypoint.pt.x;
	testUp.at<double>(1, 0) = matchings[0].second.keypoint.pt.y;
	testUp.at<double>(2, 0) = 1;
	// Choose the first matched feature
	Mat testXp = KpInv * testUp;

	cout << "testXp: " << testXp << endl;

	Mat finalR = R[0], finalt = t[0];
	// X = RX' + t
	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 2; ++j) {
			Mat testX = R[i] * testXp + t[j];
			//cout << "tj: " << t[j] << endl;
			//cout << "testX: " << testX << endl;

			bool sameSignTest =
				testX.at<double>(0, 0) * testXp.at<double>(0, 0) > 0 &&
				testX.at<double>(1, 0) * testXp.at<double>(1, 0) > 0 &&
				testX.at<double>(2, 0) * testXp.at<double>(2, 0) > 0;

			cout << "u: " << K * testX << endl;
			cout << "sameSignTest: " << sameSignTest << endl;

			if (sameSignTest) {
				finalR = R[i];
				finalt = t[j];
			}
		}
	}


	Mat p1 = K * Mat::eye(3, 4, CV_64F);	// p1 = K * (I | 0)
	Mat p2;
	hconcat(finalR, finalt, p2);		// p2 = K * (R | t)
	p2 = K * p2;
	Mat points4D;

	vector<Point2f> projPoints1, projPoints2;

	for (const pair<SIFTFeature, SIFTFeature>& matching : matchings) {
		projPoints1.push_back(matching.first.keypoint.pt);
		projPoints2.push_back(matching.second.keypoint.pt);
	}


	triangulatePoints(p1, p2, projPoints1, projPoints2, points4D);

	//cout << "points4D: " << points4D << endl;

	Mat points3D, goodPoints3D;
	convertPointsFromHomogeneous(points4D.t(), points3D);

	//cout << "points3D: " << points3D << endl;

	// Outlier check
	double outlierThreshold = 20.0;
	for (int i = 0; i < points3D.rows; ++i) {
		Mat point3D = points3D.row(i);
		double distance = norm(point3D);

		if (distance <= outlierThreshold && point3D.at<float>(0, 2) >= 0) {
			goodPoints3D.push_back(point3D);
		}
	}
	return goodPoints3D;
	/*
	// 5-point algorithm
	Mat A(5, 9, CV_64FC1);
	for (int n = 0; n < 5; ++n) {
		Point2f point1 = matchings[n].first.keypoint.pt;
		Point2f point2 = matchings[n].second.keypoint.pt;

		Mat u(3, 1, CV_64FC1, { point1.x, point1.y, 1 });		// (x, y, 1)
		Mat up(3, 1, CV_64FC1, { point2.x, point2.y, 1 });		// (x', y', 1)

		u.at<double>(0, 0) = point1.x;
		u.at<double>(1, 0) = point1.y;
		u.at<double>(2, 0) = 1;
		up.at<double>(0, 0) = point2.x;
		up.at<double>(1, 0) = point2.y;
		up.at<double>(2, 0) = 1;

		cout << "u: " << u << endl;

		// x = K-1u
		Mat x = KInv * u;
		// x' = K'-1u'
		Mat xp = KpInv * up;

		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				A.at<double>(n, i * 3 + j) = xp.at<double>(i, 0) * x.at<double>(j, 0);
			}
		}
	}

	// Apply SVD: A = UDVt
	Mat U, D, Vt;
	SVD::compute(A, D, U, Vt);



	// Get the column of the the smallest singular value of V, i.e. the row of least v of Vt
	// In fact, it is the last row of Vt
	cout << "U:" << endl << U << endl;
	cout << "D:" << endl << D << endl;
	cout << "Vt:" << endl << Vt << endl;

	*/
	
}


void ThreeDimReconstruction::writePly(const string& file, const Mat& points3D) {
	ofstream of(file);
	if (of.is_open())
	{
		of << "ply" << endl;
		of << "format ascii 1.0" << endl;
		of << "comment VTK generated PLY File" << endl;
		of << "obj_info vtkPolyData points and polygons: vtk4.0" << endl;
		of << "element vertex " << points3D.rows << endl;
		of << "property float x" << endl;
		of << "property float y" << endl;
		of << "property float z" << endl;
		of << "element face 0" << endl;
		of << "property list uchar int vertex_indices" << endl;
		of << "end_header" << endl;

		for (int i = 0; i < points3D.rows; ++i) {
			of << points3D.at<float>(i, 0) << "\t" << points3D.at<float>(i, 1) << "\t" << points3D.at<float>(i, 2) << endl;
		}
		of.close();
	}
	else cout << "Unable to open file " + file;
}

void ThreeDimReconstruction::process(void) {
	vector<vector<SIFTFeature>> featuresOfImages;

	for (const Img& img : this->images) {
		// Harris Corner
		Img visualizeCornersImg =  processHarrisCorner(img);
		visualizeCornersImg.show();
		imwrite(IMAGE_WRITE_FOLDER + visualizeCornersImg.name + ".jpg", visualizeCornersImg.mat);
		// SIFT
		SIFTFeature feature;
		vector<SIFTFeature> features = FeatureDetectors::detectSIFT(img);

		printf("%d SIFT feature(s) found in %s\n", features.size(), img.name);

		Img visualizeFeaturesImg = visualizeFeatures(img, features);
		visualizeFeaturesImg.show();
		imwrite(IMAGE_WRITE_FOLDER + visualizeFeaturesImg.name + ".jpg", visualizeFeaturesImg.mat);
		featuresOfImages.push_back(features);
	}

	if (this->images.size() >= 2) {

		vector<pair<SIFTFeature, SIFTFeature>> matchings = SIFTFeatureMatching(this->images[0], featuresOfImages[0], this->images[1], featuresOfImages[1]);
		vector<pair<SIFTFeature, SIFTFeature>> topMatchings = matchings;
		//visualizeFeatures(this->images[0], featuresOfImages[0]);
		//visualizeFeatures(this->images[1], featuresOfImages[1]);
		cout << matchings.size() << " features matched!" << endl;

		
		topMatchings.resize(50);	// Top-50 matches

		

		int outlierCount = 0, iteration = 0;

		vector<Point2f> points1(matchings.size());
		vector<Point2f> points2(matchings.size());
		for (int i = 0; i < matchings.size(); ++i) {
			points1[i] = matchings[i].first.keypoint.pt;
			points2[i] = matchings[i].second.keypoint.pt;
		}
		Mat fundamentalMatrixRansac = findFundamentalMat(points1, points2, FM_RANSAC);
		Mat fundamentalMatrix = computeFundamentalMatrix(matchings, 20);	// Change >= 8 for eight-point algorithm, 7 for seven-point algorithm;


		// Remove outlier by checking u'tFu
		outlierCount = 0;
		for (auto it = matchings.begin(); it != matchings.end(); ) {
			const auto& matching = *it;
			bool outlier = false;
			Mat up(3, 1, CV_64FC1), u(3, 1, CV_64FC1);
			up.at<double>(0, 0) = matching.second.keypoint.pt.x;
			up.at<double>(1, 0) = matching.second.keypoint.pt.y;
			up.at<double>(2, 0) = 1;
			u.at<double>(0, 0) = matching.first.keypoint.pt.x;
			u.at<double>(1, 0) = matching.first.keypoint.pt.y;
			u.at<double>(2, 0) = 1;

			Mat uptFu = up.t() * fundamentalMatrix * u;
			if (abs(uptFu.at<double>(0, 0)) > 1.0) {
				outlier = true;
				++outlierCount;
			}

			if (outlier) {
				// Remove outlier
				cout << "Outlier: u'tFu: " << uptFu << endl;
				matchings.erase(it);
			}
			else {
				++it;
			}
		}
		++iteration;

		for (auto& matching : topMatchings) {
			//printf("%f\n", sqrt(euclideanDistanceSquared(matching.first.descriptor, matching.second.descriptor)));
			cout << matching.first.keypoint.pt << "\t" << matching.second.keypoint.pt << endl;
		}
		Img visualizeMatchingsImg = visualizeMatchings(this->images[0], this->images[1], topMatchings);
		visualizeMatchingsImg.show(0.9);
		imwrite(IMAGE_WRITE_FOLDER + visualizeMatchingsImg.name + ".jpg", visualizeMatchingsImg.mat);


		Img visualizeMatchingWithEpipolarLinesImg = visualizeMatchingWithEpipolarLines(this->images[0], this->images[1], topMatchings, fundamentalMatrix);
		visualizeMatchingWithEpipolarLinesImg.show(0.9);
		imwrite(IMAGE_WRITE_FOLDER + visualizeMatchingWithEpipolarLinesImg.name + ".jpg", visualizeMatchingWithEpipolarLinesImg.mat);


		
		
		

		Mat points3D = twoViewTriangulation(matchings, fundamentalMatrixRansac);
		writePly(PLY_WRITE_FOLDER + this->images[0].name + "_" + this->images[1].name + ".ply", points3D);
		this->wait();
	}

	
	

}

