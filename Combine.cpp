#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <math.h> 
const int MAX_FEATURES = 500;

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

void VidCapture(Mat & bw)
{
	VideoCapture cam(0);
	if (!cam.isOpened()) {
		cout << "ERROR not opened " << endl;
	}
	Mat src, kernel, imgLaplacian, sharp, imgResult, frame;
	while (1)
	{
		bool b = cam.read(frame);
		if (!b) {
			cout << "ERROR : cannot read" << endl;
		}
		Rect roi(340, 100, 270, 270);
		src = frame(roi);

		imshow("Source Image", src);

		for (int x = 0; x < src.rows; x++) {
			for (int y = 0; y < src.cols; y++) {
				if (src.at<Vec3b>(x, y) == Vec3b(255, 255, 255)) {
					src.at<Vec3b>(x, y)[0] = 0;
					src.at<Vec3b>(x, y)[1] = 0;
					src.at<Vec3b>(x, y)[2] = 0;
				}
			}
		}

		//imshow("Black dst Image", src);

		kernel = (Mat_<float>(3, 3) <<
			1, 1, 1,
			1, -8, 1,
			1, 1, 1);

		sharp = src; // copy source image to another temporary one
		filter2D(sharp, imgLaplacian, CV_32F, kernel);
		src.convertTo(sharp, CV_32F);
		imgResult = sharp - imgLaplacian;
		// convert back to 8bits gray scale
		imgResult.convertTo(imgResult, CV_8UC3);
		imgLaplacian.convertTo(imgLaplacian, CV_8UC3);

		//imshow("New Sharped Image", imgResult);
		src = imgResult; // copy back


		cvtColor(src, bw, COLOR_BGR2GRAY);
		threshold(bw, bw, 40, 255, THRESH_BINARY | THRESH_OTSU);

		//imshow("Binary Image", bw);
		// Perform the distance transform algorithm
		medianBlur(bw, bw, 5);
		imshow("Median Image", bw);

		if (waitKey(30) == 27)
			break;
	}
}


void ReadingData(vector<cv::String>& fn, vector<cv::Mat>& data)
{
	cv::String path("edit/*.jpg"); //select only jpg
	cv::glob(path, fn, true); // recurse
	for (size_t i = 0; i < fn.size(); ++i)
	{
		cv::Mat im = cv::imread(fn[i]);
		if (im.empty()) continue; //only proceed if sucsessful
		data.push_back(im);
	}
}

void compare(Mat &im1, Mat &im2, Mat &imMatches, unsigned _int64 & pers_out)
{

	// Convert images to grayscale
	Mat im1Gray = im1, im2Gray = im2;

	// Variables to store keypoints and descriptors
	std::vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;

	// Detect ORB features and compute descriptors.
	Ptr<Feature2D> orb = ORB::create(MAX_FEATURES);
	orb->detectAndCompute(im1Gray, Mat(), keypoints1, descriptors1);
	orb->detectAndCompute(im2Gray, Mat(), keypoints2, descriptors2);
	//std::cout << "keypoints (" << keypoints1.size() << " , " << keypoints2.size() << std::endl;
	vector< vector<DMatch> > matches;
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	matcher->knnMatch(descriptors1, descriptors2, matches, 30);

	double tresholdDist = 8;//median - 25,
	double tresholdangel = 10;
	const double PI = 3.14159265;
	vector< DMatch > good_matches;
	good_matches.reserve(matches.size());
	for (size_t i = 0; i < matches.size(); ++i)
	{
		for (size_t j = 0; j < matches[i].size(); j++)
		{
			//calculate local distance for each possible match
			Point2f from = keypoints1[matches[i][j].queryIdx].pt;
			Point2f to = keypoints2[matches[i][j].trainIdx].pt;
			double dist = sqrt((from.x - to.x) * (from.x - to.x) + (from.y - to.y) * (from.y - to.y));
			double angel = atan2((from.y - to.y), (from.x - to.x)) * 180 / PI;
			//std::cout << dist<< std::endl;
			if (angel < tresholdangel && angel > -1 * tresholdangel && dist<tresholdDist)
			{
				good_matches.push_back(matches[i][j]);
				j = matches[i].size();
			}
		}


		const double all = (double)matches.size();
		pers_out = (unsigned _int64)(100 * good_matches.size() / all + 0.5);
		drawMatches(im1, keypoints1, im2, keypoints2, good_matches, imMatches);

	}
}
int main()
{
	//Mat test;
	//VidCapture(test);
	//imshow("Median Image", test);
	Mat test = imread("edit/ok.jpg");
	vector<cv::String> fn;
	vector<cv::Mat> data;
	ReadingData(fn, data);

	Mat imReg;
	unsigned _int64 h;
	cout << "Matching images ..." << endl;

	unsigned _int64 max = 0;
	vector<size_t> elems;
	for (size_t i = 0; i < data.size(); ++i)
	{
		compare(data[i], test, imReg, h);

		imwrite("aligned.jpg", imReg);
		std::cout << h << std::endl;
		if (h > max)
		{
			elems.clear();
			max = h;
			//imwrite("align.jpg", imReg);
			elems.push_back(i);

		}
		else if (h == max) {
			elems.push_back(i);
		}

	}

	for (size_t e : elems)
		imshow(fn[e], data[e]);


	waitKey(0);
	return 0;
}