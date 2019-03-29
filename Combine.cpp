#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"
#include <iostream>
#include <vector>
#include <string>
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
	Mat src, kernel,imgLaplacian,sharp,imgResult, frame;
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

void compare(Mat &im1, Mat &im2, Mat &imMatches, double& pers_out)
{

	// Convert images to grayscale
	Mat im1Gray=im1, im2Gray=im2;
	//cvtColor(im1, im1Gray, COLOR_BGR2GRAY);
	//cvtColor(im2, im2Gray, COLOR_BGR2GRAY);

	// Variables to store keypoints and descriptors
	std::vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;

	// Detect ORB features and compute descriptors.
	Ptr<Feature2D> orb = ORB::create(MAX_FEATURES);
	orb->detectAndCompute(im1Gray, Mat(), keypoints1, descriptors1);
	orb->detectAndCompute(im2Gray, Mat(), keypoints2, descriptors2);
	//std::cout << "keypoints (" << keypoints1.size() << " , " << keypoints2.size() << std::endl;

	// Match features.
	std::vector<DMatch> matches;
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	matcher->match(descriptors1, descriptors2, matches, Mat());
	const double all = (double)matches.size();
	const double maxDist = matches[0].distance + 5;
	std::vector<DMatch> good_matches;
	for (size_t i = 0; i < matches.size(); i++)
	{
		if (matches[i].distance < maxDist)
		{
			good_matches.push_back(matches[i]);
		}
	}


	pers_out = 100*good_matches.size() / all;
	//cout << "GOOD_MATCH_PERCENT = " << good_matches.size() << "_" << pers_out << std::endl;

	// Draw top matches
	//Mat imMatches;
	drawMatches(im1, keypoints1, im2, keypoints2, good_matches, imMatches);

}
int main()
{
	//Mat test;
	//VidCapture(test);
	//imshow("Median Image", test);
	Mat test = imread("IMG.jpg");
	vector<cv::String> fn;
	vector<cv::Mat> data;
	ReadingData(fn, data);

	Mat imReg;
	double h;
	// Align images
	cout << "Matching images ..." << endl;

	double max = 0;
	int needed_img = 0;

	for (int i = 0; i < data.size(); ++i)
	{
		compare(data[i], test, imReg, h);
		std::cout << h << std::endl;
		if (h > max)
		{
			needed_img = i;
			max = h;
		}

	}

	cout << fn[needed_img]<<"_"<<max<< "_" << needed_img << endl;
	imshow(fn[needed_img], data[needed_img]);


	waitKey(0);
	return 0;
}