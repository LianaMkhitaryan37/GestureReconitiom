#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat bg, bgU;
void run_avg(const Mat& img_gray, double avg);
bool segment(const Mat & img_gray, vector<vector<Point> > & out_segment, Mat & out_thresholded, int& index_out,double treshold = 25);
void getSkin(const Mat& input, Mat &skin);

int main()
{
	CascadeClassifier hand_cascade;
	std::vector<Rect> hands;
	

	if (!hand_cascade.load("hand.xml")) { printf("--(!)Error loading face cascade\n"); return -1; };
	double aWeight = 0.5;
	VideoCapture cam(0);
	if (!cam.isOpened()) {
		cout << "ERROR not opened " << endl;
		return -1;
	}
	Mat img;
	Mat img_threshold;
	vector<vector<Point> > img_segment;
	Mat img_gray;
	Mat img_roi;
	namedWindow("Original_image", WINDOW_AUTOSIZE);
	namedWindow("Gray_image", WINDOW_AUTOSIZE);
	namedWindow("Thresholded_image", WINDOW_AUTOSIZE);
	namedWindow("ROI", WINDOW_AUTOSIZE);

	int count = 0;
	size_t num_frames = 0;
	while (1) {
		bool b = cam.read(img);
		if (!b) {
			cout << "ERROR : cannot read" << endl;
			return -1;
		}
		Rect roi(340, 100, 270, 270);
		img_roi = img(roi);

		getSkin(img_roi, img_gray);
	
		GaussianBlur(img_gray, img_gray, Size(7, 7), 0.0, 0);

		
		if (num_frames < 30)
			run_avg(img_gray, aWeight);
		else {
			int index;
			if (segment(img_gray, img_segment, img_threshold,index,5)) {
				hand_cascade.detectMultiScale(img_roi, hands, 1.3, 5);
				for (Rect& hand : hands) {
					rectangle(img, hand, (122, 122, 0), 2);
					//rectangle(mask, hand, 255, -1);
				}
				drawContours(img, img_segment, index, Scalar(0, 255, 0), 1, 8, noArray(), 2147483647, Point(340, 100));
			}
		}
		rectangle(img, roi, Scalar(200, 255, 0));
		++num_frames;

		imshow("Original_image", img);
		imshow("Gray_image", img_gray);
		if (!img_threshold.empty())
			imshow("Thresholded_image", img_threshold);
		imshow("ROI", img_roi);
		if (waitKey(30) == 27) {
			return -1;
		}

	}
	return 0;
}
void getSkin(const Mat& input,Mat &skin)
{
	cvtColor(input, skin, COLOR_BGR2HSV);
	//HSV
	inRange(skin, Scalar(0, 30, 60), Scalar(20, 150, 255), skin);

};
void run_avg(const Mat & img_gray, double avg)
{
	if (bg.empty()) {
		img_gray.convertTo(bg, CV_32FC1);
		return;
	}
	accumulateWeighted(img_gray, bg, avg);
}
void removeBackground(const Mat& img, const Mat& bg, Mat & out, double thresholdOffset) {
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			uchar framePixel = img.at<uchar>(i, j);
			uchar bgPixel = bg.at<uchar>(i, j);

			if (framePixel >= bgPixel - thresholdOffset && framePixel <= bgPixel + thresholdOffset)
				out.at<uchar>(i, j) = 0;
			else
				out.at<uchar>(i, j) = 255;
		}
	}
}
bool segment(const Mat & img_gray, vector<vector<Point> > & out_segment, Mat & out_thresholded, int & index_out,double treshold)
{
	Mat diff(Size(img_gray.rows,img_gray.cols), CV_8UC1);
	if (bgU.empty())
		bg.convertTo(bgU, CV_8UC1);
	//Opencv versions
	//absdiff(img_gray, bgU, diff);

	//threshold(diff, out_thresholded, treshold, 255, THRESH_BINARY);

	removeBackground(img_gray, bgU, diff, treshold);
	out_thresholded = diff;
	findContours(out_thresholded, out_segment, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	if (out_segment.empty())
		return false;
	index_out = -1;
	double sizeOfBiggestContour = 0;

	for (int i = 0; i < out_segment.size(); i++) {
		double x = contourArea(out_segment[i]);
		if (x > sizeOfBiggestContour) {
			sizeOfBiggestContour = x;
			index_out = i;
		}
	}
		
	return true;
}
