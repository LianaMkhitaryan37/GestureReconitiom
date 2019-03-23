#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat bg, bgU;
void run_avg(const Mat& img_gray, double avg);
bool segment(const Mat & img_gray, vector<vector<Point> > & out_segment, Mat & out_thresholded, size_t treshold = 25);

int main()
{

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
	char a[40];
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
		cvtColor(img_roi, img_gray, COLOR_BGR2GRAY);


		GaussianBlur(img_gray, img_gray, Size(7, 7), 0.0, 0);
		if (num_frames < 30)
			run_avg(img_gray, aWeight);
		else {

			if (segment(img_gray, img_segment, img_threshold)) {
				drawContours(img, img_segment, -1, Scalar(0, 255, 0), 1, 8, noArray(), 2000, Point(340, 100));
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

void run_avg(const Mat & img_gray, double avg)
{
	if (bg.empty()) {
		img_gray.convertTo(bg, CV_32FC1);
		return;
	}
	accumulateWeighted(img_gray, bg, avg);
}

bool segment(const Mat & img_gray, vector<vector<Point> > & out_segment, Mat & out_thresholded, size_t treshold)
{
	Mat diff;
	if (bgU.empty())
		bg.convertTo(bgU, CV_8UC1);

	absdiff(img_gray, bgU, diff);


	threshold(diff, out_thresholded, treshold, 255, THRESH_BINARY);

	findContours(out_thresholded, out_segment, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	if (out_segment.empty())
		return false;
	int indexOfBiggestContour = -1;
	size_t sizeOfBiggestContour = 0;

	for (size_t i = 0; i < out_segment.size(); i++)
		if (out_segment[i].size() > sizeOfBiggestContour) {
			sizeOfBiggestContour = out_segment[i].size();
			indexOfBiggestContour = i;
		}

}
