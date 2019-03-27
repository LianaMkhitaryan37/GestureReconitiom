
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;


int main()
{
	VideoCapture cam(0);
	if (!cam.isOpened()) {
		cout << "ERROR not opened " << endl;
		return -1;
	}
	Mat src;
	Mat kernel;
	Mat imgLaplacian;
	Mat sharp;
	Mat imgResult;
	Mat bw;
	Mat dist;
	Mat kernel1;
	Mat dist_8u;
	Mat markers;
	Mat mark;
	Mat dst, frame;
	vector<Vec3b> colors;
	vector<vector<Point> > contours;
	while (1) {
		bool b = cam.read(frame);
		if (!b) {
			cout << "ERROR : cannot read" << endl;
			return -1;
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

		imshow("Black Background Image", src);
	
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

		imshow("New Sharped Image", imgResult);
		src = imgResult; // copy back


		cvtColor(src, bw, COLOR_BGR2GRAY);
		threshold(bw, bw, 40, 255, THRESH_BINARY | THRESH_OTSU);

		imshow("Binary Image", bw);
		// Perform the distance transform algorithm
		medianBlur(bw, bw, 5);
		imshow("Median Image", bw);
		distanceTransform(bw, dist, DIST_L2, 3);
		// Normalize the distance image for range = {0.0, 1.0}
		// so we can visualize and threshold it
		normalize(dist, dist, 0, 1., NORM_MINMAX);
		imshow("Distance Transform Image", dist);


		if (waitKey(30) == 27) return -1;
	}
	return 0;
}