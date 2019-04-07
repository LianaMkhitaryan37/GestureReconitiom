#include "preprocessing.h"



Preprocessing::Preprocessing(cv::Mat & src)
	:m_result(src)
{
}

void Preprocessing::start(int ksize, double thresh, double maxVal)
{
	cv::VideoCapture cam(0);
	if (!cam.isOpened()) {
		std::cout << "ERROR not opened " << std::endl;
		return;
	}
	cv::Mat frame;
	while (1)
	{
		bool b = cam.read(frame);
		if (!b) {
			std::cout << "ERROR : cannot read" << std::endl;
		}
		cv::Rect roi(340, 100, 270, 270);
		m_result = frame(roi);
		cv::imshow("Source Image", m_result);
		for (int x = 0; x < m_result.rows; x++) {
			for (int y = 0; y < m_result.cols; y++) {
				if (m_result.at<cv::Vec3b>(x, y) == cv::Vec3b(255, 255, 255)) {
					m_result.at<cv::Vec3b>(x, y)[0] = 0;
					m_result.at<cv::Vec3b>(x, y)[1] = 0;
					m_result.at<cv::Vec3b>(x, y)[2] = 0;
				}
			}
		}
		sharpening();
		binarize(thresh,maxVal);
		removeNoise(ksize);

		if (cv::waitKey(30) == 27)
			break;
	}
}

void Preprocessing::sharpening()
{
	cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
		1, 1, 1,
		1, -8, 1,
		1, 1, 1);

	cv::Mat  sharp = m_result; // copy source image to another temporary one
	cv::Mat imgLaplacian, imgResult;
	filter2D(sharp, imgLaplacian, CV_32F, kernel);
	m_result.convertTo(sharp, CV_32F);
	imgResult = sharp - imgLaplacian;
	// convert back to 8bits gray scale
	imgResult.convertTo(imgResult, CV_8UC3);
	//imshow("New Sharped Image", imgResult);
	m_result = imgResult; // copy back
}

void Preprocessing::binarize(double thresh,double maxVal)
{
	cv::cvtColor(m_result, m_result, cv::COLOR_BGR2GRAY);
	cv::threshold(m_result, m_result, thresh, maxVal, cv::THRESH_BINARY | cv::THRESH_OTSU);
}

void Preprocessing::removeNoise(int ksize)
{
	cv::medianBlur(m_result, m_result, 5);
	cv::imshow("Processing Image", m_result);
}
