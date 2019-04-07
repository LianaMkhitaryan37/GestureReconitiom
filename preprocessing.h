#pragma once
#include <opencv2/opencv.hpp>

class Preprocessing
{
public:
	Preprocessing(cv::Mat & src);
	/**
	* \brief connects to cam,make sharpening,binarization and noise removing on each frame;
	* \param[in]  ksize - kernel size for noise removing
	* \param[in]  thresh - if value > thresh set it to maxVal,otherwise 0
	*/
	void start(int ksize=5,double thresh=40, double maxVal = 255);
private:
	void sharpening();
	/**
	* \brief uses otsu method for image binarization
	* \param[in]  thresh - if value > thresh set it to maxVal,otherwise 0
	*/
	void binarize(double thresh, double maxVal);
	/**
	* \brief uses medianblur for removing noise and shows it
	* \param[in]  ksize - kernel size for filter
	*/
	void removeNoise(int ksize);
private:
	cv::Mat & m_result;
};

