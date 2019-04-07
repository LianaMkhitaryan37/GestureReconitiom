#pragma once
#include <opencv2/opencv.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

class Matcher
{
public:
	/**
	* \brief sets Base grayscale image with which others must be compared 
	* \param[in]  im - Base grayscale image
	* \param[in] nF - max number of features that can be find in images
	*/
	Matcher(const cv::Mat & im,int nF=500);
	/**
	* \brief  compares given image with setted img using feature matching
	* \param[in]  cim - grayscale image need to be compared
	* \param[in]  save - if true draws matches and save it in  file
	* \param[in]  name - filename where matches are saved
	* \return  Percentage of matching
	*/
	unsigned _int64 compare(const cv::Mat & cim, const char * name = "matches.jpg",bool save=false);
private:
	const cv::Mat & m_image;
	const int m_MAX_FEATURES;
	cv::Ptr<cv::Feature2D> m_orb;
	cv::Ptr<cv::DescriptorMatcher> m_matcher;
	std::vector<cv::KeyPoint> m_keypoints;
	cv::Mat m_descriptors;
};

