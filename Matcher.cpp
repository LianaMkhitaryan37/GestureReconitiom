#include "Matcher.h"


Matcher::Matcher(const cv::Mat & im, int nF)
	:m_image(im)
	, m_MAX_FEATURES(nF)
	, m_orb(cv::ORB::create(m_MAX_FEATURES))
	, m_matcher(cv::DescriptorMatcher::create("BruteForce-Hamming"))
{
	m_orb->detectAndCompute(m_image, cv::Mat(), m_keypoints, m_descriptors);
}

unsigned _int64 Matcher::compare(const cv::Mat & cim, const char * name , bool save)
{
	cv::Mat descriptors2;
	std::vector<cv::KeyPoint> keypoints2;
	std::vector< std::vector<cv::DMatch> > matches;
	m_orb->detectAndCompute(cim, cv::Mat(), keypoints2, descriptors2);
	m_matcher->knnMatch(descriptors2,m_descriptors,  matches, 30);

	double tresholdDist = 25;
	double tresholdangel = 10;
	const double PI = 3.14159265;
	std::vector< cv::DMatch > good_matches;
	good_matches.reserve(matches.size());
	for (size_t i = 0; i < matches.size(); ++i)
	{
		for (size_t j = 0; j < matches[i].size(); j++)
		{
			//calculate local distance for each possible match
			cv::Point2f from = keypoints2[matches[i][j].queryIdx].pt;
			cv::Point2f to = m_keypoints[matches[i][j].trainIdx].pt;
			double dist = sqrt((from.x - to.x) * (from.x - to.x) + (from.y - to.y) * (from.y - to.y));
			double angel = atan2((from.y - to.y), (from.x - to.x)) * 180 / PI;
			//std::cout << dist<< std::endl;
			if (angel < tresholdangel && angel > -1 * tresholdangel && dist<tresholdDist)
			{
				good_matches.push_back(matches[i][j]);
				j = matches[i].size();
			}
		}

	}
	if (save) {
		cv::Mat imMatches;
		drawMatches(m_image, m_keypoints, cim, keypoints2, good_matches, imMatches);
		cv::imwrite(name, imMatches);
	}
	const double all = (double)matches.size();
	return (unsigned _int64)(100 * good_matches.size() / all + 0.5);
}


