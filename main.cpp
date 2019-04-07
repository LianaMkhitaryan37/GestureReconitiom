#include <iostream>
#include <vector>
#include <string>
#include <math.h> 
#include "Matcher.h" 
#include "preprocessing.h"

void ReadingData(std::vector<cv::String>& fn, std::vector<cv::Mat>& data);

int main()
{
	cv::Mat test;
	Preprocessing cam(test);
	cam.start();
	cv::imshow("Processing Image", test);
	//cv::Mat test = cv::imread("test/median.jpg");
	std::vector<cv::String> fn;
	std::vector<cv::Mat> data;
	ReadingData(fn, data);

	cv::Mat imReg;
	unsigned _int64 h;
	std::cout << "Matching images ..." << std::endl;

	unsigned _int64 max = 0;
	std::vector<size_t> elems;
	Matcher cmp(test);
	for (size_t i = 0; i < data.size(); ++i)
	{
		h = cmp.compare(data[i]);
		std::cout << h << std::endl;
		if (h > max)
		{
			elems.clear();
			max = h;
			elems.push_back(i);

		}
		else if (h == max) {
			elems.push_back(i);
		}

	}

	for (size_t e : elems)
		imshow(fn[e], data[e]);


	cv::waitKey(0);
	return 0;
}
void ReadingData(std::vector<cv::String>& fn, std::vector<cv::Mat>& data)
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