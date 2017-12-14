#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <Windows.h>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void improvedShotDetection(string videoFilePath, string toDir, string fileName);
void judgeOverThreshold(bool& is, double value, const double threshold);

void exampleTemplateMatching();
void testTemplateMatchingDiff();

class Frame {

public:
	Mat mImg;
	bool mIsOverColor;
	bool mIsOverBlock;
	bool mIsOverFlush;

	double mColorDiff;
	double mBlockHistDiff;
	double mTMDiff;
	double mTMDiffSum;

	vector<vector<Mat>> mBlockHist;
	vector<vector<double>> mBlockHistDiffVec;

	Frame() {
		mIsOverColor = false;
		mIsOverBlock = false;
		mIsOverFlush = false;

		mColorDiff = 0;
		mBlockHistDiff = 0;
		mTMDiff = 0;
		mTMDiffSum = 0;
	}

	void calcColorDiff(Mat compareImage);
	void calcBlockHistogram();
	void calcBlockHistDiff(vector<vector<Mat>> nowHist, Mat prevFrame);
	void calcTemplateMatchingDiff(Mat prevFrame);
};