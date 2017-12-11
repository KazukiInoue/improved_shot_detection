#include "improved_shot_detection_functions.h"

using namespace std;
using namespace cv;

void calcHistogram(Mat& histogram, const int binNum, Mat img) {

	assert(img.type() == CV_8UC1);
	assert(histogram.type() == CV_32SC1);

	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {

			int histIndex = (int)img.ptr<uchar>(y)[x] / (256 / binNum);

			if (histIndex == binNum) {
				histIndex = binNum - 1;  // 0 <= histIndex <= binNum-1
			}

			histogram.ptr<int>(0)[histIndex] += 1;
		}
	}
}

void Frame::calcBlockHistogram() {

	const int splitNum[2] = { 10, 10 };
	const int blockNum = splitNum[0] * splitNum[1];
	const int blockPixels[2] = { (int)(mImg.rows / splitNum[0]), (int)(mImg.cols / splitNum[1]) };

	const int binNum = 16;

	vector<vector<Mat>> tmpBlockHist(splitNum[0], vector<Mat>(splitNum[1], Mat::zeros(/*rows=*/1, /*cols=*/binNum, CV_32SC1)));
	mBlockHist = tmpBlockHist;

	// フレームをブロックに分割し、ブロックごとのヒストグラムを作成
	for (int ySplit = 0; ySplit < splitNum[0]; ySplit++) {
		for (int xSplit = 0; xSplit < splitNum[1]; xSplit++) {

			cv::Rect tmpRect(/*x_begin=*/xSplit*blockPixels[1], /*y_begin=*/ySplit*blockPixels[0], /*Δx=*/blockPixels[1], /*Δy=*/blockPixels[0]);
			cv::Mat tmpBlock = cv::Mat(mImg, tmpRect);
			cvtColor(tmpBlock, tmpBlock, CV_BGR2GRAY);

			Mat tmpHist = Mat::zeros(mBlockHist[0][0].rows, mBlockHist[0][0].cols, mBlockHist[0][0].type());

			// 一つのブロックのヒストグラムを計算
			calcHistogram(/*&*/tmpHist, binNum, tmpBlock);

			mBlockHist[ySplit][xSplit] = tmpHist;
		}
	}
}

void Frame::calcBlockHistDiff(vector<vector<Mat>> nowHist, Mat prevFrame) {

	// 前後フレームの各ブロックのヒストグラムの差を計算し、ブロック間の最小値を格納
	const int ySplitNum = nowHist.size();
	const int xSplitNum = nowHist[0].size();
	const int blockNum = ySplitNum * xSplitNum;
	const int blockPixels[2] = { (int)(prevFrame.rows / ySplitNum), (int)(prevFrame.cols / xSplitNum) };
	const int binNum = nowHist[0][0].cols;

	const double searchRange[2] = { 10, 10 };

	vector<vector<double>> tmpHistDiffVec(ySplitNum, vector<double>(xSplitNum, 1000));  // 1000はとりあえず大きな数字というだけの意味
	mBlockHistDiffVec = tmpHistDiffVec;

	cvtColor(prevFrame, prevFrame, CV_BGR2GRAY);

	for (int yCenter = 0; yCenter < ySplitNum; yCenter++) {
		for (int xCenter = 0; xCenter < xSplitNum; xCenter++) {

			int begin[2] = { yCenter*blockPixels[0] - searchRange[0], xCenter*blockPixels[1] - searchRange[1] };
			int end[2] = { (yCenter + 1)*blockPixels[0] + searchRange[0], (xCenter + 1)*blockPixels[1] + searchRange[1] };

			if (yCenter == 0) { begin[0] = 0; }
			if (xCenter == 0) { begin[1] = 0; }
			if (yCenter == ySplitNum - 1) { end[0] = (yCenter + 1)*blockPixels[0]; }
			if (xCenter == xSplitNum - 1) { end[1] = (xCenter + 1)*blockPixels[1]; }

			// 比較する画像の範囲を切り出す
			cv::Rect imgRect(/*x_begin=*/begin[1], /*y_begin=*/begin[0], /*Δx=*/end[1] - begin[1], /*Δy=*/end[0] - begin[0]);
			cv::Mat compareImg = cv::Mat(prevFrame, imgRect);

			for (int y = 0; y < compareImg.rows - blockPixels[0]; y++) {
				for (int x = 0; x < compareImg.cols - blockPixels[1]; x++) {

					cv::Rect areaRect(/*x_begin=*/x, /*y_begin=*/y, /*Δx=*/blockPixels[1], /*Δy=*/blockPixels[0]);
					cv::Mat compareArea = cv::Mat(compareImg, areaRect);

					cv::Mat areaHist = Mat::zeros(/*rows=*/1, /*cols=*/binNum, CV_32SC1);
					calcHistogram(areaHist, binNum, compareArea);

					cv::Mat histDiffMat;

					absdiff(nowHist[yCenter][xCenter], areaHist, histDiffMat);
					double tmpHistDiff = mean(histDiffMat)[0] / binNum;

					if (mBlockHistDiffVec[yCenter][xCenter] > tmpHistDiff) {
						mBlockHistDiffVec[yCenter][xCenter] = tmpHistDiff;
					}
				}
			}
		}
	}

	// 式(2)の計算
	const double thresholdLamda = 0.3;

	for (int y = 0; y < ySplitNum; y++) {
		for (int x = 0; x < xSplitNum; x++) {
			if (mBlockHistDiffVec[y][x] > thresholdLamda) {
				mBlockHistDiff += (double)1 / blockNum;
			}
		}
	}
}