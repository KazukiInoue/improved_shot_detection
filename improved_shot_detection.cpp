#include "improved_shot_detection_functions.h"

void judgeOverThreshold(bool& is, double value, const double threshold) {
	if (value > threshold) {
		is = true;
	}
	else {
		is = false;
	}
}


void improvedShotDetection() {

	const int width = 176;
	const int height = 144;

	const double thresholdSad = 15.0;
	const double thresholdCut = 0.3;
	const double thresholdSum = 1000.0;

	const int flushRange = 3; // いくつ前・先のフレームを保存しておくか
	const int nowIndex = flushRange; // 現在のフレーム番号インデックス

	const int keepSize = 1 + 2 * flushRange;  // 保存しておくフレームの数の総計

	string videoPath = "../input_video/OMV200_video00001.mp4";
	VideoCapture cap(videoPath);
	if (!cap.isOpened()) {
		std::cerr << "動画を開くことができません" << endl;
		exit(1);
	}

	vector<Frame> frameInfo(keepSize);

	for (int frameItr = 0; frameItr < keepSize; frameItr++) {

		cv::Mat nowFrame;
		cap >> nowFrame;
		cv::resize(nowFrame, frameInfo[frameItr].mImg, Size(), width / (double)nowFrame.cols, height / (double)nowFrame.rows);
	}

	for (int entireItr = 0;; entireItr++) {

		if (entireItr > 0) {
			Frame aheadFrame;
			cap >> aheadFrame.mImg;
			if (aheadFrame.mImg.empty()) {
				break;
			}

			cv::resize(aheadFrame.mImg, aheadFrame.mImg, Size(), width / (double)aheadFrame.mImg.cols, height / (double)aheadFrame.mImg.rows);

			frameInfo.erase(frameInfo.begin());
			frameInfo.push_back(aheadFrame);
		}

		cv::imshow("video", frameInfo[nowIndex].mImg);
		cv::waitKey(30);

		//----------カット検出----------
		// 1.画素値の差分による検出

		frameInfo[nowIndex].calcColorDiff(frameInfo[nowIndex - 1].mImg);

		judgeOverThreshold(/*&*/frameInfo[nowIndex].mIsOverColor, frameInfo[nowIndex].mColorDiff, thresholdSad);

		// cout << "coloDiff = " << frameInfo[nowIndex].mColorDiff << endl;

		if (frameInfo[nowIndex].mIsOverColor) {

			//cout << "pass color" << endl;

			// 2.ブロックマッチングによる検出
		/*	for (int frameItr = 0; frameItr < nowIndex + 1; frameItr++) {
				if (frameInfo[frameItr].mBlockHist.size() == 0) {
					frameInfo[frameItr].calcBlockHistogram();
				}
			}*/

			frameInfo[nowIndex - 1].calcTemplateMatchingDiff(frameInfo[nowIndex - 2].mImg);
			frameInfo[nowIndex].calcTemplateMatchingDiff(frameInfo[nowIndex - 1].mImg);

			double passedColorDet = 0;

			// -----diff単体を使ったver-----
			if (frameInfo[nowIndex - 1].mIsOverColor) {
				passedColorDet = frameInfo[nowIndex - 1].mTMDiff;
			}
			judgeOverThreshold(/*&*/frameInfo[nowIndex].mIsOverBlock, frameInfo[nowIndex].mTMDiff - passedColorDet, thresholdCut);

			// ------Sumを使ったver-----
		/*	if (frameInfo[nowIndex - 1].mIsOverColor) {
				passedColorDet = frameInfo[nowIndex - 1].mTMDiffSum;
			}*/

			// judgeOverThreshold(/*&*/frameInfo[nowIndex].mIsOverBlock, frameInfo[nowIndex].mTMDiffSum - passedColorDet, thresholdSum);


			if (frameInfo[nowIndex].mIsOverBlock) {

				//cout << "pass block matching" << endl;
				// cout << "mTMDiffSum = " << frameInfo[nowIndex].mTMDiffSum << endl;

				// 3.フラッシュによる誤検出を防ぐ
				// テンプレートマッチングの値が最小となる数フレーム先の画像を探す

				// test_test_____論文通り、cv::minを使ったver_____test_test
				//Frame valueMin;
				//valueMin.mImg = frameInfo[nowIndex].mImg.clone();
				//for (int futureItr = 1; futureItr < flushRange + 1; futureItr++) {
				//	cv::min(valueMin.mImg, frameInfo[nowIndex + futureItr].mImg, valueMin.mImg);
				//}

				//valueMin.calcTemplateMatchingDiff(frameInfo[nowIndex - 1].mImg);

				//valueMin.calcColorDiff(frameInfo[nowIndex - 1].mImg);
				//judgeOverThreshold(/*&*/valueMin.mIsOverColor, valueMin.mColorDiff, thresholdSad);

				//double ifOverColor[2] = {};

				//if (valueMin.mIsOverColor) {
				//	ifOverColor[0] = valueMin.mBlockHistDiff;
				//}

				//if (frameInfo[nowIndex - 1].mIsOverColor) {
				//	ifOverColor[1] = frameInfo[nowIndex - 1].mBlockHistDiff;
				//}

				//judgeOverThreshold(/*&*/frameInfo[nowIndex].mIsOverFlush, ifOverColor[0] - ifOverColor[1], thresholdCut);



				// test_test_______現在フレームよりも先のフレーム画像のみを使う方法_______test_test

				Frame valueMin;
				valueMin.mImg = frameInfo[nowIndex].mImg.clone();

				valueMin.calcTemplateMatchingDiff(frameInfo[nowIndex - 1].mImg);

				double tmpTMDiff = valueMin.mTMDiff;

				for (int futureItr = 1; futureItr < flushRange + 1; futureItr++) {

					Frame tmpFuture;

					tmpFuture.mImg = frameInfo[nowIndex + futureItr].mImg.clone();

					tmpFuture.calcTemplateMatchingDiff(frameInfo[nowIndex-1].mImg);

					if (tmpFuture.mTMDiff < tmpTMDiff) {
						valueMin.mImg = tmpFuture.mImg.clone();

						valueMin.mTMDiff = tmpFuture.mTMDiff;
						tmpTMDiff = tmpFuture.mTMDiff;
					}
				}

				valueMin.calcColorDiff(frameInfo[nowIndex - 1].mImg);
				judgeOverThreshold(/*&*/valueMin.mIsOverColor, valueMin.mColorDiff, thresholdSad);

				double ifOverColor[2] = {};

				if (valueMin.mIsOverColor) {
					ifOverColor[0] = valueMin.mTMDiff;
				}

				if (frameInfo[nowIndex - 1].mIsOverColor) {
					ifOverColor[1] = frameInfo[nowIndex - 1].mTMDiff;
				}

				// test_test_______現在フレームよりも前のフレーム画像も使う方法_______test_test

			/*	Frame pastMin;
				Frame futureMin;*/

				// -----cv::minを使ったver------
		/*

				pastMin.mImg = frameInfo[nowIndex].mImg.clone();
				futureMin.mImg = frameInfo[nowIndex].mImg.clone();

				for (int timeItr = 1; timeItr < flushRange + 1; timeItr++) {
					cv::min(pastMin.mImg, frameInfo[nowIndex - timeItr].mImg, pastMin.mImg);
					cv::min(futureMin.mImg, frameInfo[nowIndex + timeItr].mImg, futureMin.mImg);
				}

				pastMin.calcTemplateMatchingDiff(futureMin.mImg);*/

				// -------現在フレームよりも先のフレームにおいて、最も差が小さくなるような画像を探索する方法-----

				/*pastMin.mImg = frameInfo[nowIndex].mImg.clone();
				futureMin.mImg = frameInfo[nowIndex + 1].mImg.clone();

				pastMin.calcTemplateMatchingDiff(futureMin.mImg);

				double tmpBlockHistDiff = pastMin.mBlockHistDiff;

				for (int pastItr = 0; pastItr < flushRange + 1; pastItr++) {
					for (int futureItr = 1; futureItr < flushRange + 1; futureItr++) {

						Frame tmpPast;
						Frame tmpFuture;

						tmpPast.mImg = frameInfo[nowIndex - pastItr].mImg.clone();
						tmpFuture.mImg = frameInfo[nowIndex + futureItr].mImg.clone();

						tmpPast.calcTemplateMatchingDiff(tmpFuture.mImg);

						if (tmpPast.mBlockHistDiff < tmpBlockHistDiff) {
							pastMin.mImg = frameInfo[nowIndex - pastItr].mImg.clone();
							futureMin.mImg = frameInfo[nowIndex + futureItr].mImg.clone();

							pastMin.mBlockHistDiff = tmpPast.mBlockHistDiff;
							tmpBlockHistDiff = tmpPast.mBlockHistDiff;
						}

					}

				}*/

				judgeOverThreshold(/*&*/frameInfo[nowIndex].mIsOverFlush, ifOverColor[0] - ifOverColor[1], thresholdCut);


				if (frameInfo[nowIndex].mIsOverFlush) {
					// cout << "flush diff = " << pastMin.mTMDiffSum << endl;
					cout << "pass flush" << endl;
					//system("pause");
					imwrite("../output_not_cvmin_kankoku/frame_" + to_string(entireItr) + ".jpg", frameInfo[nowIndex].mImg);
					/*		cv::imwrite("../output_shot/frame_" + to_string(entireItr) + "_now" + ".jpg", frameInfo[nowIndex].mImg);
							cv::imwrite("../output_shot/frame_" + to_string(entireItr) + "_pastMin" + ".jpg", pastMin.mImg);
							cv::imwrite("../output_shot/frame_" + to_string(entireItr) + "_futureMin" + ".jpg", futureMin.mImg);*/
				}
			}
		}
	}
}