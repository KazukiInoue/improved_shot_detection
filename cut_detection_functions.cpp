#include "improved_shot_detection_functions.h"

using namespace std;
using namespace cv;

// ��f�l�ŃV���b�g���o
void Frame::calcColorDiff(Mat compareImage) {

	Mat targetCh[3];
	Mat compareCh[3];

	split(mImg, targetCh);
	split(compareImage, compareCh);

	Mat colorDiffMat;

	for (int c = 0; c < 3; c++) {
		absdiff(targetCh[c], compareCh[c], colorDiffMat);
		mColorDiff += (double)mean(colorDiffMat)[0];
	}
}

void Frame::calcTemplateMatchingDiff(Mat prevFrame) {

	// ���t���[���̃u���b�N��؂�o���O�t���[���ƃe���v���[�g�}�b�`���O���s��
	const int ySplitNum = 10;
	const int xSplitNum = 10;
	const int blockNum = ySplitNum * xSplitNum;
	const int blockPixels[2] = { (int)(mImg.rows / ySplitNum), (int)(mImg.cols / xSplitNum) };

	const double thresholdLamda = 5000; // ���̒l����

	for (int y = 0; y < ySplitNum; y++) {
		for (int x = 0; x < xSplitNum; x++) {

			cv::Rect srcRect(/*x_begin=*/x*blockPixels[1], /*y_begin=*/y*blockPixels[0], /*��x=*/blockPixels[1], /*��y=*/blockPixels[0]);
			cv::Mat srcBlock(mImg, srcRect);

			// �e���v���[�g�}�b�`���O����摜�͈̔͂��i��
			int begin[2] = { (y - 1)*blockPixels[0], (x - 1)*blockPixels[1] };
			int end[2] = { (y + 2)*blockPixels[0], (x + 2)*blockPixels[1] };

			if (y == 0) { begin[0] = 0; }
			if (x == 0) { begin[1] = 0; }
			if (y == ySplitNum - 1) { end[0] = (y + 1)*blockPixels[0]; }
			if (x == xSplitNum - 1) { end[1] = (x + 1)*blockPixels[1]; }

			cv::Rect compareRect(/*x_begin=*/begin[1], /*y_begin=*/begin[0], /*��x=*/end[1] - begin[1], /*��y=*/end[0] - begin[0]);
			cv::Mat compareBlock(prevFrame, compareRect);

			Mat result;
			double minValue = 0;
			matchTemplate(compareBlock, srcBlock, result, TM_SQDIFF);
			minMaxLoc(result, &minValue, 0, 0, 0);

			double tmpTMDiff = minValue / (blockPixels[0] * blockPixels[1]);
			mTMDiffSum += tmpTMDiff;

			// cout << "tmpTMDiff = " << tmpTMDiff << endl;

			// ��(2)�̌v�Z
			if (tmpTMDiff > thresholdLamda) {
				mTMDiff += (double)1 / blockNum;
			}
		}
	}

	mTMDiffSum /= blockNum;
}