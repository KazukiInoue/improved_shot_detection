#include "improved_shot_detection_functions.h"

void testTemplateMatchingDiff() {
	string img[2] = { "OMV200_video00020_00072_2.602600.png", "OMV200_video00020_00077_2.769433.png" };

	Frame src;
	src.mImg = imread("../input_image/" + img[0]);
	Frame ref;
	ref.mImg = imread("../input_image/" + img[1]);

	cv::resize(src.mImg, src.mImg, Size(), 176 / (double)src.mImg.cols, 144 / (double)src.mImg.rows);
	cv::resize(ref.mImg, ref.mImg, Size(), 176 / (double)ref.mImg.cols, 144 / (double)ref.mImg.rows);

	cout << img[0] << " and " << img[1] << endl;

	src.calcColorDiff(ref.mImg);
	cout << "colorDiff = " << src.mColorDiff << endl;

	src.calcTemplateMatchingDiff(ref.mImg);
	cout << "mTMDiff = " << src.mTMDiff << endl;
	cv::imshow("src", src.mImg);
	cv::imshow("ref", ref.mImg);

	cv::waitKey(0);
}