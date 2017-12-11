#include "improved_shot_detection_functions.h"

void exampleTemplateMatching() {

	Mat src = imread("C:/MUSIC_RECOMMENDATION/nhk_shot_detection/input_image/frame1.jpg");
	Mat ref = imread("C:/MUSIC_RECOMMENDATION/nhk_shot_detection/input_image/frame1.jpg");
	if (src.empty()) {
		exit(1);
	}
	if (ref.empty()) {
		exit(1);
	}

	cv::resize(src, src, Size(), 256 / (double)src.cols, 144 / (double)src.rows);
	cv::resize(ref, ref, Size(), 256 / (double)ref.cols, 144 / (double)ref.rows);

	Point begin = Point(0, 0);
	Point blockPixels(src.cols / 10, src.rows / 10);

	cv::Rect rect((int)begin.x, (int)begin.y, (int)blockPixels.x, (int)blockPixels.y);
	cv::Mat srcTemplate = cv::Mat(ref, rect);

	//rectangle(ref, begin, Point(begin.x + srcTemplate.cols, begin.y + srcTemplate.rows), Scalar(0, 0, 255), 2, 8, 0);
	cv::imshow("テンプレート画像", srcTemplate);

	cv::imshow("参照画像", ref);

	Mat result;
	matchTemplate(src, srcTemplate, result, TM_SQDIFF);

	//imshow("マッチング画像", result);

	Point minPt;
	Point maxPt;
	double minValue = 0;
	double maxValue = 0;

	minMaxLoc(result, &minValue, &maxValue, &minPt, &maxPt);

	minValue = minValue / (double)(blockPixels.x*blockPixels.y);
	maxValue = maxValue / (double)(blockPixels.x*blockPixels.y);

	rectangle(src, minPt, Point(minPt.x + srcTemplate.cols, minPt.y + srcTemplate.rows), Scalar(0, 0, 255), 2, 8, 0);
	rectangle(src, maxPt, Point(maxPt.x + srcTemplate.cols, maxPt.y + srcTemplate.rows), Scalar(255, 0, 0), 2, 8, 0);

	imshow("マッチング表示画像", src);

	cout << "minValue:" << minValue << endl;
	cout << "maxValue:" << maxValue << endl;

	cvWaitKey(0);
}