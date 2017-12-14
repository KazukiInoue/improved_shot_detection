#include <iostream>

#include <vector>

#include "improved_shot_detection_functions.h"
#include "accessDirectory.h"

using namespace std;
using namespace cv;


int main() {

	string fromDir[2] = {
		"../../src_data/OMV62of65/OMV62of65/",
		"C:/MUSIC_RECOMMENDATION/src_data/OMV200/" };

	string toDir[2] = {
		"../../src_data/shots_OMV62of65_improved/",
		"C:/MUSIC_RECOMMENDATION/src_data/shots_OMV200_improved/" };

	for (int videoType = 0; videoType < 1; videoType++) {

		vector<string> videoList = Dir::readOutOfFolder(fromDir[videoType]);

		for (int videoItr = 0; videoItr < videoList.size(); videoItr++) {

			string fileName;
			string extension;
			stringstream tmpFileName(videoList[videoItr]);
			string tmpString;
			vector<string> splitedFileName;  // = video00001

			while (getline(tmpFileName, tmpString, '.')) {
				splitedFileName.push_back(tmpString);
			}
			fileName = splitedFileName[0];
			extension = splitedFileName[1];

			if (videoList[videoItr] != "." && videoList[videoItr] != ".." && extension == "mp4") {



				string videoFilePath = fromDir[videoType] + videoList[videoItr];

				cout << videoFilePath << endl;

				improvedShotDetection(videoFilePath, toDir[videoType], fileName);
			}
		}
	}

	// improvedShotDetection(../input_video/OMV200_video00001.mp4);

	////--------test---------

	// testTemplateMatchingDiff();+

	// exampleTemplateMatching();

	system("pause");
	return 0;
}