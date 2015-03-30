#include "stdafx.h"
#include <opencv2\opencv.hpp>
#include <opencv2\ml\ml.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>

using namespace cv;
#define MAX_NUM_IMAGES    60000
class DigitRecognizer
{
public:
    DigitRecognizer();

    ~DigitRecognizer();

    bool train(char* trainPath, char* labelsPath);

    int classify(Mat img);

	void data_training();

private:
	
	Mat preprocessImage(Mat img);
    int readFlippedInteger(FILE *fp);


private:
    KNearest    *knn;
    int numImages,numRows, numCols;

};