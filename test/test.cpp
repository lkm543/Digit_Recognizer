#include "digitrecognizer.h"
#include "stdafx.h"
#include "windows.h"
#include <opencv2\opencv.hpp>
#include <opencv2\ml\ml.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

Mat image,image_gray,image_threshold,image_processed;
Mat image_process(Mat image_origin);
Mat preprocessImage(Mat image_input);
void save_image(Mat image_input,string filename_input,bool right_or_false);
void data_training();
int number_recognition(Mat image_input);
bool c;
DigitRecognizer *dr;
int number,correct,type,number_train;
KNearest knn;
int numRows,numCols;
bool right_or_false;
fstream fp;
int a=1,b=1; //the filename is set to be "a_b.PNG"
int _tmain(int argc, _TCHAR* argv[])
{	
	char filename[10];

	int correct=0,flag=0; //if previous picutre is empty=> flag=1
	int total_number=0;

	char filename_errorlog[]="error_log.txt";

    fp.open(filename_errorlog, ios::out);//¶}±ÒÀÉ®×

	cout << "**********************************" << endl ;
	cout << "*Type 1 to use default database. *" << endl ;
	cout << "*Type 2 to use Our Own database. *" << endl ;
	cout << "*Type any button to exit program.*" << endl ;
	cout << "**********************************" << endl ;
	cin>>type;
	if(type==2){
	cout << "*Type the number to train.*" << endl ;
	cin>>number_train;
	}cout << "Start process!" << endl ;

	dr = new DigitRecognizer();
	
	if(type==1){
	dr->train("C:\\Users\\Mick\\Desktop\\train-images.idx3-ubyte", "C:\\Users\\Mick\\Desktop\\train-labels.idx1-ubyte");
	}
	else if(type==2){
	dr->data_training();
	}
	else{
		exit(0);
	}
	a=1;
	b=number_train+1;
	while(1){
	sprintf_s(filename,"%d_%d",a,b); // replace to get the filename now
	string filename2(filename); //convert char to string
    cout<<"Start to Read and Save File "<<filename2<<".PNG"<<endl;
	image = imread("C:\\Users\\mick\\Desktop\\data\\TrainData_Origin\\"+filename2+".PNG");

	//Load Data Succeed!
	if(image.data && b>number_train) {
		total_number++;
		b++;
		flag=0;
		image_processed=image_process(image);
		number=number_recognition(image_processed);
		//cout<<number<<endl;;
		if(number==a) {
			++correct;
			right_or_false=true;
		}
		else {
			right_or_false=false;
			cout<<"******Wrong****** Right number is :"<<a<<endl;
		}
		save_image(image_processed,filename2,right_or_false);
	}
	//No data be read!
	else{
		cout<<"File "<<filename2<<" Not exist!"<<endl;
		//Previous data not read , either. Exit program.
		if (flag==1) {
			cout<<"End of image processing!"<<endl;
			cout<<"Correct Number:  "<<correct<<endl;
			cout<<"Total   Number:  "<<total_number<<endl;
			cout<<"Correct Ratio:   "<<(float)correct/total_number*100<<"%"<<endl;
			break;
		}
		//Previous data read , try next one.
		else {
			a++;
			b=number_train+1;
			flag=1;
		}
	}
	}
	fp.close();//Ãö³¬ÀÉ®×
	system("pause"); 
	return 0;
}

Mat image_process(Mat image_origin){
	cvtColor(image_origin,image_gray,CV_BGR2GRAY);
	equalizeHist( image_gray, image_threshold );
	//adaptiveThreshold(image_threshold, image_threshold, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV,5, 2);
	//cv::threshold(image_threshold, image_threshold, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	//Canny(image_threshold, image_threshold, 1, 60, 7);
	blur( image_threshold, image_threshold, Size( 2,2 ), Point(-1,-1));
	blur( image_threshold, image_threshold, Size( 2,2 ), Point(-1,-1));
	cv::threshold(image_threshold, image_threshold, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

	//GaussianBlur( image_threshold, image_threshold, Size(2,2 ),0,0,4);

	//threshold(image_threshold,image_threshold, 155, 255,0);

	//erode( image_threshold, image_threshold,Mat());
	//blur( image_threshold, image_threshold, Size( 2,2 ), Point(-1,-1));
	//blur( image_threshold, image_threshold, Size( 2,2 ), Point(-1,-1));
	//dilate( image_threshold, image_threshold,Mat());
	//erode( image_threshold, image_threshold,Mat());
	/*int numberofcontour=0;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	blur( image_threshold, image_threshold, Size( 2,2 ), Point(-1,-1));*/
	/// Detect edges using canny
	//Canny(image_threshold, image_threshold, 2, 40, 3);
	/// Find contours
	//findContours( image_threshold, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
	//numberofcontour=contours.size();
	/*while (numberofcontour>10) {
		erode( image_threshold, image_threshold,Mat());
		findContours( image_threshold, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
		numberofcontour=contours.size();
	}*/
	/*
	cv::Mat skel(image_threshold.size(), CV_8UC1, cv::Scalar(0));
	cv::Mat temp;
	cv::Mat eroded;
 
	cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

	bool done;		
	do
	{
	  cv::erode(image_threshold, eroded, element);
	  cv::dilate(eroded, temp, element); // temp = open(img)
	  cv::subtract(image_threshold, temp, temp);
	  cv::bitwise_or(skel, temp, skel);
	  eroded.copyTo(image_threshold);
 
	  done = (cv::countNonZero(image_threshold) == 0);
	} while (!done);
	*/
	//Canny(image_threshold, image_threshold, 1, 60, 7);
	//blur( image_threshold, image_threshold, Size( 2,2 ), Point(-1,-1));	


	
	//erode( image_threshold, image_threshold,Mat());
	//blur( image_threshold, image_threshold, Size( 2,2 ), Point(-1,-1));
	//Canny(image_threshold, image_threshold, 1, 60, 7);
	//Canny(image_threshold, image_threshold, 1, 60, 7);
	/*dilate( image_threshold, image_threshold,Mat());
	erode( image_threshold, image_threshold,Mat());
	*/
	return image_threshold;
}

void DigitRecognizer:: data_training(){
	/*Mat original;//original is used only in data_training
	cvtColor(image,original,CV_BGR2GRAY);
	equalizeHist( original, original );
	cv::Canny(original, original, 0, 100.0);
	vector<cv::Point> points;
	cv::Mat dilateKernel(cv::Size(3,3), CV_8UC1, cv::Scalar(1));
	cv::dilate(original, original, dilateKernel);
     
	vector<vector<cv::Point> > foundc;
	cv::findContours(original, foundc,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, cv::Point(0,0));
	if(foundc.size() >0) {
	    points = foundc[0];
	} */
	Mat train_data,train_labels;
	int a=1,b=1; //the filename is set to be "a_b.PNG"
	int flag=0; //if previous picutre is empty=> flag=1
	char filename[10];
	Mat img,image_gray_2;


	while(1){
	sprintf_s(filename,"%d_%d",a,b); // replace to get the filename now
	string filename2(filename); //convert char to string
    cout<<"Start to Train File "<<filename2<<".PNG"<<endl;
	image = imread("C:\\Users\\mick\\Desktop\\data\\TrainData_Origin\\"+filename2+".PNG");

	//Load Data Succeed!
	if(image.data&&b<=number_train) {
		image_gray_2=image_process(image);
			
		//cvtColor(image,image_gray_2,CV_BGR2GRAY);
		b++;
		flag=0;
		//resize is for test
		//resize(image_gray_2,image_gray_2, Size(30, 80));
		//img=image.clone();
		Mat float_data;
		image_gray_2.convertTo(float_data, CV_32FC1);             // to float
		train_data.push_back( float_data.reshape(1,1) ); // add 1 row (flattened image)
		train_labels.push_back(a);       // add 1 item
	}
	//No data be read!
	else{
		cout<<"File "<<filename2<<" Not exist!"<<endl;
		//Previous data not read , either. Exit program.
		if (flag==1) {
			cout<<"End of image trainning!"<<endl;
			break;
		}
		//Previous data read , try next one.
		else {
			a++;
			b=1;
			flag=1;
		}
	}
	}
	if(knn->train(train_data, train_labels)) cout<<"Train succeed!!"<<endl;
	else {cout<<"Train failed"<<endl;}
}

void save_image(Mat image_input,string filename_input,bool right_or_false){
		string filename_saved;
		//resize(image_input, image_input, Size(30, 80));
	if (right_or_false==true) {
		filename_saved=filename_input+".PNG";
	}
	else if (right_or_false==false) {
		filename_saved=filename_input+"_x"+std::to_string((long double) number)+".PNG";
	}
	imwrite( filename_saved,image_input);
}

int number_recognition(Mat image_input){
	number=0;
	number = dr->classify(image_input);
	/*
	if(type==1)	number = dr->classify(image_input);
	else if(type==2){
	image_input=dr->preprocessImage(image_input);
	image_input = image_input.reshape(1, 1);
	number=knn.find_nearest(Mat_<float>(image_input), 1);
	}
	*/
	return number;
}

DigitRecognizer::DigitRecognizer()
{
	knn = new KNearest();
}

DigitRecognizer::~DigitRecognizer()
{
    delete knn;
}

Mat DigitRecognizer::preprocessImage(Mat img)
{	
    int rowTop=-1, rowBottom=-1, colLeft=-1, colRight=-1;

    Mat temp;
    int thresholdBottom = 1;
    int thresholdTop = 1;
    int thresholdLeft = 1;
    int thresholdRight = 1;
    int center = img.rows/2;
    for(int i=center;i<img.rows;i++)
    {
        if(rowBottom==-1)
        {
            temp = img.row(i);
            IplImage stub = temp;
            if(cvSum(&stub).val[0] < thresholdBottom || i==img.rows-1)
                rowBottom = i;

        }

        if(rowTop==-1)
        {
            temp = img.row(img.rows-i);
            IplImage stub = temp;
            if(cvSum(&stub).val[0] < thresholdTop || i==img.rows-1)
                rowTop = img.rows-i;

        }

        if(colRight==-1)
        {
            temp = img.col(i);
            IplImage stub = temp;
            if(cvSum(&stub).val[0] < thresholdRight|| i==img.cols-1)
                colRight = i;

        }

        if(colLeft==-1)
        {
            temp = img.col(img.cols-i);
            IplImage stub = temp;
            if(cvSum(&stub).val[0] < thresholdLeft|| i==img.cols-1)
                colLeft = img.cols-i;
        }
    }
    Mat newImg;

    newImg = newImg.zeros(img.rows, img.cols, CV_8UC1);

    int startAtX = (newImg.cols/2)-(colRight-colLeft)/2;

    int startAtY = (newImg.rows/2)-(rowBottom-rowTop)/2;

    for(int y=startAtY;y<(newImg.rows/2)+(rowBottom-rowTop)/2;y++)
    {
        uchar *ptr = newImg.ptr<uchar>(y);
        for(int x=startAtX;x<(newImg.cols/2)+(colRight-colLeft)/2;x++)
        {
            ptr[x] = img.at<uchar>(rowTop+(y-startAtY),colLeft+(x-startAtX));
        }
    }

    Mat cloneImg = Mat(numRows, numCols, CV_8UC1);

    resize(newImg, cloneImg, Size(numCols, numRows));

    // Now fill along the borders
    for(int i=0;i<cloneImg.rows;i++)
    {
        floodFill(cloneImg, cvPoint(0, i), cvScalar(0,0,0));

        floodFill(cloneImg, cvPoint(cloneImg.cols-1, i), cvScalar(0,0,0));

        floodFill(cloneImg, cvPoint(i, 0), cvScalar(0));
        floodFill(cloneImg, cvPoint(i, cloneImg.rows-1), cvScalar(0));
    }
    cloneImg = cloneImg.reshape(1, 1);

    return cloneImg;
}

int DigitRecognizer::readFlippedInteger(FILE *fp)
{
    int ret = 0;

    BYTE *temp;

    temp = (BYTE*)(&ret);
    fread(&temp[3], sizeof(BYTE), 1, fp);
    fread(&temp[2], sizeof(BYTE), 1, fp);
    fread(&temp[1], sizeof(BYTE), 1, fp);

    fread(&temp[0], sizeof(BYTE), 1, fp);
	cout<<"ret: "<<ret<<endl;
    return ret;

}

bool DigitRecognizer::train(char *trainPath, char *labelsPath)
{
    FILE *fp = fopen(trainPath, "rb");
    FILE *fp2 = fopen(labelsPath, "rb");

    if(!fp || !fp2)  return false;

    // Read bytes in flipped order
    int magicNumber = readFlippedInteger(fp);
    numImages = readFlippedInteger(fp);
    numRows = readFlippedInteger(fp);
    numCols = readFlippedInteger(fp);
    fseek(fp2, 0x08, SEEK_SET);

    if(numImages > MAX_NUM_IMAGES) numImages = MAX_NUM_IMAGES;
    //////////////////////////////////////////////////////////////////
    // Go through each training data entry and save a
    // label for each digit

    int size = numRows*numCols;
    CvMat *trainingVectors = cvCreateMat(numImages, size, CV_32FC1);
    CvMat *trainingClasses = cvCreateMat(numImages, 1, CV_32FC1);
    memset(trainingClasses->data.ptr, 0, sizeof(float)*numImages);
    BYTE *temp = new BYTE[size];
    BYTE tempClass=0;

    for(int i=0;i<numImages;i++)
    {
        fread((void*)temp, size, 1, fp);
        fread((void*)(&tempClass), sizeof(BYTE), 1, fp2);
        trainingClasses->data.fl[i] = tempClass;
        for(int k=0;k<size;k++)
            trainingVectors->data.fl[i*size+k] = temp[k]; ///sumofsquares;
    }

    knn->train(trainingVectors, trainingClasses);
    fclose(fp);
    fclose(fp2);
    return true;
}

int recheck_5(Mat image_input){
	cv::Rect roi_rect = cv::Rect(0,image_input.rows/2-2,image_input.cols/2,image_input.rows/2);
	image_input = image_input(roi_rect);

	vector<vector<Point>> contours;
    findContours(image_input,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);

	if(contours.size()==1) {
		cout<<"Modify by double check function: 5->6"<<endl;
		return 6;    
	}
	else {
		return 5;
	}
}

int recheck_8(Mat image_input){
	cv::Rect roi_rect = cv::Rect(image_input.cols/2,0,image_input.cols/2,image_input.rows/2);
	image_input = image_input(roi_rect);

	vector<vector<Point>> contours;
    findContours(image_input,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);

	if(contours.size()!=1) {
		cout<<"Modify by double check function: 8->6"<<endl;
		return 6;    
	}
	else {
		return 8;
	}
}

int DigitRecognizer::classify(cv::Mat img)
{	
	Mat results=cvCreateMat( 135, 1, CV_32FC1);
	Mat neighborResponses=cvCreateMat( 135, 3, CV_32FC1);
	Mat dists=cvCreateMat( 135, 3, CV_32FC1);
	int first=0;
	Mat img_float,image_flatten;
	if (type==1) resize(img, img, Size(28, 28));
    img.convertTo(img_float, CV_32FC1);
	image_flatten=img_float.reshape(1,1);
	number=knn->find_nearest(Mat_<float>(image_flatten),3,results,neighborResponses,dists);
	//number = knn->find_nearest(Mat_<float>(image_flatten),5,results,neighborResponses,dists);
	//if(number==5) number=recheck_5(img);
	//if(number==8) number=recheck_8(img);
	first=neighborResponses.at<int>(0);
	cout<<results<<endl;
	cout<<neighborResponses<<endl;
	//if (number!=first) number=10;
	if (a!=number){
	fp<<"Right Number is "<<a<<endl;//¼g¤J¦r¦ê
	fp<<"Mistinguish  as "<<number<<endl;//¼g¤J¦r¦ê
	fp<<"Array is"<<neighborResponses<<endl;//¼g¤J¦r¦ê
	}
	//cout<<dists<<endl;
	return number;

	/*default*/
	/*
	Mat cloneImg2 = preprocessImage(img);
	return knn->find_nearest(Mat_<float>(cloneImg2), 1);
	*/
}