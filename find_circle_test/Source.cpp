#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <cv.h>
#include <iostream>
#include <stdio.h>

using namespace cv;

/** @function main */
int main(int argc, char** argv)
{
	Mat src, src_gray,src2;
	String str="D1";
	/// Read the image DSC06657
	src = imread(str+".jpg", 1 );

	if( !src.data )
	{ return -1; }

	/// Convert it to gray
	cvtColor( src, src_gray, CV_RGB2GRAY );

	//sobel------------------------------------------------
		int scale = 2;
	int delta = 0;
	int ddepth = CV_16S;

	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( grad_x, abs_grad_x );
	Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( grad_y, abs_grad_y );
	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, src_gray );
	//cvThreshold(src,src,128,255,CV_THRESH_BINARY); 
	//-------------------------------------------
	//namedWindow( "Sobel", CV_WINDOW_AUTOSIZE );
	//imshow("Sobel", src_gray );
	imwrite( str+"_sobel.jpg", src_gray );
	src2=src_gray;
  //
	Mat img_threshold;
	threshold(src2, img_threshold, 0, 255, CV_THRESH_OTSU+CV_THRESH_BINARY);
	//namedWindow( "binary", CV_WINDOW_AUTOSIZE );
	//imshow("binary", img_threshold);
	

	/// Reduce the noise so we avoid false circle detection
	GaussianBlur( img_threshold, img_threshold, Size(9,9), 2, 2 );

	//imwrite( str+"_binary.jpg", img_threshold );
	vector<Vec3f> circles;
	/// Apply the Hough Transform to find the circles 
	HoughCircles( img_threshold, circles, CV_HOUGH_GRADIENT, 1,src2.rows/8, 200, 100, 0, 0 );

	cvtColor(  img_threshold,  img_threshold, CV_GRAY2RGB);
	/// Draw the circles detected
	for( size_t i = 0; i < circles.size(); i++ )
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// circle center
		circle( img_threshold , center, 3, Scalar(0,0,255), 10, 8, 0 );
		// circle outline
		circle( img_threshold , center, radius, Scalar(0,0,255), 10, 8, 0 );
	}
   
	/// Show your results
	//namedWindow( "Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE );
	imshow( "Hough Circle Transform Demo", img_threshold  );
  
	imwrite( str+"_fc.jpg", img_threshold );
	waitKey(0);
	return 0;
}