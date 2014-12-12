#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <cv.h>
#include <iostream>
#include <stdio.h>
#include <sstream>
using namespace cv;
using namespace std;
void find_circle(int n);
void draw_rec(int n);
double pi =3.141592653589793;
/** @function main */
int main(int argc, char** argv)
{
	
	for(int i=746;i<=768;i++){
		draw_rec(i);
	}
	waitKey(0);
	return 0;
}
void find_circle(int n){
	stringstream ss;
	ss<<n;
	String str="DSC06"+ss.str();
	cout<<"TEST"<<str<<endl;
	Mat src, src_gray,src2;
	//String str="DSC06537";
	/// Read the image DSC06657
	src = imread("LENS/"+str+".jpg", 1 );

	if( !src.data )
	{return; }

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
	//imwrite( str+"_sobel.jpg", src_gray );
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
	HoughCircles( img_threshold, circles, CV_HOUGH_GRADIENT,1,img_threshold.rows/4, 200, 100, 900, 1100 );

	cvtColor(  img_threshold,  img_threshold, CV_GRAY2RGB);
	/// Draw the circles detected
	for( size_t i = 0; i < 1; i++ )
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// circle center
		circle( src , center, 3, Scalar(0,0,255), 10, 8, 0 );
		// circle outline
		circle( src , center, radius, Scalar(0,0,255), 10, 8, 0 );
	}
	/*
	cout<<"rad:"<<cvRound(circles[0][2])<<endl;
	cout<<"x:"<<cvRound(circles[0][0])<<endl;
	cout<<"y:"<<cvRound(circles[0][1])<<endl;*/
	Mat result(cvRound(circles[0][2])*2,cvRound(circles[0][2])*2 , CV_8UC3, Scalar(0,0,0));
	
	for(int x=cvRound(circles[0][0])-cvRound(circles[0][2]),x1=0;x<cvRound(circles[0][0])+cvRound(circles[0][2]);x++,x1++)
	{
		for(int y=cvRound(circles[0][1])-cvRound(circles[0][2]),y1=0;y<cvRound(circles[0][1])+cvRound(circles[0][2]);y++,y1++)
		{
			
			Vec3b color=src.at<Vec3b>(Point(x,y));
			result.at<Vec3b>(Point(x1,y1)) = color;

		}
	}
	
	/// Show your results
	//namedWindow( "Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE );
	//imshow( "Hough Circle Transform Demo", result  );
  
	imwrite( "LENS/after/"+str+"_fc.jpg", result );
}
void draw_rec(int n){
	stringstream ss;
	ss<<n;
	String str="DSC06"+ss.str();
	cout<<"TEST"<<str<<endl;
	Mat src, src_gray,src2;
	/// Read the image DSC06657
	src = imread("LENS/"+str+".jpg", 1 );

	if( !src.data )
	{return; }

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
	//imwrite( str+"_sobel.jpg", src_gray );
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
	HoughCircles( img_threshold, circles, CV_HOUGH_GRADIENT,1,img_threshold.rows/4, 200, 100, 900, 1100 );
	cvtColor(  img_threshold,  img_threshold, CV_GRAY2RGB);
	/// Draw the circles detected
	for( size_t i = 0; i < 1; i++ )
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// circle center
		circle( src , center, 3, Scalar(0,0,255), 10, 8, 0 );
		// circle outline
		circle( src , center, radius, Scalar(0,0,255), 10, 8, 0 );
	}

	/*
	cout<<"rad:"<<cvRound(circles[0][2])<<endl;
	cout<<"x:"<<cvRound(circles[0][0])<<endl;
	cout<<"y:"<<cvRound(circles[0][1])<<endl;
	*/
	int size=1800;
	Mat result(size,size , CV_8UC3, Scalar(0,0,0));
	double h=cvRound(circles[0][0]);//A(h,k)¶ê¤ß
	double k=cvRound(circles[0][1]);
	double r=cvRound(circles[0][2]);
	double theata=0;//0.2
	double x=0,y=0;
	double dd=360/(double)size;
	for(int i=0;i<size;i++){
		double deg=(i*dd*pi)/180;
		x=h+r*cos(deg);
		y=k+r*sin(deg);
		/*
		cout<<"h:"<<h<<endl;
		cout<<"k:"<<k<<endl;
		cout<<"x:"<<x<<endl;
		cout<<"y:"<<y<<endl;*/
	
		/*
		cout<<"a:"<<a<<endl;
		cout<<"----------------"<<endl;
		*/
		//system("pause"); 
		double a=(y-k)/(x-h);//y=ax; x=y/a;
		/*
		if(abs(a)==1){
			double ds=(x-h)/size;
			double tx=0;
			double ty=0;
			for(int j=0;j<size;j++){
				ty=k+j*ds;
				tx=h;
				Vec3b color=src.at<Vec3b>(Point(tx,ty));
				result.at<Vec3b>(Point(i,j)) = color;
			}
		}
		else if(abs(a)==0){
			double ds=(y-k)/size;
			double tx=0;
			double ty=0;
			for(int j=0;j<size;j++){
				ty=k;
				tx=h+j*ds;
				Vec3b color=src.at<Vec3b>(Point(tx,ty));
				result.at<Vec3b>(Point(i,j)) = color;
			}
		}*/
		if(abs(a)>=1){//¬Ýy
			double ds=(y-k)/size;
			double tx=0;
			double ty=0;
			for(int j=0;j<size;j++){
				ty=k+j*ds;
				tx=h+j*ds/a;
				Vec3b color=src.at<Vec3b>(Point(tx,ty));
				result.at<Vec3b>(Point(i,j)) = color;
			}
		}
		else{//¬ÝX
			double ds=(x-h)/size;
			double tx=0;
			double ty=0;
			for(int j=0;j<size;j++){
				tx=h+j*ds;
				ty=k+j*ds*a;
				//
				/*
				if(x<3483.52){
					cout<<"H:"<<h<<endl;
					cout<<"K:"<<k<<endl;
					cout<<"j:"<<j<<endl;
					cout<<"ds:"<<ds<<endl;
					cout<<"tx:"<<tx<<endl;
					cout<<"ty:"<<ty<<endl;
					cout<<"a:"<<a<<endl;
				}*/
				Vec3b color=src.at<Vec3b>(Point(tx,ty));
				result.at<Vec3b>(Point(i,j)) = color;
			}
		}


	}
	//system("pause"); 
	/// Show your results
	//namedWindow( "Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE );
	//imshow( "Hough Circle Transform Demo", result  );
  
	imwrite( "LENS/after/"+str+"_c2s.jpg", result );
}