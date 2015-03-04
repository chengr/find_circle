#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <cv.h>
#include <iostream>
#include <stdio.h>
#include <sstream>
using namespace cv;
using namespace std;
Mat find_circle(Mat src_img);
Mat find_circle_and_draw_rec(Mat src_img,int size);
Mat adjust(Mat src_img);
void mouse(int event, int x, int y, int flags, void* param);
void draw_rec(int n);
Mat sub_draw(Mat i1,Mat i2);
Mat diff1(Mat i1,Mat i2);
Mat diff2(Mat i1,Mat i2);
Mat diff3(Mat i1,Mat i2);
Mat diff4(Mat i1,Mat i2);
Mat diff5(Mat i1,Mat i2);
double pi =3.141592653589793;
bool fg=true;
int adj_x= 0;
Mat adj_img;
String str[]={"std","中央有高對比白點","外圈有高對比黑點","外圍有區域黑","外圍有高對比黑點或白點","有黑線","中央有白點","大範圍的雜色點","不規則雜點加線"};//9
int cn=8;
/** @function main */
int main(int argc, char** argv)
{
	/*
	Mat src = imread("LENS/"+str[cn]+".jpg", 1 );
	src=find_circle_and_draw_rec(src,720);
	adj_img=src;
	namedWindow("adjust", CV_WINDOW_AUTOSIZE );
	imshow("adjust",src);
	cvSetMouseCallback("adjust", mouse,NULL);
	imwrite( "LENS/after/TEST_"+str[cn]+"_test.jpg", src );
	*/
	/*
	for(int i=0;i<9;i++){
		Mat src = imread("LENS/"+str[i]+".jpg", 1 );
		//src=find_circle(src);
		src=find_circle(src);
		imwrite( "LENS/after/sqr_"+str[i]+".jpg", src );
	}
	*/
	/*
	for(int i=0;i<9;i++){
		Mat org = imread("LENS/after/Adj_std.jpg", 1 );
		Mat src2 = imread("LENS/after/Adj_"+str[i]+".jpg", 1 );
		imwrite( "LENS/after/Diff_(外圍有高對比黑點或白點)"+str[i]+".jpg", diff4(org,src2));
	}*/
	
	for(int i=0;i<9;i++){
		Mat src = imread("LENS/after/Adj_"+str[i]+".jpg", 1 );
		src.convertTo(src, -1, 2.5, -100);// Enter the alpha value [1.0-3.0]: 2.2 Enter the beta value [0-100]: 50
		//GaussianBlur( src ,src , Size(9,9), 0, 0 );
		//blur(src ,src , Size(7,7));
		Mat element2(7,7,CV_8U,Scalar(1));  
		dilate(src,src,element2);  

		Mat element(11,11,CV_8U,Scalar(1));  
		erode(src,src,element);  
		
		src=diff4(src,src);
		imwrite( "LENS/after/ErDX_"+str[i]+".jpg", src);
	}
	
	/*
	for(int i=0;i<9;i++){
		Mat src = imread("LENS/after/Adj_"+str[i]+".jpg", 1 );
		src.convertTo(src, -1, 2.0, -150);// Enter the alpha value [1.0-3.0]: 2.2 Enter the beta value [0-100]: 50
		//GaussianBlur( src ,src , Size(3,3), 0, 0 );
		Mat element(5,5,CV_8U,Scalar(1));  
		erode(src,src,element); 
		Mat element2(11,11,CV_8U,Scalar(1));  
		dilate(src,src,element2);  
		src=diff5(src,src);
		imwrite( "LENS/after/EandDX_"+str[i]+".jpg", src);
	}
	*/
	waitKey(0);
	return 0;
}
Mat find_circle(Mat src_img){
	Mat src_gray;
	//if exist
	if( !src_img.data )
	{return src_img; }
	/// Convert it to gray
	cvtColor( src_img, src_gray, CV_RGB2GRAY );
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
	//namedWindow( "sobel", CV_WINDOW_AUTOSIZE );
	//imshow("sobel",src_gray);
	/// Reduce the noise so we avoid false circle detection
	GaussianBlur( src_gray,src_gray, Size(9,9), 2, 2 );
	//namedWindow( "blur", CV_WINDOW_AUTOSIZE );
	//imshow("blur",src_gray);
	//find circle
	vector<Vec3f> circles;
	/// Apply the Hough Transform to find the circles 
	HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT,1,src_gray.rows/4, 200, 100, 900, 1100);
	/// Draw the circles detected
	for( size_t i = 0; i < 1; i++ )
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// circle center
		circle( src_gray , center, 3, Scalar(0,0,255), 10, 8, 0 );
		// circle outline
		circle( src_gray , center, radius, Scalar(0,0,255), 10, 8, 0 );
	}
	Mat result(cvRound(circles[0][2])*2,cvRound(circles[0][2])*2 , CV_8UC3, Scalar(0,0,0));
	for(int x=cvRound(circles[0][0])-cvRound(circles[0][2]),x1=0;x<cvRound(circles[0][0])+cvRound(circles[0][2]);x++,x1++)
	{
		for(int y=cvRound(circles[0][1])-cvRound(circles[0][2]),y1=0;y<cvRound(circles[0][1])+cvRound(circles[0][2]);y++,y1++)
		{
			
			Vec3b color=src_img.at<Vec3b>(Point(x,y));
			result.at<Vec3b>(Point(x1,y1)) = color;
		}
	}

	
	return result;
}
Mat find_circle_and_draw_rec(Mat src_img,int size){
	Mat src_gray;
	//if exist
	if( !src_img.data )
	{return src_img; }
	/// Convert it to gray
	cvtColor( src_img, src_gray, CV_RGB2GRAY );
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
	//namedWindow( "sobel", CV_WINDOW_AUTOSIZE );
	//imshow("sobel",src_gray);
	/// Reduce the noise so we avoid false circle detection
	GaussianBlur( src_gray,src_gray, Size(9,9), 2, 2 );
	//namedWindow( "blur", CV_WINDOW_AUTOSIZE );
	//imshow("blur",src_gray);
	//find circle
	vector<Vec3f> circles;
	/// Apply the Hough Transform to find the circles 
	HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT,1,src_gray.rows/4, 200, 100, 900, 1100 );
	/// Draw the circles detected
	for( size_t i = 0; i < 1; i++ )
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// circle center
		circle( src_gray , center, 3, Scalar(0,0,255), 10, 8, 0 );
		// circle outline
		circle( src_gray , center, radius, Scalar(0,0,255), 10, 8, 0 );
	}
	//new a square
	Mat result(size,size , CV_8UC3, Scalar(0,0,0));
	double h=cvRound(circles[0][0]);//A(h,k)圓心
	double k=cvRound(circles[0][1]);
	double r=cvRound(circles[0][2]);
	double x=0,y=0;
	double dd=360/(double)size;
	for(int i=0;i<size;i++){
		double deg=(i*dd*pi)/180;
		x=h+r*cos(deg);
		y=k+r*sin(deg);
		double a=(y-k)/(x-h);//y=ax; x=y/a;
		if(abs(a)>=1){//看y
			double ds=(y-k)/size;
			double tx=0;
			double ty=0;
			for(int j=0;j<size;j++){
				ty=k+j*ds;
				tx=h+j*ds/a;
				Vec3b color=src_img.at<Vec3b>(Point(tx,ty));
				result.at<Vec3b>(Point(i,j)) = color;
			}
		}
		else{//看X
			double ds=(x-h)/size;
			double tx=0;
			double ty=0;
			for(int j=0;j<size;j++){
				tx=h+j*ds;
				ty=k+j*ds*a;
				Vec3b color=src_img.at<Vec3b>(Point(tx,ty));
				result.at<Vec3b>(Point(i,j)) = color;
			}
		}


	}
	//GaussianBlur( result,result, Size(9,9), 2, 2 );
	return result;
}
Mat adjust(Mat src_img){
	//cout<<"TEST"<<str[0]<<endl;
	//if exist
	if( !src_img.data )
	{return src_img; }
	Mat result(src_img.cols,src_img.rows, CV_8UC3, Scalar(0,0,0));
	for(int x=0;x<src_img.cols;x++){
		for(int y=0;y<src_img.rows;y++){
			Vec3b color=src_img.at<Vec3b>(Point((x+adj_x)%src_img.cols,y));
			result.at<Vec3b>(Point(x,y)) = color;
		}
	}
	//GaussianBlur( result,result, Size(11,11), 0, 0);
	imwrite( "LENS/after/Adj_"+str[cn]+".jpg", result );
	return result;
}
void mouse(int event, int x, int y, int flags, void* param)
{
	if(fg){
		//cout<<"TEST"<<str[0]<<endl;
		if(event==CV_EVENT_LBUTTONDOWN||event==CV_EVENT_RBUTTONDOWN){
			printf("now x: %d y: %d\n", x,y);
			adj_x=x;
			Mat result=adjust(adj_img);

			//cn++;
			fg=false;
		}
	}
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
	int size=3600;
	Mat result(size,size , CV_8UC3, Scalar(0,0,0));
	double h=cvRound(circles[0][0]);//A(h,k)圓心
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

		if(abs(a)>=1){//看y
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
		else{//看X
			double ds=(x-h)/size;
			double tx=0;
			double ty=0;
			for(int j=0;j<size;j++){
				tx=h+j*ds;
				ty=k+j*ds*a;
				Vec3b color=src.at<Vec3b>(Point(tx,ty));
				result.at<Vec3b>(Point(i,j)) = color;
			}
		}


	}
	imwrite( "LENS/after/"+str+".jpg", result );
}
Mat sub_draw(Mat i1,Mat i2){
	Mat result=i1;
	Vec3b color;
	for(int x=0;x<i1.cols;x++){
		for(int y=0;y<i1.rows;y++){
			Vec3b color1=i1.at<Vec3b>(Point(x,y));
			Vec3b color2=i2.at<Vec3b>(Point(x,y));
			color[0]=abs(color1[0]-color2[0]);
			color[1]=abs(color1[1]-color2[1]);
			color[2]=0;
			/*
			color[0]=abs(color1[0]-color2[0]);
			color[1]=abs(color1[1]-color2[1]);
			color[2]=abs(color1[2]-color2[2]);
			*/
			result.at<Vec3b>(Point(x,y)) = color;
		}
	}
	return result;
}

Mat diff1(Mat i1,Mat i2){
	cvtColor( i1,i1, CV_RGB2GRAY );
	cvtColor( i2,i2, CV_RGB2GRAY );
	Mat result=i1;
	uchar color;
	for(int x=0;x<i1.cols;x++){
		for(int y=0;y<i1.rows;y++){
			uchar color1=i1.at<uchar>(Point(x,y));
			uchar color2=i2.at<uchar>(Point(x,y));
			color=abs(color1-color2);
			result.at<uchar>(Point(x,y)) = color;
		}
	}
	cvtColor(result,result, CV_GRAY2RGB );
	return result;
}
//不規則雜點加線
Mat diff2(Mat i1,Mat i2){
	Mat result=i1;
	int up=130;
	int dn=90;
	for(int x=0;x<i1.cols;x++){
		for(int y=0;y<i1.rows;y++){
			Vec3b color=i2.at<Vec3b>(Point(x,y));
			if(y<300){
				if(color[0]<up&&color[1]<up&&color[2]<up&&color[0]>dn&&color[1]>dn&&color[2]>dn){
					color[0]=255;
					color[1]=255;
					color[2]=255;
				}
				else{
					color[0]=0;
					color[1]=0;
					color[2]=0;
				}

			}
			else if(y>500&&y<670){
				if(color[0]<dn&&color[1]<dn&&color[2]<dn){
					color[0]=0;
					color[1]=0;
					color[2]=0;
				}
				else{
					color[0]=0;
					color[1]=0;
					color[2]=0;
				}
			}
			else{
				color[0]=0;
				color[1]=0;
				color[2]=0;
			}
			result.at<Vec3b>(Point(x,y)) = color;
		}
	}
	return result;
}
//Adj_大範圍的雜色點
Mat diff3(Mat i1,Mat i2){
	Mat result=i1;
	int up=100;
	int dn=60;
	for(int x=0;x<i1.cols;x++){
		for(int y=0;y<i1.rows;y++){
			Vec3b color=i2.at<Vec3b>(Point(x,y));
			if(y<300){
				if(color[0]<up&&color[1]<up&&color[2]<up&&color[0]>dn&&color[1]>dn&&color[2]>dn){
					color[0]=255;
					color[1]=255;
					color[2]=255;
				}
				else{
					color[0]=0;
					color[1]=0;
					color[2]=0;
				}

			}
			else{
				color[0]=0;
				color[1]=0;
				color[2]=0;
			}
			result.at<Vec3b>(Point(x,y)) = color;
		}
	}
	return result;
}
//外圍黑
Mat diff4(Mat i1,Mat i2){
	//GaussianBlur( i2, i2, Size(9,9), 0, 0 );
	Mat result=i1;
	int up=50;
	int dn=-1;
	for(int x=0;x<i1.cols;x++){
		for(int y=0;y<i1.rows;y++){
			Vec3b color=i2.at<Vec3b>(Point(x,y));
			if(y>500&&y<670){
				if(color[2]<up&&color[2]>dn&&color[0]<up&&color[0]>dn&&color[1]<up&&color[1]>dn){
					color[0]=255;
					color[1]=255;
					color[2]=255;
				}
				else{
					color[0]=0;
					color[1]=0;
					color[2]=0;
				}

			}
			else{
				color[0]=0;
				color[1]=0;
				color[2]=0;
			}
			result.at<Vec3b>(Point(x,y)) = color;
		}
	}
	return result;
}
//中央高對比白點
Mat diff5(Mat i1,Mat i2){
	Mat result=i1;
	int up=256;
	int dn=50;
	for(int x=0;x<i1.cols;x++){
		for(int y=0;y<i1.rows;y++){
			Vec3b color=i2.at<Vec3b>(Point(x,y));
			if(y<320&&y>170){
				if(color[0]<up&&color[1]<up&&color[2]<up&&color[0]>dn&&color[1]>dn&&color[2]>dn){
					color[0]=255;
					color[1]=255;
					color[2]=255;
				}
				else{
					color[0]=0;
					color[1]=0;
					color[2]=0;
				}

			}
			else{
				color[0]=0;
				color[1]=0;
				color[2]=0;
			}
			result.at<Vec3b>(Point(x,y)) = color;
		}
	}
	return result;
}