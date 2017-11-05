#pragma once
#include "stdafx.h"
#include "app.h"
#include "util.h"
#include "highgui.h"
#include "opencv.hpp"

#include <thread>
#include <chrono>
#include <ppl.h>
using namespace cv;

Kinect::Kinect() {
	m_bInit = false;
}

void Kinect::initialize(int type){
	if (type == 0) {
		initializeColor();
	}
	else if(type == 1){
		initializeDepth();
		initializeBody();
	}

	std::this_thread::sleep_for(std::chrono::seconds(2));
	m_bInit = true;
}

// Destructor
Kinect::~Kinect()
{
    // Finalize
    finalize();
}

// Initialize Sensor
inline void Kinect::initializeSensor()
{
    // Open Sensor
    ERROR_CHECK( GetDefaultKinectSensor( &kinect ) );

    ERROR_CHECK( kinect->Open() );

    // Check Open
    BOOLEAN isOpen = FALSE;
    ERROR_CHECK( kinect->get_IsOpen( &isOpen ) );
    if( !isOpen ){
        throw std::runtime_error( "failed IKinectSensor::get_IsOpen( &isOpen )" );
    }

	// Retrieve Coordinate Mapper
	ERROR_CHECK(kinect->get_CoordinateMapper(&coordinateMapper));
}

// Initialize Color
inline void Kinect::initializeColor()
{
	cv::setUseOptimized(true);
	initializeSensor();

    // Open Color Reader
    ComPtr<IColorFrameSource> colorFrameSource;
    ERROR_CHECK( kinect->get_ColorFrameSource( &colorFrameSource ) );
    ERROR_CHECK( colorFrameSource->OpenReader( &colorFrameReader ) );
	
    // Retrieve Color Description
    ComPtr<IFrameDescription> colorFrameDescription;
    ERROR_CHECK( colorFrameSource->CreateFrameDescription( ColorImageFormat::ColorImageFormat_Bgra, &colorFrameDescription ) );
    ERROR_CHECK( colorFrameDescription->get_Width( &colorWidth ) ); // 1920
    ERROR_CHECK( colorFrameDescription->get_Height( &colorHeight ) ); // 1080
    ERROR_CHECK( colorFrameDescription->get_BytesPerPixel( &colorBytesPerPixel ) ); // 4

    // Allocation Color Buffer
    colorBuffer.resize( colorWidth * colorHeight * colorBytesPerPixel );

	std::this_thread::sleep_for(std::chrono::seconds(2));
}

// Initialize Depth
inline void Kinect::initializeDepth()
{
	cv::setUseOptimized(true);
	initializeSensor();

	// Open Depth Reader
	ComPtr<IDepthFrameSource> depthFrameSource;
	ERROR_CHECK(kinect->get_DepthFrameSource(&depthFrameSource));
	ERROR_CHECK(depthFrameSource->OpenReader(&depthFrameReader));

	// Retrieve Depth Description
	ComPtr<IFrameDescription> depthFrameDescription;
	ERROR_CHECK(depthFrameSource->get_FrameDescription(&depthFrameDescription));
	ERROR_CHECK(depthFrameDescription->get_Width(&depthWidth)); // 512
	ERROR_CHECK(depthFrameDescription->get_Height(&depthHeight)); // 424
	ERROR_CHECK(depthFrameDescription->get_BytesPerPixel(&depthBytesPerPixel)); // 2

																				// Retrieve Depth Reliable Range
	UINT16 minReliableDistance;
	UINT16 maxReliableDistance;
	ERROR_CHECK(depthFrameSource->get_DepthMinReliableDistance(&minReliableDistance)); // 500
	ERROR_CHECK(depthFrameSource->get_DepthMaxReliableDistance(&maxReliableDistance)); // 4500
	std::cout << "Depth Reliable Range : " << minReliableDistance << " - " << maxReliableDistance << std::endl;

	// Allocation Depth Buffer
	depthBuffer.resize(depthWidth * depthHeight);

	std::this_thread::sleep_for(std::chrono::seconds(2));
}

inline void Kinect::initializeBody()
{
	// Open Body Reader
	ComPtr<IBodyFrameSource> bodyFrameSource;
	ERROR_CHECK(kinect->get_BodyFrameSource(&bodyFrameSource));
	ERROR_CHECK(bodyFrameSource->OpenReader(&bodyFrameReader));

	// Initialize Body Buffer
	Concurrency::parallel_for_each(bodies.begin(), bodies.end(), [](IBody*& body) {
		SafeRelease(body);
	});

	// Color Table for Visualization
	colors[0] = cv::Vec3b(255, 0, 0); // Blue
	colors[1] = cv::Vec3b(0, 255, 0); // Green
	colors[2] = cv::Vec3b(0, 0, 255); // Red
	colors[3] = cv::Vec3b(255, 255, 0); // Cyan
	colors[4] = cv::Vec3b(255, 0, 255); // Magenta
	colors[5] = cv::Vec3b(0, 255, 255); // Yellow
}

// Finalize
void Kinect::finalize()
{
    cv::destroyAllWindows();

    // Close Sensor
    if( kinect != nullptr ){
        kinect->Close();
    }
}

// Update Color
bool Kinect::updateColor()
{
    // Retrieve Color Frame
    ComPtr<IColorFrame> colorFrame;
    const HRESULT ret = colorFrameReader->AcquireLatestFrame( &colorFrame );

    if( FAILED( ret )){
		return false;
    }

    // Convert Format ( YUY2 -> BGRA )
    ERROR_CHECK( colorFrame->CopyConvertedFrameDataToArray( static_cast<UINT>( colorBuffer.size() ), &colorBuffer[0], ColorImageFormat::ColorImageFormat_Bgra ) );

	return true;
}

bool Kinect::updateDepth()
{
	// Retrieve Depth Frame
	ComPtr<IDepthFrame> depthFrame;
	const HRESULT ret = depthFrameReader->AcquireLatestFrame(&depthFrame);

	if (FAILED(ret)) {
		return false;
	}

	// Retrieve Depth Data
	ERROR_CHECK(depthFrame->CopyFrameDataToArray(static_cast<UINT>(depthBuffer.size()), &depthBuffer[0]));

	return true;
}

void Kinect::updateBody()
{
	// Retrieve Body Frame
	ComPtr<IBodyFrame> bodyFrame;
	const HRESULT ret = bodyFrameReader->AcquireLatestFrame(&bodyFrame);
	if (FAILED(ret)) {
		return;
	}

	// Release Previous Bodies
	Concurrency::parallel_for_each(bodies.begin(), bodies.end(), [](IBody*& body) {
		SafeRelease(body);
	});

	// Retrieve Body Data
	ERROR_CHECK(bodyFrame->GetAndRefreshBodyData(static_cast<UINT>(bodies.size()), &bodies[0]));
}

void Kinect::drawBody()
{
	// Draw Body Data to Color Data
	Concurrency::parallel_for(0, BODY_COUNT, [&](const int count) {
		const ComPtr<IBody> body = bodies[count];
		if (body == nullptr) {
			return;
		}

		// Check Body Tracked
		BOOLEAN tracked = FALSE;
		ERROR_CHECK(body->get_IsTracked(&tracked));
		if (!tracked) {
			return;
		}

		// Retrieve Joints
		std::array<Joint, JointType::JointType_Count> joints;
		ERROR_CHECK(body->GetJoints(static_cast<UINT>(joints.size()), &joints[0]));

		Concurrency::parallel_for_each(joints.begin(), joints.end(), [&](const Joint& joint) {
			// Check Joint Tracked
			if (joint.TrackingState == TrackingState::TrackingState_NotTracked) {
				return;
			}

			// Draw Joint Position
			drawEllipse(depthMat, joint, 5, colors[count]);

			// Draw Left Hand State
			if (joint.JointType == JointType::JointType_HandLeft) {
				HandState handState;
				TrackingConfidence handConfidence;
				ERROR_CHECK(body->get_HandLeftState(&handState));
				ERROR_CHECK(body->get_HandLeftConfidence(&handConfidence));

				drawHandState(depthMat, joint, handState, handConfidence);
			}

			// Draw Right Hand State
			if (joint.JointType == JointType::JointType_HandRight) {
				HandState handState;
				TrackingConfidence handConfidence;
				ERROR_CHECK(body->get_HandRightState(&handState));
				ERROR_CHECK(body->get_HandRightConfidence(&handConfidence));

				drawHandState(depthMat, joint, handState, handConfidence);
			}
		});

		/*
		// Retrieve Joint Orientations
		std::array<JointOrientation, JointType::JointType_Count> orientations;
		ERROR_CHECK( body->GetJointOrientations( JointType::JointType_Count, &orientations[0] ) );
		*/

		/*
		// Retrieve Amount of Body Lean
		PointF amount;
		ERROR_CHECK( body->get_Lean( &amount ) );
		*/
	});
}

// Draw Ellipse
inline void Kinect::drawEllipse(cv::Mat& image, const Joint& joint, const int radius, const cv::Vec3b& color, const int thickness)
{
	if (image.empty()) {
		return;
	}

	// Convert Coordinate System and Draw Joint
	DepthSpacePoint DepthSpacePoint;
	ERROR_CHECK(coordinateMapper->MapCameraPointToDepthSpace(joint.Position, &DepthSpacePoint));
	const int x = static_cast<int>(DepthSpacePoint.X + 0.5f);
	const int y = static_cast<int>(DepthSpacePoint.Y + 0.5f);
	if ((0 <= x) && (x < image.cols) && (0 <= y) && (y < image.rows)) {
		cv::circle(image, cv::Point(x, y), radius, static_cast<cv::Scalar>(color), thickness, cv::LINE_AA);
	}
}

// Draw Hand State
inline void Kinect::drawHandState(cv::Mat& image, const Joint& joint, HandState handState, TrackingConfidence handConfidence)
{
	if (image.empty()) {
		return;
	}

	// Check Tracking Confidence
	if (handConfidence != TrackingConfidence::TrackingConfidence_High) {
		return;
	}

	// Draw Hand State 
	const int radius = 75;
	const cv::Vec3b blue = cv::Vec3b(128, 0, 0), green = cv::Vec3b(0, 128, 0), red = cv::Vec3b(0, 0, 128);
	switch (handState) {
		// Open
	case HandState::HandState_Open:
		drawEllipse(image, joint, radius, green, 5);
		break;
		// Close
	case HandState::HandState_Closed:
		drawEllipse(image, joint, radius, red, 5);
		break;
		// Lasso
	case HandState::HandState_Lasso:
		drawEllipse(image, joint, radius, blue, 5);
		break;
	default:
		break;
	}
}
// Draw Color
void Kinect::drawColor()
{
    // Create cv::Mat from Color Buffer
    colorMat = cv::Mat( colorHeight, colorWidth, CV_8UC4, &colorBuffer[0] );
}

void Kinect::drawDepth()
{
	// Create cv::Mat from Depth Buffer
	depthMat = cv::Mat(depthHeight, depthWidth, CV_16UC1, &depthBuffer[0]);
}