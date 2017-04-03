#include "ofMain.h"
#ifdef SHIFT
#undef SHIFT
#endif

#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>

#include "ofFaceRecognition.h"

class ofApp : public ofBaseApp
{
	public:
	ofFaceRecognition face_recognition;

	void setup()
	{
		ofImage img("test.jpg");

		face_recognition.setup();
		face_recognition.find(img);
		face_recognition.cluster();
	}

	void update()
	{
	        ofSetWindowTitle(ofToString(ofGetFrameRate()));
	}

	void draw()
	{
		face_recognition.draw();
	}
};

int main()
{
	ofSetupOpenGL(1024,768, OF_WINDOW);
	ofRunApp( new ofApp());
}
