#include "ofMain.h"
#ifdef SHIFT
#undef SHIFT
#endif

using namespace dlib;
using namespace std; 

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

class ofFaceRecognition 
{
	public:

	shape_predictor sp;
	anet_type net;
	frontal_face_detector detector;
	std::vector<matrix<rgb_pixel>> faces;
	array2d<rgb_pixel> img;
	std::vector< std::vector<ofImage> > images_cluster;

	ofPixels toOf(const dlib::matrix<dlib::rgb_pixel> rgb)
	{
	    ofPixels p;
	    int w = rgb.nc();
	    int h = rgb.nr();
	    p.allocate(w, h, 1);
	    for(int y = 0; y<h; y++)
	    {
		 for(int x=0; x<w;x++)
		 {
			p.setColor(x, y, ofColor(rgb(y,x).red,	
						 rgb(y,x).green,
						 rgb(y,x).blue));
		 }
	    }
	    return p;
	}

	dlib::array2d<dlib::rgb_pixel> toDLib(const ofPixels px)
	{
	    dlib::array2d<dlib::rgb_pixel> out;
	    int width = px.getWidth();
	    int height = px.getHeight();
	    int ch = px.getNumChannels();

	    out.set_size( height, width );
	    const unsigned char* data = px.getData();
	    for ( unsigned n = 0; n < height;n++ )
	    {
		const unsigned char* v =  &data[n * width *  ch];
		for ( unsigned m = 0; m < width;m++ )
		{
		    if ( ch==1 )
		    {
			unsigned char p = v[m];
			dlib::assign_pixel( out[n][m], p );
		    }
		    else{
			dlib::rgb_pixel p;
			p.red = v[m*3];
			p.green = v[m*3+1];
			p.blue = v[m*3+2];
			dlib::assign_pixel( out[n][m], p );
		    }
		}
	    }
	    return out;
	}

	std::vector<matrix<rgb_pixel>> jitter_image( const matrix<rgb_pixel>& img )
	{
	    thread_local random_cropper cropper;
	    cropper.set_chip_dims(150,150);
	    cropper.set_randomly_flip(true);
	    cropper.set_max_object_height(0.99999);
	    cropper.set_background_crops_fraction(0);
	    cropper.set_min_object_height(0.97);
	    cropper.set_translate_amount(0.02);
	    cropper.set_max_rotation_degrees(3);

	    std::vector<mmod_rect> raw_boxes(1), ignored_crop_boxes;
	    raw_boxes[0] = shrink_rect(get_rect(img),3);
	    std::vector<matrix<rgb_pixel>> crops;

	    matrix<rgb_pixel> temp;
	    for (int i = 0; i < 100; ++i)
	    {
		cropper(img, raw_boxes, temp, ignored_crop_boxes);
		crops.push_back(move(temp));
	    }
	    return crops;
	}

	void find(ofPixels p)
	{
		img = toDLib(p);
	        std::vector<dlib::rectangle> fac = detector(img);
		for (auto face : fac)
		{
		        auto shape = sp(img, face);
		        matrix<rgb_pixel> face_chip;
		        extract_image_chip(img, get_face_chip_details(shape,150,0.25), face_chip);
		        faces.push_back(move(face_chip));
    		}
		ofLog()<<"num-faces:"<<faces.size();
	}

	void cluster()
	{
		std::vector<matrix<float,0,1>> face_descriptors = net(faces);
		std::vector<sample_pair> edges;
    		for (size_t i = 0; i < face_descriptors.size(); ++i)
    		{
        		for (size_t j = i+1; j < face_descriptors.size(); ++j)
        		{
            			if (length(face_descriptors[i]-face_descriptors[j]) < 0.6)
		                	edges.push_back(sample_pair(i,j));
        		}
    		}
		std::vector<unsigned long> labels;
		const auto num_clusters = chinese_whispers(edges, labels);
		ofLog()<< "number of people found in the image: "<< num_clusters;
       		std::vector<matrix<rgb_pixel>> temp;
		images_cluster.clear();

		for (size_t cluster_id = 0; cluster_id < num_clusters; ++cluster_id)
		{
			temp.clear();
			std::vector<ofImage> cluster_temp;
        		for (size_t j = 0; j < labels.size(); ++j)
        		{
            			if (cluster_id == labels[j]) 
				{
					temp.push_back(faces[j]);
					ofImage im = toOf(faces[j]);
					if(im.isAllocated())
					{
						cluster_temp.push_back(im);
					}
				}
        		}
			images_cluster.push_back(cluster_temp);
		        ofLog()<<"face cluster " + cast_to_string(cluster_id);
    		}
		ofLog()<<"cluster-all-ofimage:"<<images_cluster.size();
	}

	void setup(string pred = "shape_predictor_68_face_landmarks.dat", string recogn = "dlib_face_recognition_resnet_model_v1.dat" )
	{
		detector = get_frontal_face_detector();
		deserialize(ofToDataPath(pred)) >> sp;
    		deserialize(ofToDataPath(recogn)) >> net;
	}

	void draw()
	{
		int x = 0;
		int y = 0;
		for(int k = 0; k < images_cluster.size(); k++)
		{
			for(int j = 0; j < images_cluster[k].size(); j++)
			{
				images_cluster[k][j].draw(x,y);
				x+=images_cluster[k][j].getWidth();
			}
			ofDrawBitmapStringHighlight("cluster num:"+ofToString(k),0,y+13);
			x = 0;
			y+= 160;
		}
	}
};
