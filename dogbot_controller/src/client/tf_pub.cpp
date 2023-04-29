#include "ros/ros.h"
#include <tf/transform_broadcaster.h>
#include <turtlesim/Pose.h>
#include "gazebo_msgs/ModelStates.h"
#include "boost/thread.hpp"

#include "visualization_msgs/Marker.h"
#include "visualization_msgs/MarkerArray.h"
#include <Eigen/Core>

#include "geometry_msgs/Vector3.h"
/*

    Input data: 
            - attractive field
            - repulsive field
            - total field

            - Position error

*/

#define BLindex  0
#define BRindex  1
#define FLindex  2
#define FRindex  3

using namespace std;

class Tf_pub {

    public:
        Tf_pub();
        void run();
        void pub();
        void modelStateCallback(const gazebo_msgs::ModelStates & msg);
        void marker_pub();

        void vf_att_bl(geometry_msgs::Vector3 );
        void vf_att_br(geometry_msgs::Vector3 );
        void vf_att_fl(geometry_msgs::Vector3 );
        void vf_att_fr(geometry_msgs::Vector3 );


        void vf_rep_bl(geometry_msgs::Vector3 );
        void vf_rep_br(geometry_msgs::Vector3 );
        void vf_rep_fl(geometry_msgs::Vector3 );
        void vf_rep_fr(geometry_msgs::Vector3 );

    private:
        ros::NodeHandle _nh;
        ros::Subscriber _model_state_sub;
        tf::TransformBroadcaster _broadcaster;

        float _wp[3];
        float _wq[4];

        bool _first_wpose;


        //field direction
        ros::Publisher _field_pub; 
        ros::Publisher _rep_field_pub;
        ros::Publisher _total_field_pub;

        //bl br fl fr
        ros::Subscriber _vf_att[4];
        ros::Subscriber _rf_att[4];

        Eigen::Vector2d _f_a_bl;
        Eigen::Vector2d _f_a_br;
        Eigen::Vector2d _f_a_fl;
        Eigen::Vector2d _f_a_fr;
        
        Eigen::Vector2d _f_r_bl;
        Eigen::Vector2d _f_r_br;
        Eigen::Vector2d _f_r_fl;
        Eigen::Vector2d _f_r_fr;
        
};



Tf_pub::Tf_pub() {

    _model_state_sub = _nh.subscribe("/gazebo/model_states", 1, &Tf_pub::modelStateCallback, this);
    _field_pub = _nh.advertise< visualization_msgs::MarkerArray > ("/dogbot/field", 1);


    _vf_att[BLindex] = _nh.subscribe("/dogbot/att_field/bl", 1, &Tf_pub::vf_att_bl, this);
    _vf_att[BRindex] = _nh.subscribe("/dogbot/att_field/br", 1, &Tf_pub::vf_att_br, this);
    _vf_att[FLindex] = _nh.subscribe("/dogbot/att_field/fl", 1, &Tf_pub::vf_att_fl, this);
    _vf_att[FRindex] = _nh.subscribe("/dogbot/att_field/fr", 1, &Tf_pub::vf_att_fr, this);

    _rf_att[BLindex] = _nh.subscribe("/dogbot/rep_field/bl", 1, &Tf_pub::vf_rep_bl, this);
    _rf_att[BRindex] = _nh.subscribe("/dogbot/rep_field/br", 1, &Tf_pub::vf_rep_br, this);
    _rf_att[FLindex] = _nh.subscribe("/dogbot/rep_field/fl", 1, &Tf_pub::vf_rep_fl, this);
    _rf_att[FRindex] = _nh.subscribe("/dogbot/rep_field/fr", 1, &Tf_pub::vf_rep_fr, this);

    _f_a_bl << 0.0, 0.0;
    _f_a_fl << 0.0, 0.0;  
    _f_a_br << 0.0, 0.0;
    _f_a_fr << 0.0, 0.0;


    _f_r_bl << 0.0, 0.0;
    _f_r_fl << 0.0, 0.0;  
    _f_r_br << 0.0, 0.0;
    _f_r_fr << 0.0, 0.0;

    _first_wpose = false;
}


void Tf_pub::vf_att_bl(geometry_msgs::Vector3 fa_bl_) {
    _f_a_bl << fa_bl_.x, fa_bl_.y;

}

void Tf_pub::vf_att_br(geometry_msgs::Vector3 fa_br_ ) {
    _f_a_br << fa_br_.x,  fa_br_.y; 

}
void Tf_pub::vf_att_fl(geometry_msgs::Vector3 fa_fl_ ) {
    _f_a_fl << fa_fl_.x,  fa_fl_.y;

}
void Tf_pub::vf_att_fr(geometry_msgs::Vector3 fa_fr_) {
    _f_a_fr << fa_fr_.x,  fa_fr_.y;

}


void Tf_pub::vf_rep_bl(geometry_msgs::Vector3 fr_bl_) {
    _f_r_bl << fr_bl_.x, fr_bl_.y;
}

void Tf_pub::vf_rep_br(geometry_msgs::Vector3 fr_br_ ) {
    _f_r_br << fr_br_.x,  fr_br_.y; 

}
void Tf_pub::vf_rep_fl(geometry_msgs::Vector3 fr_fl_ ) {
    _f_r_fl << fr_fl_.x,  fr_fl_.y;

}
void Tf_pub::vf_rep_fr(geometry_msgs::Vector3 fr_fr_) {
    _f_r_fr << fr_fr_.x,  fr_fr_.y;
}


// Get base position and velocity
void Tf_pub::modelStateCallback(const gazebo_msgs::ModelStates & msg) {

    bool found = false;
    int index = 0;
    while( !found  && index < msg.name.size() ) {

        if( msg.name[index] == "dogbot" )
            found = true;
        else index++;
    }

    if( found ) {
        
        _wp[0] = msg.pose[index].position.x;
        _wp[1] = msg.pose[index].position.y;
        _wp[2] = msg.pose[index].position.z;


        _wq[0] = msg.pose[index].orientation.x;
        _wq[1] = msg.pose[index].orientation.y;
        _wq[2] = msg.pose[index].orientation.z;
        _wq[3] = msg.pose[index].orientation.w;
        
        _first_wpose = true;
    }
}

void Tf_pub::pub () {

    ros::Rate r(10);
    tf::Transform transform;
        
    while (ros::ok()) {

        transform.setOrigin( tf::Vector3(_wp[0],_wp[1],_wp[2]));
        tf::Quaternion q( _wq[0], _wq[1], _wq[2], _wq[3] );

        transform.setRotation(q);
        _broadcaster.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "map", "base_link"));

        r.sleep();
    }
}

void Tf_pub::marker_pub() {

    ros::Rate r(10);
    
    visualization_msgs::Marker att_field_marker[4];
    visualization_msgs::MarkerArray field_markerS;
    field_markerS.markers.resize(8);  
    visualization_msgs::Marker rep_field_marker[4];
    visualization_msgs::Marker total_field_marker[4];

    att_field_marker[BLindex].header.frame_id = "bl_foot";
    att_field_marker[BLindex].id = 0;
    att_field_marker[BLindex].header.stamp = ros::Time::now();
    att_field_marker[BLindex].ns = "basic_shapes";
    att_field_marker[BLindex].type = visualization_msgs::Marker::ARROW;
    att_field_marker[BLindex].action = visualization_msgs::Marker::ADD;
    att_field_marker[BLindex].pose.position.x = 0;
    att_field_marker[BLindex].pose.position.y = 0;
    att_field_marker[BLindex].pose.position.z = 0;
    att_field_marker[BLindex].pose.orientation.x = 0.0;
    att_field_marker[BLindex].pose.orientation.y = 0.0;
    att_field_marker[BLindex].pose.orientation.z = 0.0;
    att_field_marker[BLindex].pose.orientation.w = 1.0;
    att_field_marker[BLindex].scale.x = 0.01;
    att_field_marker[BLindex].scale.y = 0.01;
    att_field_marker[BLindex].scale.z = 0.01;
    att_field_marker[BLindex].color.r = 1.0f;
    att_field_marker[BLindex].points.resize(2);
    att_field_marker[BLindex].color.g = 0.0f;
    att_field_marker[BLindex].color.b = 0.0f;
    att_field_marker[BLindex].color.a = 1.0;
    att_field_marker[BLindex].lifetime = ros::Duration();

    att_field_marker[BRindex] = att_field_marker[0];
    att_field_marker[FLindex] = att_field_marker[1];
    att_field_marker[FRindex] = att_field_marker[2];

    rep_field_marker[BLindex] = att_field_marker[0];
    rep_field_marker[BLindex].id = BLindex+10;  
    rep_field_marker[BLindex].color.r = 0.0f;
    rep_field_marker[BLindex].color.g = 1.0f;
    rep_field_marker[BLindex].color.b = 0.0f;
    rep_field_marker[BLindex].color.a = 1.0;
    

    att_field_marker[BRindex].header.frame_id = "br_foot";
    att_field_marker[BRindex].id = BRindex;
    att_field_marker[FLindex].header.frame_id = "fl_foot";
    att_field_marker[FLindex].id = FLindex;
    att_field_marker[FRindex].header.frame_id = "fr_foot";
    att_field_marker[FRindex].id = FRindex;

    rep_field_marker[BRindex] = rep_field_marker[BLindex];
    rep_field_marker[BRindex].header.frame_id = "br_foot";
    rep_field_marker[BRindex].id = BRindex+10;
    rep_field_marker[FLindex] = rep_field_marker[BLindex];
    rep_field_marker[FLindex].header.frame_id = "fl_foot";
    rep_field_marker[FLindex].id = FLindex+10;
    rep_field_marker[FRindex] = rep_field_marker[BLindex];
    rep_field_marker[FRindex].header.frame_id = "fr_foot";
    rep_field_marker[FRindex].id = FRindex+10;

    field_markerS.markers[BLindex] = att_field_marker[BLindex];
    field_markerS.markers[BRindex] = att_field_marker[BRindex];
    field_markerS.markers[FLindex] = att_field_marker[FLindex];
    field_markerS.markers[FRindex] = att_field_marker[FRindex];


    field_markerS.markers[BLindex+4] = rep_field_marker[BLindex];
    field_markerS.markers[BRindex+4] = rep_field_marker[BRindex];
    field_markerS.markers[FLindex+4] = rep_field_marker[FLindex];
    field_markerS.markers[FRindex+4] = rep_field_marker[FRindex];
    
    while( ros::ok() ) {

        //0: bl foot
        
        field_markerS.markers[BLindex].points[1].x = _f_a_bl[0];
        field_markerS.markers[BLindex].points[1].y = _f_a_bl[1];
        field_markerS.markers[BLindex].points[1].z = 0.0;
        field_markerS.markers[BLindex].header.stamp = ros::Time::now();


        field_markerS.markers[BLindex+4].points[1].x = -_f_r_bl[0];
        field_markerS.markers[BLindex+4].points[1].y = -_f_r_bl[1];
        field_markerS.markers[BLindex+4].points[1].z = 0.0;
        field_markerS.markers[BLindex+4].header.stamp = ros::Time::now();
       
       
        //1: br foot
        field_markerS.markers[BRindex].points[1].x = _f_a_br[0];
        field_markerS.markers[BRindex].points[1].y = _f_a_br[1];
        field_markerS.markers[BRindex].points[1].z = 0.0;
        field_markerS.markers[BRindex].header.stamp = ros::Time::now();

        field_markerS.markers[BRindex+4].points[1].x = -_f_r_br[0];
        field_markerS.markers[BRindex+4].points[1].y = -_f_r_br[1];
        field_markerS.markers[BRindex+4].points[1].z = 0.0;
        field_markerS.markers[BRindex+4].header.stamp = ros::Time::now();

        //2: fl foot
        field_markerS.markers[FLindex].points[1].x = _f_a_fl[0];
        field_markerS.markers[FLindex].points[1].y = _f_a_fl[1];
        field_markerS.markers[FLindex].points[1].z = 0.0;
        field_markerS.markers[FLindex].header.stamp = ros::Time::now();

        field_markerS.markers[FLindex+4].points[1].x = -_f_r_fl[0];
        field_markerS.markers[FLindex+4].points[1].y = -_f_r_fl[1];
        field_markerS.markers[FLindex+4].points[1].z = 0.0;
        field_markerS.markers[FLindex+4].header.stamp = ros::Time::now();

        //3: fr foot
        field_markerS.markers[FRindex].points[1].x = _f_a_fr[0];
        field_markerS.markers[FRindex].points[1].y = _f_a_fr[1];
        field_markerS.markers[FRindex].points[1].z = 0.0;
        field_markerS.markers[FRindex].header.stamp = ros::Time::now();


        field_markerS.markers[FRindex+4].points[1].x = -_f_r_fr[0];
        field_markerS.markers[FRindex+4].points[1].y = -_f_r_fr[1];
        field_markerS.markers[FRindex+4].points[1].z = 0.0;
        field_markerS.markers[FRindex+4].header.stamp = ros::Time::now();
    

        _field_pub.publish( field_markerS );








        r.sleep();
    }





}

void Tf_pub::run() {
    
    boost::thread pub_t( &Tf_pub::pub, this);
    boost::thread field_pub_t( &Tf_pub::marker_pub, this);
    ros::spin();
}



int main( int argc, char** argv ) {
    
    ros::init(argc, argv, "tf_pub"); 
    Tf_pub tf;
    tf.run();

    return 1;

}