#include "iostream"
#include "../lopt.h"
#include "gazebo_msgs/ModelStates.h"
#include "sensor_msgs/JointState.h"
#include <tf/tf.h>
#include "tf_conversions/tf_eigen.h"

#include <cstdlib>
#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include "qpSWIFT/qpSWIFT.h"
#include <chrono>
#include <boost/math/special_functions/sign.hpp>


// iDynTree headers
#include <iDynTree/Model/FreeFloatingState.h>
#include <iDynTree/KinDynComputations.h>
#include <iDynTree/ModelIO/ModelLoader.h>
#include <iDynTree/Core/EigenHelpers.h>
#include <ros/package.h>
#include "std_msgs/Float64MultiArray.h"
#include "std_msgs/Float64.h"
#include "boost/thread.hpp"
#include "gazebo_msgs/SetModelState.h"
#include "gazebo_msgs/SetModelConfiguration.h"
#include <std_srvs/Empty.h>

#include <towr/nlp_formulation.h>
#include <ifopt/ipopt_solver.h>
#include <towr/terrain/examples/height_map_examples.h>
#include <towr/nlp_formulation.h>
#include <ifopt/ipopt_solver.h>
#include <towr/initialization/gait_generator.h>
#include <map>
#include <unistd.h>
#include <unordered_map>
#include "../topt.h"
#include <angles/angles.h>
#include "geometry_msgs/WrenchStamped.h"
#include <geometry_msgs/PointStamped.h>

#include "gazebo_msgs/ContactsState.h"
#include <mutex>
#include "std_msgs/Float32MultiArray.h"
#include "std_msgs/Int32.h"
#include "std_msgs/Float32.h"

#include "geometry_msgs/Twist.h"

#define BLindex  0
#define BRindex  1
#define FLindex  2
#define FRindex  3

std::mutex update_mutex;
std::mutex jstate_mutex;
std::mutex wpos_mutex;    


#define REP_FIELD       0       //0: repulsive field not taken into account
#define MIN_EXIT        0       // 1: exit from local minima
#define CHANGE_GAIT     1       // 1: CHANGE GAIT


using namespace std;

enum SWING_LEGS { L1, L2, L3}; // 4 legs, br-fl, bl-fr
enum SWING_LEG { BR, FL, BL, FR}; // 4 legs, br-fl, bl-fr


class DOGCTRL {
    public:
        DOGCTRL();
        void jointStateCallback(const sensor_msgs::JointState & msg);
        void modelStateCallback(const gazebo_msgs::ModelStates & msg);
        void run();
        void update(Eigen::Matrix4d &eigenWorld_H_base, Eigen::Matrix<double,12,1> &eigenJointPos, Eigen::Matrix<double,12,1> &eigenJointVel, Eigen::Matrix<double,6,1> &eigenBasevel, Eigen::Vector3d &eigenGravity);			

        //Create the robot
        void createrobot(std::string modelFile);

        // Compute the Jacobian
        void  computeJac();
        void  ComputeJaclinear();

        // Compute matrix transformation T needed to recompute matrices/vecotor after the coordinate transform to the CoM
        void computeTransformation(const Eigen::VectorXd &Vel_);
        void computeJacDotQDot();
        void computeJdqdCOMlinear();
        void ctrl_loop();
        void update_loop();
        void estimate_loop();
        void publish_cmd(  Eigen::VectorXd tau  );
        void eebr_cb(gazebo_msgs::ContactsStateConstPtr eebr);
        void eebl_cb(gazebo_msgs::ContactsStateConstPtr eebl);
        void eefr_cb(gazebo_msgs::ContactsStateConstPtr eefr);
        void eefl_cb(gazebo_msgs::ContactsStateConstPtr eefl);
        void pos_cmd_cb(std_msgs::Float64 _pos_cmd);
        void pos_cmd_x_cb(std_msgs::Float64 _pos_cmd_x);
        void pos_cmd_ang_cb(std_msgs::Float64 _pos_cmd_ang);
        void estimate();
        void linear_vel_cb( std_msgs::Float64 msg);
        void compute_Kpa(Eigen::Vector2d e_a);
        double compute_fr(double v);
        double saturate_x(double v);
        double saturate_y(double v);
        double saturate_xstep(double v);
        double saturate_ystep(double v);


        void target_cb(geometry_msgs::Point p);


    private:
        ros::NodeHandle _nh;
        ros::Subscriber _target_sub;
        ros::Subscriber _joint_state_sub; 
        ros::Subscriber _model_state_sub; 
        ros::Subscriber _eebl_sub;
        ros::Subscriber _eebr_sub;
        ros::Subscriber _eefl_sub;
        ros::Subscriber _eefr_sub;
        ros::Subscriber _pos_cmd_sub;
        ros::Subscriber _pos_cmd_x_sub;
        ros::Subscriber _pos_cmd_ang_sub;
        ros::Subscriber _linear_vel_sub;

        ros::Publisher estimation_pub;
        ros::Publisher traj_des_pub;
        ros::Publisher traj_pub;
        ros::Publisher  _joint_pub;
        ros::Publisher  com_vel_pub;

        ros::Publisher _rob_index_pub;
        ros::Publisher _att_field[4];
        ros::Publisher _rep_field[4];

        ros::Publisher _cmd_vel_pub;
             
        Eigen::Matrix<double,12,1> _jnt_pos; 
        Eigen::Matrix<double,12,1> _jnt_vel;
        Eigen::Matrix<double,6,1> _base_pos;
        Eigen::Matrix<double,6,1> _base_vel;
        Eigen::Matrix4d _world_H_base;
        Eigen::Matrix<double,2,2> K_pa;


        double pos_cmd_x;
        double pos_cmd_y;
        double pos_cmd_ang;
        double time_traj;
        string _model_name;
        int phase_flag;

        // Solve quadratic problem for contact forces
        Eigen::VectorXd qpproblem( Eigen::Matrix<double,6,1> &Wcom_des, Eigen::Matrix<double,12,1> &fext);
        Eigen::VectorXd qpproblemtr( Eigen::Matrix<double,6,1> &Wcom_des, Eigen::VectorXd vdotswdes, SWING_LEGS swinglegs, Eigen::Matrix<double,12,1> &fext);
        Eigen::VectorXd qpproblemol( Eigen::Matrix<double,6,1> &Wcom_des, Eigen::Vector3d vdotswdes,  SWING_LEG swingleg, Eigen::Matrix<double,12,1> &fext);
        void qpproblemcrawl(  int swingleg, towr::SplineHolder solution,  double duration);
        // get function
        Eigen::VectorXd getBiasMatrix();
        Eigen::VectorXd getGravityMatrix();
        Eigen::MatrixXd getMassMatrix();
        Eigen::MatrixXd getJacobian();
        Eigen::MatrixXd getBiasAcc();
        Eigen::MatrixXd getTransMatrix();
        Eigen::VectorXd getBiasMatrixCOM();
        Eigen::VectorXd getGravityMatrixCOM();
        Eigen::MatrixXd getMassMatrixCOM();
        Eigen::MatrixXd getMassMatrixCOM_com();
        Eigen::MatrixXd getMassMatrixCOM_joints();
        Eigen::MatrixXd getJacobianCOM();
        Eigen::MatrixXd getJacobianCOM_linear();
        Eigen::MatrixXd getBiasAccCOM();
        Eigen::MatrixXd getBiasAccCOM_linear();
        Eigen::MatrixXd getCOMpos();
        Eigen::MatrixXd getCOMvel();
        Eigen::MatrixXd getBRpos();
        Eigen::MatrixXd getBLpos();
        Eigen::MatrixXd getFLpos();
        Eigen::MatrixXd getFRpos();
        Eigen::MatrixXd getBRvel();
        Eigen::MatrixXd getBLvel();
        Eigen::MatrixXd getFLvel();
        Eigen::MatrixXd getFRvel();
        Eigen::MatrixXd getfest();
        Eigen::Matrix<double,3,3> getBRworldtransform();
        Eigen::Matrix<double,3,3> getBLworldtransform();
        Eigen::Matrix<double,3,3> getFLworldtransform();
        Eigen::Matrix<double,3,3> getFRworldtransform();
        Eigen::Matrix<double,3,1> getbrlowerleg();
        Eigen::MatrixXd getCtq();
        Eigen::MatrixXd getsolution();
        Eigen::Vector3d _target_point;

        double getMass();
        int getDoFsnumber();

        // int for DoFs number
        unsigned int n;
        // Total mass of the robot
        double robot_mass;
        // KinDynComputations element
        iDynTree::KinDynComputations kinDynComp;
        // world to floating base transformation
        iDynTree::Transform world_H_base;
        // Joint position
        iDynTree::VectorDynSize jointPos;
        // Floating base velocity
        iDynTree::Twist         baseVel;
        // Joint velocity
        iDynTree::VectorDynSize jointVel;
        // Gravity acceleration
        iDynTree::Vector3       gravity; 
        // Position vector base+joints
        iDynTree::VectorDynSize  qb;
        // Velocity vector base+joints
        iDynTree::VectorDynSize  dqb;
        // Position vector COM+joints
        iDynTree::VectorDynSize  q;
        // Velocity vector COM+joints
        iDynTree::VectorDynSize  dq;
        // Joints limit vector
        iDynTree::VectorDynSize  qmin;
        iDynTree::VectorDynSize  qmax;
        // Center of Mass Position
        iDynTree::Vector6 CoM;
        // Center of mass velocity
        iDynTree::Vector6 CoM_vel;
        //Mass matrix
        iDynTree::FreeFloatingMassMatrix MassMatrix;
        //Bias Matrix
        iDynTree::VectorDynSize Bias;
        //Gravity Matrix
        iDynTree::MatrixDynSize GravMatrix;
        // Jacobian
        iDynTree::MatrixDynSize Jac;
        // Jacobian derivative
        iDynTree::MatrixDynSize JacDot;
        //CoM Jacobian
        iDynTree::MatrixDynSize Jcom;
        // Bias acceleration J_dot*q_dot
        iDynTree::MatrixDynSize Jdqd;
        // Transformation Matrix
        iDynTree::MatrixDynSize T;
        // Transformation matrix time derivative
        iDynTree::MatrixDynSize T_inv_dot;
        //Model
        iDynTree::Model model;
        iDynTree::ModelLoader mdlLoader;
        //Mass matrix in CoM representation
        iDynTree::FreeFloatingMassMatrix MassMatrixCOM;
        //Bias Matrix in CoM representation
        iDynTree::VectorDynSize BiasCOM;
        //Gravity Matrix in CoM representation
        iDynTree::MatrixDynSize GravMatrixCOM;
        // Jacobian in CoM representation
        iDynTree::MatrixDynSize JacCOM;
        //Jacobian in CoM representation (only linear part)
        iDynTree::MatrixDynSize JacCOM_lin;
        // Bias acceleration J_dot*q_dot in CoM representation
        iDynTree::MatrixDynSize JdqdCOM;
        // Bias acceleration J_dot*q_dot in CoM representation
        iDynTree::MatrixDynSize JdqdCOM_lin;

        Eigen::VectorXd x_eigen;
        Eigen::Matrix<double,3,1> fest; // party

        Eigen::Matrix<double,18,1> Ctq;
        void ComputeCtq(Eigen::Matrix<double,18,18> Ctq_pinocchio);
        Eigen::Matrix<double,18,18> Cm;
        bool _first_wpose;
        bool _first_jpos;
        unordered_map<int, string> _id2idname;  
        unordered_map<int, int> _id2index;      
        unordered_map<int, int> _index2id; 

        Eigen::Matrix<double,3,1>  force_br, force_bl, force_fl, force_fr;
        Eigen::Matrix<double,12,1> Fgrf;
        double h_bl_prev;
        double h_br_prev;
        double h_fl_prev;
        double h_fr_prev;
        double period_st;
        double period_tot;

        Eigen::Vector2d f_r_bl;
        Eigen::Vector2d f_r_br;
        Eigen::Vector2d f_r_fl;
        Eigen::Vector2d f_r_fr;

        double rob_foot_bl;
        double rob_foot_fr;
        double rob_foot_br;
        double rob_foot_fl;
        double comb_rob;
        double comb_rob2;

        double h_bl_min;
        double h_br_min;
        double h_fr_min;
        double h_fl_min;
        int step_num;

        
        Eigen::Vector2d f_bl_prev;
        Eigen::VectorXd tau;
        Eigen::Matrix<double,12,30> Sigma_st;
        Eigen::Matrix<double,6,30> Sigma_sw;
        double x_nominal;
        double y_nominal;
        Eigen::Vector2d bl_versor;
        Eigen::Vector2d br_versor;
        Eigen::Vector2d fl_versor;
        Eigen::Vector2d fr_versor;
        Eigen::Vector2d lat_versor;

        std::vector<Eigen::Matrix<double,6,1>>  yd,  yw, w, ygamma, yd_prev,  yw_prev, w_prev, ygamma_prev;     

        OPT *_o;
        QP *myQP;

        ros::Time begin;

        bool contact_br;
        bool contact_bl;
        bool contact_fl;
        bool contact_fr;
        bool flag_exit;
        double alfa;
        bool crawl;
        bool fake_crawl;

        double gait_type;
        geometry_msgs::WrenchStamped estimation_msg;
        geometry_msgs::PointStamped p ; 
        geometry_msgs::PointStamped com_vel_p ; 
        ros::Time time_prev;
        iDynTree::Transform  World_bl;


        float _linear_vel;
};


DOGCTRL::DOGCTRL() {

    _o = new OPT(30,86,82);
    f_bl_prev<<0,0;
    h_bl_prev=0.01;
    h_br_prev=0.01;
    h_fl_prev=0.01;
    h_fr_prev=0.01;

    _joint_state_sub = _nh.subscribe("/dogbot/joint_states", 1, &DOGCTRL::jointStateCallback, this);
    _model_state_sub = _nh.subscribe("/gazebo/model_states", 1, &DOGCTRL::modelStateCallback, this);
    _eebl_sub = _nh.subscribe("/dogbot/back_left_contactsensor_state",1, &DOGCTRL::eebl_cb, this);
    _eefl_sub = _nh.subscribe("/dogbot/front_left_contactsensor_state",1, &DOGCTRL::eefl_cb, this);
    _eebr_sub = _nh.subscribe("/dogbot/back_right_contactsensor_state",1, &DOGCTRL::eebr_cb, this);
    _eefr_sub = _nh.subscribe("/dogbot/front_right_contactsensor_state",1,&DOGCTRL::eefr_cb, this);
    _pos_cmd_sub = _nh.subscribe("/pos_cmd",1,&DOGCTRL::pos_cmd_cb, this);
    _pos_cmd_x_sub = _nh.subscribe("/pos_cmd_x",1,&DOGCTRL::pos_cmd_x_cb, this);
    _pos_cmd_ang_sub = _nh.subscribe("/ang_cmd",1,&DOGCTRL::pos_cmd_ang_cb, this);
    estimation_pub = _nh.advertise<geometry_msgs::WrenchStamped>("estimation_ee", 1);
    traj_des_pub = _nh.advertise<geometry_msgs::PointStamped>("traj_des", 1);
    traj_pub = _nh.advertise<geometry_msgs::PointStamped>("traj", 1);
    _linear_vel_sub = _nh.subscribe("/human/linear_vel", 1, &DOGCTRL::linear_vel_cb, this);
    com_vel_pub=  _nh.advertise<geometry_msgs::PointStamped>("/com_vel", 1);


    //bl br fl fr
    _att_field[BLindex] = _nh.advertise<geometry_msgs::Vector3>("/dogbot/att_field/bl", 1);
    _att_field[BRindex] = _nh.advertise<geometry_msgs::Vector3>("/dogbot/att_field/br", 1);
    _att_field[FLindex] = _nh.advertise<geometry_msgs::Vector3>("/dogbot/att_field/fl", 1);
    _att_field[FRindex] = _nh.advertise<geometry_msgs::Vector3>("/dogbot/att_field/fr", 1);


    _rep_field[BLindex] = _nh.advertise<geometry_msgs::Vector3>("/dogbot/rep_field/bl", 1);
    _rep_field[BRindex] = _nh.advertise<geometry_msgs::Vector3>("/dogbot/rep_field/br", 1);
    _rep_field[FLindex] = _nh.advertise<geometry_msgs::Vector3>("/dogbot/rep_field/fl", 1);
    _rep_field[FRindex] = _nh.advertise<geometry_msgs::Vector3>("/dogbot/rep_field/fr", 1);


    _cmd_vel_pub = _nh.advertise<std_msgs::Float32>("/dogbot/cmd_vel/norm", 1);


    _rob_index_pub = _nh.advertise<std_msgs::Float32MultiArray>("/dogbot/rob_index", 1);


    _joint_pub = _nh.advertise<std_msgs::Float64MultiArray>("/dogbot/joint_position_controller/command", 1);
    _model_name = "dogbot";


    std::string path = ros::package::getPath("dogbot_description");
    path += "/urdf/dogbot.urdf";

    createrobot(path);

    model = kinDynComp.model();
	kinDynComp.setFrameVelocityRepresentation(iDynTree::MIXED_REPRESENTATION);
	// Resize matrices of the class given the number of DOFs
    n = model.getNrOfDOFs();
    
    robot_mass = model.getTotalMass();
    jointPos = iDynTree::VectorDynSize(n);
    baseVel = iDynTree::Twist();
    jointVel = iDynTree::VectorDynSize(n);
	q = iDynTree::VectorDynSize(6+n);
	dq = iDynTree::VectorDynSize(6+n);
	qb = iDynTree::VectorDynSize(6+n);
	dqb=iDynTree::VectorDynSize(6+n);
	qmin= iDynTree::VectorDynSize(n);
	qmax= iDynTree::VectorDynSize(n);
	Bias=iDynTree::VectorDynSize(n+6);
	GravMatrix=iDynTree::MatrixDynSize(n+6,1);
    MassMatrix=iDynTree::FreeFloatingMassMatrix(model) ;
    Jcom=iDynTree::MatrixDynSize(3,6+n);
	Jac=iDynTree::MatrixDynSize(24,6+n);	
	JacDot=iDynTree::MatrixDynSize(24,6+n);
	Jdqd=iDynTree::MatrixDynSize(24,1);
    T=iDynTree::MatrixDynSize(6+n,6+n);
	T_inv_dot=iDynTree::MatrixDynSize(6+n,6+n);
    MassMatrixCOM=iDynTree::FreeFloatingMassMatrix(model) ;
    BiasCOM=iDynTree::VectorDynSize(n+6);
	GravMatrixCOM=iDynTree::MatrixDynSize(n+6,1);
	JacCOM=iDynTree::MatrixDynSize(24,6+n);
	JacCOM_lin=iDynTree::MatrixDynSize(12,6+n);
	JdqdCOM=iDynTree::MatrixDynSize(24,1);
	JdqdCOM_lin=iDynTree::MatrixDynSize(12,1);
	x_eigen= Eigen::VectorXd::Zero(30);
    x_nominal = 0.186571;
    y_nominal = 0.289186;
    Eigen::Vector2d bl_vector;
    Eigen::Vector2d br_vector;
    Eigen::Vector2d fl_vector;
    Eigen::Vector2d fr_vector;

    bl_vector<<-x_nominal,-y_nominal;
    br_vector<<x_nominal,-y_nominal;
    fl_vector<<-x_nominal,y_nominal;
    fr_vector<<x_nominal,y_nominal;
    period_st=0.01;
    period_tot=0.01;
    step_num=0;
    rob_foot_bl=0;
    rob_foot_br=0;
    rob_foot_fl=0;
    rob_foot_fr=0;
    comb_rob=0;
    comb_rob2=0;

    bl_versor<<bl_vector(0)/bl_vector.norm(),bl_vector(1)/bl_vector.norm();
    br_versor<<br_vector(0)/br_vector.norm(),br_vector(1)/br_vector.norm();
    fl_versor<<fl_vector(0)/fl_vector.norm(),fl_vector(1)/fl_vector.norm();
    fr_versor<<fr_vector(0)/fr_vector.norm(),fr_vector(1)/fr_vector.norm();
    lat_versor<<1, 0;
    
    tau.resize(12);
    tau= Eigen::VectorXd::Zero(12);
    //---

    yd.resize(4);  
    yw.resize(4); 
    w.resize(4); 
    ygamma.resize(4);
    yd_prev.resize(4);  
    yw_prev.resize(4); 
    w_prev.resize(4); 
    ygamma_prev.resize(4);

    std::fill(yd.begin(), yd.end(), Eigen::Matrix<double,6,1>::Zero()); 
    std::fill(yw.begin(), yw.end(), Eigen::Matrix<double,6,1>::Zero());
    std::fill(w.begin(), w.end(), Eigen::Matrix<double,6,1>::Zero());
    std::fill(ygamma.begin(), ygamma.end(), Eigen::Matrix<double,6,1>::Zero());
    std::fill(yd_prev.begin(), yd_prev.end(), Eigen::Matrix<double,6,1>::Zero()); 
    std::fill(yw_prev.begin(), yw_prev.end(), Eigen::Matrix<double,6,1>::Zero());
    std::fill(w_prev.begin(), w_prev.end(), Eigen::Matrix<double,6,1>::Zero());
    std::fill(ygamma_prev.begin(), ygamma_prev.end(), Eigen::Matrix<double,6,1>::Zero());

    _first_wpose = false;
    _first_jpos = false;
    contact_br = true; 
    contact_bl = true; 
    contact_bl = true; 
    contact_fr = true;
    flag_exit = false;
    crawl=false;
    fake_crawl = false;

    // Joint limits
    toEigen(qmin)<<-1.75 , -1.75,-1.75,-1.75,-1.58, -2.62, -3.15, -0.02,  -1.58, -2.62, -3.15, -0.02;
    toEigen(qmax)<<1.75, 1.75, 1.75, 1.75, 3.15, 0.02, 1.58, 2.62,  3.15, 0.02, 1.58, 2.62;

    Sigma_st<< Eigen::Matrix<double,12,30>::Zero();
    Sigma_st.block(0,18,12,12)= Eigen::Matrix<double,12,12>::Identity();

    Sigma_sw<< Eigen::Matrix<double,6,30>::Zero();
    Sigma_sw.block(0,18,6,6)= Eigen::Matrix<double,6,6>::Identity();
    
    
}


void DOGCTRL::target_cb(geometry_msgs::Point p) {
    _target_point << p.x, p.y, p.z;
}

void DOGCTRL::linear_vel_cb( std_msgs::Float64 msg ) {
    _linear_vel = msg.data;
}

void DOGCTRL::createrobot(std::string modelFile) {  
    
    if( !mdlLoader.loadModelFromFile(modelFile) ) {
        std::cerr << "KinDynComputationsWithEigen: impossible to load model from " << modelFile << std::endl;
        return ;
    }
    if( !kinDynComp.loadRobotModel(mdlLoader.model()) )
    {
        std::cerr << "KinDynComputationsWithEigen: impossible to load the following model in a KinDynComputations class:" << std::endl
                  << mdlLoader.model().toString() << std::endl;
        return ;
    }

    _id2idname.insert( pair< int, string > ( 0, kinDynComp.getDescriptionOfDegreeOfFreedom(0) ));
    _id2idname.insert( pair< int, string > ( 1, kinDynComp.getDescriptionOfDegreeOfFreedom(1) ));
    _id2idname.insert( pair< int, string > ( 2, kinDynComp.getDescriptionOfDegreeOfFreedom(2) ));
    _id2idname.insert( pair< int, string > ( 3, kinDynComp.getDescriptionOfDegreeOfFreedom(3) ));
    _id2idname.insert( pair< int, string > ( 4, kinDynComp.getDescriptionOfDegreeOfFreedom(4) ));
    _id2idname.insert( pair< int, string > ( 5, kinDynComp.getDescriptionOfDegreeOfFreedom(5) ));
    _id2idname.insert( pair< int, string > ( 6, kinDynComp.getDescriptionOfDegreeOfFreedom(6) ));
    _id2idname.insert( pair< int, string > ( 7, kinDynComp.getDescriptionOfDegreeOfFreedom(7) ));
    _id2idname.insert( pair< int, string > ( 8, kinDynComp.getDescriptionOfDegreeOfFreedom(8) ));
    _id2idname.insert( pair< int, string > ( 9, kinDynComp.getDescriptionOfDegreeOfFreedom(9) ));
    _id2idname.insert( pair< int, string > ( 10, kinDynComp.getDescriptionOfDegreeOfFreedom(10) ));
    _id2idname.insert( pair< int, string > ( 11, kinDynComp.getDescriptionOfDegreeOfFreedom(11) ));

}

// Get joints position and velocity
void DOGCTRL::jointStateCallback(const sensor_msgs::JointState & msg) {

    if( _first_jpos == false ) {
        for( int i=0; i<12; i++) {
            bool found = false;
            int index = 0;
            while( !found && index <  msg.name.size() ) {
                if( msg.name[index] == _id2idname.at( i )    ) {
                    found = true;

                    _id2index.insert( pair< int, int > ( i, index ));
                    _index2id.insert( pair< int, int > ( index, i ));

                }
                else index++;
            }
        }
    }

    for( int i=0; i<12; i++ ) {
        _jnt_pos( i, 0) = msg.position[    _id2index.at(i)    ];
    }

    for( int i=0; i<12; i++ ) {
        _jnt_vel( i, 0) = msg.velocity[    _id2index.at(i)    ];
    }
    
    _first_jpos = true;
}

// Get base position and velocity
void DOGCTRL::modelStateCallback(const gazebo_msgs::ModelStates & msg) {

    bool found = false;
    int index = 0;
    while( !found  && index < msg.name.size() ) {

        if( msg.name[index] == _model_name )
            found = true;
        else index++;
    }

    if( found ) {
        
        _world_H_base.setIdentity();
        
        //quaternion
        tf::Quaternion q(msg.pose[index].orientation.x, msg.pose[index].orientation.y, msg.pose[index].orientation.z,  msg.pose[index].orientation.w);
        q.normalize();
        Eigen::Matrix<double,3,3> rot;
        tf::matrixTFToEigen(tf::Matrix3x3(q),rot);

        //Roll, pitch, yaw
        double roll, pitch, yaw;
        tf::Matrix3x3(q).getRPY(roll, pitch, yaw);

        //Set base pos (position and orientation)
        _base_pos << msg.pose[index].position.x, msg.pose[index].position.y, msg.pose[index].position.z, roll, pitch, yaw;
      
        //Set transformation matrix
        _world_H_base.block(0,0,3,3)= rot;
        _world_H_base.block(0,3,3,1)= _base_pos.block(0,0,3,1);

        //Set base vel
        _base_vel << msg.twist[index].linear.x, msg.twist[index].linear.y, msg.twist[index].linear.z, msg.twist[index].angular.x, msg.twist[index].angular.y, msg.twist[index].angular.z;
        _first_wpose = true;
    }
}

Eigen::Matrix<double,3,3> DOGCTRL::getBRworldtransform()
{    
	iDynTree::Transform  World_br;
    World_br=kinDynComp.getWorldTransform(kinDynComp.getFrameIndex("back_right_foot"));
    return toEigen(World_br.getRotation());
}

Eigen::Matrix<double,3,3> DOGCTRL::getBLworldtransform()
{    
	iDynTree::Transform  World_br;
    World_br=kinDynComp.getWorldTransform(kinDynComp.getFrameIndex("back_left_foot"));
    return toEigen(World_br.getRotation());
}

Eigen::Matrix<double,3,3> DOGCTRL::getFRworldtransform()
{    
	
	iDynTree::Transform  World_br;
    World_br=kinDynComp.getWorldTransform(kinDynComp.getFrameIndex("front_right_foot"));
    return toEigen(World_br.getRotation());
}

Eigen::Matrix<double,3,3> DOGCTRL::getFLworldtransform()
{    
	
	iDynTree::Transform  World_br;
    World_br=kinDynComp.getWorldTransform(kinDynComp.getFrameIndex("front_left_foot"));
    return toEigen(World_br.getRotation());
}

// Compute matrix transformation T needed to recompute matrices/vector after the coordinate transform to the CoM
void DOGCTRL::computeTransformation(const Eigen::VectorXd &Vel_) {
    
    //Set ausiliary matrices
    //iDynTree::MatrixDynSize Jb(6,6+n);
    //iDynTree::MatrixDynSize Jbc(3,n);
    iDynTree::Vector3 xbc;
    iDynTree::MatrixDynSize xbc_hat(3,3);
    iDynTree::MatrixDynSize xbc_hat_dot(3,3);
    //iDynTree::MatrixDynSize Jbc_dot(6,6+n);
    //iDynTree::Vector3 xbo_dot;

    //Set ausiliary matrices
    iDynTree::Vector3 xbc_dot;

    // Compute T matrix
    // Get jacobians of the floating base and of the com
    //kinDynComp.getFrameFreeFloatingJacobian(0,Jb);
    //kinDynComp.getCenterOfMassJacobian(Jcom);

    // Compute jacobian Jbc=d(xc-xb)/dq used in matrix T
    //toEigen(Jbc)<<toEigen(Jcom).block<3,12>(0,6)-toEigen(Jb).block<3,12>(0,6);

    // Get xb (floating base position) and xc ( com position)
    iDynTree::Position xb = world_H_base.getPosition();
    iDynTree::Position xc= kinDynComp.getCenterOfMassPosition();

    // Vector xcb=xc-xb
    toEigen(xbc)=toEigen(xc)-toEigen(xb);

    // Skew of xcb
    toEigen(xbc_hat)<<0, -toEigen(xbc)[2], toEigen(xbc)[1],
    toEigen(xbc)[2], 0, -toEigen(xbc)[0], -toEigen(xbc)[1], toEigen(xbc)[0], 0;

    Eigen::Matrix<double,6,6> X;
    X<<Eigen::MatrixXd::Identity(3,3), toEigen(xbc_hat).transpose(), 
    Eigen::MatrixXd::Zero(3,3), Eigen::MatrixXd::Identity(3,3);

    Eigen::MatrixXd Mb_Mj= toEigen(MassMatrix).block(0,0,6,6).bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(toEigen(MassMatrix).block(0,6,6,12));
    Eigen::Matrix<double,6,12> Js=X*Mb_Mj;

    // Matrix T for the transformation
    toEigen(T)<<Eigen::MatrixXd::Identity(3,3), toEigen(xbc_hat).transpose(), Js.block(0,0,3,12),
    Eigen::MatrixXd::Zero(3,3), Eigen::MatrixXd::Identity(3,3), Js.block(3,0,3,12),
    Eigen::MatrixXd::Zero(12,3),  Eigen::MatrixXd::Zero(12,3), Eigen::MatrixXd::Identity(12,12);

    //Compute time derivative of T 
    // Compute derivative of xbc
    toEigen(xbc_dot)=toEigen(kinDynComp.getCenterOfMassVelocity())-toEigen(baseVel.getLinearVec3());
    Eigen::VectorXd  mdr=robot_mass*toEigen(xbc_dot);
    Eigen::Matrix<double,3,3> mdr_hat;
    mdr_hat<<0, -mdr[2], mdr[1],
    mdr[2], 0, -mdr[0],                          
    -mdr[1], mdr[0], 0;

    //Compute skew of xbc
    toEigen(xbc_hat_dot)<<0, -toEigen(xbc_dot)[2], toEigen(xbc_dot)[1],
    toEigen(xbc_dot)[2], 0, -toEigen(xbc_dot)[0],                          
    -toEigen(xbc_dot)[1], toEigen(xbc_dot)[0], 0;

    Eigen::Matrix<double,6,6> dX;
    dX<<Eigen::MatrixXd::Zero(3,3), toEigen(xbc_hat_dot).transpose(),
    Eigen::MatrixXd::Zero(3,6);
    // Time derivative of Jbc
    //kinDynComp.getCentroidalAverageVelocityJacobian(Jbc_dot);

    Eigen::Matrix<double,6,6> dMb;
    dMb<<Eigen::MatrixXd::Zero(3,3), mdr_hat.transpose(),
    mdr_hat, Eigen::MatrixXd::Zero(3,3);

    Eigen::MatrixXd inv_dMb1=(toEigen(MassMatrix).block(0,0,6,6).transpose().bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(dMb.transpose())).transpose();
    Eigen::MatrixXd inv_dMb2=-(toEigen(MassMatrix).block(0,0,6,6).bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve( inv_dMb1));

    Eigen::Matrix<double,6,12> dJs=dX*Mb_Mj+X*inv_dMb2*toEigen(MassMatrix).block(0,6,6,12);

    toEigen(T_inv_dot)<<Eigen::MatrixXd::Zero(3,3), toEigen(xbc_hat_dot), -dJs.block(0,0,3,12),
    Eigen::MatrixXd::Zero(15,18);

}

//Update elements of the class given the new state
void DOGCTRL::update (Eigen::Matrix4d &eigenWorld_H_base, Eigen::Matrix<double,12,1> &eigenJointPos, Eigen::Matrix<double,12,1> &eigenJointVel, Eigen::Matrix<double,6,1> &eigenBasevel, Eigen::Vector3d &eigenGravity)
{   
   
    // Update joints, base and gravity from inputs
    iDynTree::fromEigen(world_H_base,eigenWorld_H_base);
    iDynTree::toEigen(jointPos) = eigenJointPos;
    iDynTree::fromEigen(baseVel,eigenBasevel);
    toEigen(jointVel) = eigenJointVel;
    toEigen(gravity)  = eigenGravity;

    Eigen::Vector3d worldeigen=toEigen(world_H_base.getPosition());

    while (worldeigen==Eigen::Vector3d::Zero()){
        iDynTree::fromEigen(world_H_base,eigenWorld_H_base);
        worldeigen=toEigen(world_H_base.getPosition());
        //cout<<"world update"<<world_H_base.getPosition().toString()<<endl;
    }

    //Set the state for the robot 
    kinDynComp.setRobotState(world_H_base,jointPos,
    baseVel,jointVel,gravity);

    // Compute Center of Mass
    iDynTree::Vector3 base_angle;
    toEigen(base_angle)=_base_pos.block(3,0,3,1);
    toEigen(CoM) << toEigen(kinDynComp.getCenterOfMassPosition()), toEigen(base_angle);
 

   
	//Compute velocity of the center of mass
	toEigen(CoM_vel)<<toEigen(kinDynComp.getCenterOfMassVelocity()), eigenBasevel.block(3,0,3,1);
		   
    // Compute position base +joints
	//toEigen(qb)<<toEigen(world_H_base.getPosition()), toEigen(base_angle), eigenJointPos;
    // Compute position COM+joints
	toEigen(q)<<toEigen(CoM), eigenJointPos;
   	toEigen(dq)<<toEigen(CoM_vel), eigenJointVel;
	toEigen(dqb) << eigenBasevel, eigenJointVel;
    Eigen::MatrixXd qdinv=toEigen(dqb).completeOrthogonalDecomposition().pseudoInverse();



    // Get mass, bias (C(q,v)*v+g(q)) and gravity (g(q)) matrices
    //Initialize ausiliary vector
    iDynTree::FreeFloatingGeneralizedTorques bias_force(model);
    iDynTree::FreeFloatingGeneralizedTorques grav_force(model);
    //Compute Mass Matrix
    kinDynComp.getFreeFloatingMassMatrix(MassMatrix); 
    //Compute Coriolis + gravitational terms (Bias)
    kinDynComp.generalizedBiasForces(bias_force);
    toEigen(Bias)<<iDynTree::toEigen(bias_force.baseWrench()),
        iDynTree::toEigen(bias_force.jointTorques());


    //Compute Gravitational term
    kinDynComp.generalizedGravityForces(grav_force);
    toEigen(GravMatrix)<<iDynTree::toEigen(grav_force.baseWrench()),
                         iDynTree::toEigen(grav_force.jointTorques());

    computeJac();	
    // Compute Bias Acceleration -> J_dot*q_dot
    computeJacDotQDot();
    
    Eigen::Matrix<double, 18,1> q_dot;

    q_dot<< eigenBasevel,
    eigenJointVel;

    // Compute Matrix needed for transformation from floating base representation to CoM representation
    computeTransformation(q_dot);

    // Compute Mass Matrix in CoM representation 
    toEigen(MassMatrixCOM)=toEigen(T).transpose().inverse()*toEigen(MassMatrix)*toEigen(T).inverse();

    // Compute Coriolis+gravitational term in CoM representation
    toEigen(BiasCOM)=toEigen(T).transpose().inverse()*toEigen(Bias)+toEigen(T).transpose().inverse()*toEigen(MassMatrix)*toEigen(T_inv_dot)*toEigen(dq);
    
    Ctq=(toEigen(T).transpose().inverse()*((toEigen(Bias)-toEigen(GravMatrix))*qdinv)*toEigen(T).inverse()+toEigen(T).transpose().inverse()*toEigen(MassMatrix)*toEigen(T_inv_dot)).transpose()*toEigen(dq);

    // Compute gravitational term in CoM representation	
    //toEigen(GravMatrixCOM)=toEigen(T).transpose().inverse()*toEigen(GravMatrix);

    // Compute Jacobian term in CoM representation
    toEigen(JacCOM)=toEigen(Jac)*toEigen(T).inverse();
    ComputeJaclinear();

    // Compute Bias Acceleration -> J_dot*q_dot  in CoM representation
    toEigen(JdqdCOM)=toEigen(Jdqd)+toEigen(Jac)*toEigen(T_inv_dot)*toEigen(dq);
    computeJdqdCOMlinear();
}

// Compute Jacobian
void  DOGCTRL::computeJac() {     

    //Set ausiliary matrices
    iDynTree::MatrixDynSize Jac1(6,6+n);
    iDynTree::MatrixDynSize Jac2(6,6+n);
    iDynTree::MatrixDynSize Jac3(6,6+n);
    iDynTree::MatrixDynSize Jac4(6,6+n);

    // Compute Jacobian for each leg
    // Jacobian for back right leg
    kinDynComp.getFrameFreeFloatingJacobian( kinDynComp.getFrameIndex("back_right_foot"), Jac1);

    // Jacobian for back left leg
    kinDynComp.getFrameFreeFloatingJacobian( kinDynComp.getFrameIndex("back_left_foot"),Jac2);

    // Jacobian for front left leg
    kinDynComp.getFrameFreeFloatingJacobian( kinDynComp.getFrameIndex("front_left_foot"), Jac3);

    // Jacobian for front right leg
    kinDynComp.getFrameFreeFloatingJacobian( kinDynComp.getFrameIndex("front_right_foot"), Jac4);

    // Full Jacobian
    toEigen(Jac)<<toEigen(Jac1), toEigen(Jac2), toEigen(Jac3), toEigen(Jac4);


    
}

void DOGCTRL::estimate_loop() {


     //wait for first data...
    while( !_first_wpose  )
        usleep(0.1*1e6);

    while( !_first_jpos  )
        usleep(0.1*1e6);

    ros::Rate r(1000);

while(ros::ok()){

    auto t1_c = std::chrono::high_resolution_clock::now();

    Eigen::MatrixXd Mcom=toEigen(MassMatrixCOM).block(0,0,6,6);
    Eigen::Matrix<double,6,1> q_dot=toEigen(CoM_vel);
    Eigen::Matrix<double,12,6> J=toEigen(JacCOM_lin).block(0,0,12,6);

    if(phase_flag==0)
    {
        Eigen::Matrix<double,3,3> Tbr=getBRworldtransform();
        Eigen::Matrix<double,3,3> Tbl=getBLworldtransform();
        Eigen::Matrix<double,3,3> Tfl=getFLworldtransform();
        Eigen::Matrix<double,3,3> Tfr=getFRworldtransform();
        Fgrf<< Tbr*force_br, Tbl*force_bl,Tfl*force_fl,Tfr*force_fr;
    }
    else if(phase_flag==1)
    {
        Eigen::Matrix<double,3,3> Tbr= getBRworldtransform();
        Eigen::Matrix<double,3,3> Tbl= getBLworldtransform();
        Eigen::Matrix<double,3,3> Tfl= getFLworldtransform();
        Eigen::Matrix<double,3,3> Tfr= getFRworldtransform();
        Fgrf<< Eigen::Matrix<double,3,1>::Zero(), Tbl*force_bl, Eigen::Matrix<double,3,1>::Zero(), Tfr*force_fr;
    }
    else if(phase_flag==2)
    {
        Eigen::Matrix<double,3,3> Tbr=getBRworldtransform();
        Eigen::Matrix<double,3,3> Tbl=getBLworldtransform();
        Eigen::Matrix<double,3,3> Tfl=getFLworldtransform();
        Eigen::Matrix<double,3,3> Tfr=getFRworldtransform();
        Fgrf<<Tbr*force_br, Eigen::Matrix<double,3,1>::Zero(),  Tfl*force_fl, Eigen::Matrix<double,3,1>::Zero();
    }

  
    Eigen::Matrix<double,6,1> fc=J.transpose()*Fgrf;
    Eigen::MatrixXd p1 = Mcom*q_dot;
    double mass_robot=model.getTotalMass();
    Eigen::MatrixXd g_acc=Eigen::MatrixXd::Zero(6,1);
    g_acc(2,0)=9.81;

       // std::cout <<"force" <<Fgrf << "ms\n";


    Eigen::MatrixXd intg=Ctq.block(0,0,6,1)-mass_robot*g_acc+fc;
    Eigen::MatrixXd rho=p1;
    Eigen::MatrixXd d=intg;
    Eigen::VectorXd coeffs=Eigen::VectorXd::Zero(2);
    coeffs<< 0.5, 1;
    std::vector<Eigen::Matrix<double,6,6>> k(2);

    for (int i=0; i<1; i++) {
        k[i]=coeffs[i]*Eigen::Matrix<double,6,6>::Identity();
    }

    double T=0.001;
    Eigen::Matrix<double,6,6> m=(Eigen::Matrix<double,6,6>::Identity()+k[0]*T).inverse()*k[0];
    yd[0]=yd_prev[0]+(d*T);
    w[0] = m*(rho-yw_prev[0]-yd[0]);
    yw[0]=yw_prev[0]+w[0]*T;

    yd_prev =yd;
    yw_prev =yw;
    w_prev =w;
    ygamma_prev =ygamma;

    std::cout <<"estimation" <<w[0] << "ms\n";

   
    auto t2_c = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms_double_c = t2_c - t1_c;
    std::cout <<"estimate" <<ms_double_c.count() << "ms\n";

   r.sleep();
}
}

void DOGCTRL::ComputeJaclinear() {
    
  Eigen::Matrix<double,12,24> B;
  B<< Eigen::MatrixXd::Identity(3,3) , Eigen::MatrixXd::Zero(3,21),
      Eigen::MatrixXd::Zero(3,6), Eigen::MatrixXd::Identity(3,3), Eigen::MatrixXd::Zero(3,15),
	  Eigen::MatrixXd::Zero(3,12), Eigen::MatrixXd::Identity(3,3),  Eigen::MatrixXd::Zero(3,9),
	  Eigen::MatrixXd::Zero(3,18), Eigen::MatrixXd::Identity(3,3), Eigen::MatrixXd::Zero(3,3);

  toEigen(JacCOM_lin)=B*toEigen(JacCOM);
    
}

void DOGCTRL::computeJdqdCOMlinear() {
	
    Eigen::Matrix<double,12,24> B;
    B<< Eigen::MatrixXd::Identity(3,3) , Eigen::MatrixXd::Zero(3,21),
      Eigen::MatrixXd::Zero(3,6), Eigen::MatrixXd::Identity(3,3), Eigen::MatrixXd::Zero(3,15),
	  Eigen::MatrixXd::Zero(3,12), Eigen::MatrixXd::Identity(3,3),  Eigen::MatrixXd::Zero(3,9),
	  Eigen::MatrixXd::Zero(3,18), Eigen::MatrixXd::Identity(3,3), Eigen::MatrixXd::Zero(3,3);


    toEigen(JdqdCOM_lin)= Eigen::MatrixXd::Zero(12,1);
    toEigen(JdqdCOM_lin)=B*toEigen(JdqdCOM);
	
}

// Compute Bias acceleration: J_dot*q_dot
void  DOGCTRL::computeJacDotQDot() {

    // Bias acceleration for back right leg
    iDynTree::Vector6 Jdqd1=kinDynComp.getFrameBiasAcc("back_right_foot"); 
    // Bias acceleration for back left leg
    iDynTree::Vector6 Jdqd2=kinDynComp.getFrameBiasAcc("back_left_foot"); 
    // Bias acceleration for front left leg
    iDynTree::Vector6 Jdqd3=kinDynComp.getFrameBiasAcc("front_left_foot"); 
    // Bias acceleration for front right leg
    iDynTree::Vector6 Jdqd4=kinDynComp.getFrameBiasAcc("front_right_foot"); 
    toEigen(Jdqd)<<toEigen(Jdqd1), toEigen(Jdqd2), toEigen(Jdqd3), toEigen(Jdqd4);
	
}

void  DOGCTRL::publish_cmd(  Eigen::VectorXd tau  ) {
    std_msgs::Float64MultiArray tau1_msg;
    
    // Fill Command message
    for(int i=11; i>=0; i--) {
        tau1_msg.data.push_back(  tau( _index2id.at(i) )    );
    }

    //Sending command
    _joint_pub.publish(tau1_msg);

}

void DOGCTRL::pos_cmd_cb(std_msgs::Float64 _pos_cmd){
    pos_cmd_y = _pos_cmd.data;
}

void DOGCTRL::pos_cmd_x_cb(std_msgs::Float64 _pos_cmd){
    pos_cmd_x = _pos_cmd.data;
}

void DOGCTRL::pos_cmd_ang_cb(std_msgs::Float64 _pos_cmd_ang){
    pos_cmd_ang = _pos_cmd_ang.data;
}

void DOGCTRL::eebr_cb(gazebo_msgs::ContactsStateConstPtr eebr){

	if(eebr->states.empty()){ 
        contact_br= false;
	}
	else{
		contact_br= true;
        force_br<<eebr->states[0].total_wrench.force.x, eebr->states[0].total_wrench.force.y, eebr->states[0].total_wrench.force.z;
	}
}

void DOGCTRL::eefl_cb(gazebo_msgs::ContactsStateConstPtr eefl){
	if(eefl->states.empty()){ 
        contact_fl= false;
	}
	else{
		contact_fl= true;
        force_fl<<eefl->states[0].total_wrench.force.x, eefl->states[0].total_wrench.force.y, eefl->states[0].total_wrench.force.z;
    }
}

void DOGCTRL::eebl_cb(gazebo_msgs::ContactsStateConstPtr eebl){

	if(eebl->states.empty()){ 
        contact_bl= false;
	}
	else{
		contact_bl= true;
        force_bl<<eebl->states[0].total_wrench.force.x, eebl->states[0].total_wrench.force.y, eebl->states[0].total_wrench.force.z;
    }
}

void DOGCTRL::eefr_cb(gazebo_msgs::ContactsStateConstPtr eefr){
	if(eefr->states.empty()){ 
        contact_fr= false;
	}
	else {
		contact_fr= true;
        force_fr<<eefr->states[0].total_wrench.force.x, eefr->states[0].total_wrench.force.y, eefr->states[0].total_wrench.force.z;
	}
}

void DOGCTRL::update_loop() {

   //wait for first data...
    while( !_first_wpose  )
        usleep(0.1*1e6);

    while( !_first_jpos  )
        usleep(0.1*1e6);

    Eigen::Vector3d gravity;
    gravity << 0, 0, -9.8;

    ros::Rate r(1000);

     while( ros::ok() ) {

    //auto t1_c = std::chrono::high_resolution_clock::now();

    update(_world_H_base, _jnt_pos, _jnt_vel, _base_vel, gravity);


    //auto t2_c = std::chrono::high_resolution_clock::now();

    //std::chrono::duration<double, std::milli> ms_double_c = t2_c - t1_c;
    //std::cout <<"update" <<ms_double_c.count() << "ms\n";

    r.sleep();

     }






}

void DOGCTRL::ctrl_loop() {

    ros::ServiceClient pauseGazebo = _nh.serviceClient<std_srvs::Empty>("/gazebo/pause_physics");
    ros::ServiceClient unpauseGazebo = _nh.serviceClient<std_srvs::Empty>("/gazebo/unpause_physics");
    std_srvs::Empty pauseSrv;

    _target_sub = _nh.subscribe("/dogbot/target", 1, &DOGCTRL::target_cb, this);



    _eebl_sub = _nh.subscribe("/dogbot/back_left_contactsensor_state",1, &DOGCTRL::eebl_cb, this);
    _eefl_sub = _nh.subscribe("/dogbot/front_left_contactsensor_state",1, &DOGCTRL::eefl_cb, this);
    _eebr_sub = _nh.subscribe("/dogbot/back_right_contactsensor_state",1, &DOGCTRL::eebr_cb, this);
    _eefr_sub = _nh.subscribe("/dogbot/front_right_contactsensor_state",1,&DOGCTRL::eefr_cb, this);

    //wait for first data...
    while( !_first_wpose  )
        usleep(0.1*1e6);

    while( !_first_jpos  )
        usleep(0.1*1e6);

    //update
    Eigen::Vector3d gravity;
    gravity << 0, 0, -9.8;

    update_mutex.lock();
    update(_world_H_base, _jnt_pos, _jnt_vel, _base_vel, gravity);
    update_mutex.unlock();

    ros::Rate r(400);
    
    Eigen::VectorXd CoMPosDes;
    CoMPosDes = Eigen::VectorXd::Zero(6);

    //CoMAccDes
    std_msgs::Float64MultiArray tau1_msg;
    
    //feet
	Eigen::MatrixXd eeBLposM;
    Eigen::Vector3d eeBLpos;
    Eigen::MatrixXd eeFLposM;
    Eigen::Vector3d eeFLpos;
    Eigen::MatrixXd eeFRposM;
    Eigen::Vector3d eeFRpos;
    Eigen::MatrixXd eeBRposM;
    Eigen::Vector3d eeBRpos;

    iDynTree::Vector6 des_com_pos;
    des_com_pos = CoM;
    toEigen(des_com_pos)[2] = 0.39;
    toEigen(des_com_pos)[3] = 0.0;
    toEigen(des_com_pos)[4] = 0.0;

    _target_point << toEigen( des_com_pos )[0], toEigen( des_com_pos )[1], toEigen( des_com_pos )[2]; //2, -1, 0.4;
    
    Eigen::Vector3d  goal;
    goal << 2, -1, 0.4;

    Eigen::Vector2d  goal_bl;
    Eigen::Vector2d  goal_br;
    Eigen::Vector2d  goal_fl;
    Eigen::Vector2d  goal_fr;


    /*
    goal_bl << goal(0)-0.18656, goal(1)-0.289164;
    goal_br << goal(0)+0.186998, goal(1)-0.288178;
    goal_fl << goal(0)-0.186756, goal(1)+0.286064;
    goal_fr << goal(0)+0.18695, goal(1)+0.286809273887;
    */

    K_pa << 0.1, 0, 
              0, 0.4;
  
    std_msgs::Float32MultiArray robf;
    robf.data.resize(4);

    double robf_to_mean;  
    vector<double> robf_list;

    ros::Publisher robf_mean_pub;
    std_msgs::Float64 robf_data;
    robf_data.data = 0.0;
    robf_mean_pub = _nh.advertise<std_msgs::Float64>("/dogbot/robf_mean", 1);
    

    ros::Publisher fake_crawl_pub;
    std_msgs::Int32 fake_crawl_state; // 1: trot, 0: crawl
    fake_crawl_pub = _nh.advertise<std_msgs::Int32>("/dogbot/gait_type", 1);

    while( ros::ok() ) {

        //Calculate the goal of single feet
        goal_bl << _target_point(0) - 0.186571, _target_point(1) - 0.289186;
        goal_br << _target_point(0) + 0.186571, _target_point(1) - 0.289186;
        goal_fl << _target_point(0) - 0.186571, _target_point(1) + 0.289186;
        goal_fr << _target_point(0) + 0.186571, _target_point(1) + 0.289186;

        towr::SplineHolder solution;
        towr::SplineHolder solution2;
        towr::NlpFormulation  formulation;
        towr::NlpFormulation  formulation2;
        
        des_com_pos = CoM;

        toEigen(des_com_pos)[2] = 0.4; //z
        
        toEigen(des_com_pos)[3] = 0.0;
        toEigen(des_com_pos)[4] = 0.0;
        //Set this considering the .z force on the COM
        

        CoMPosDes << toEigen(des_com_pos)[0],toEigen(des_com_pos)[1], toEigen(des_com_pos)[2], 
        toEigen(des_com_pos)[3], toEigen(des_com_pos)[4], toEigen(des_com_pos)[5];

        ros::Time now= ros::Time::now();
        p.header.stamp.sec=now.sec;
        p.header.stamp.nsec=now.nsec;              
        p.point.x = CoMPosDes[0];
        p.point.y = CoMPosDes[1];
        p.point.z = CoMPosDes[2];
        traj_des_pub.publish( p );

        p.point.x = toEigen(CoM)[0];
        p.point.y =  toEigen(CoM)[1];
        p.point.z =  toEigen(CoM)[2];
        traj_pub.publish( p );


        // Trajectory
        if(now.toSec()>7 && now.toSec()<17)
         {   CoMPosDes << toEigen(des_com_pos)[0]+0.04*sin(toEigen(CoM)[5]),toEigen(des_com_pos)[1]+0.04*sin(toEigen(CoM)[5]), 0.39, 
                          toEigen(des_com_pos)[3], toEigen(des_com_pos)[4], toEigen(des_com_pos)[5]-0.025;
             }
        else if(now.toSec()<7){
           CoMPosDes << toEigen(des_com_pos)[0],toEigen(des_com_pos)[1]-0.05, toEigen(des_com_pos)[2], 
                          toEigen(des_com_pos)[3], toEigen(des_com_pos)[4], 0.0;
        }
        else if(now.toSec()>17  && now.toSec()<24 )
         {  CoMPosDes << toEigen(des_com_pos)[0]+0.04*sin(toEigen(CoM)[5]),toEigen(des_com_pos)[1]+0.04*sin(toEigen(CoM)[5]), 0.39, 
                          toEigen(des_com_pos)[3], toEigen(des_com_pos)[4], toEigen(des_com_pos)[5];}
        else if(now.toSec()>24 && now.toSec()<34)
         {  CoMPosDes << toEigen(des_com_pos)[0]+0.04*sin(toEigen(CoM)[5]),toEigen(des_com_pos)[1]+0.04*sin(toEigen(CoM)[5]), 0.39, 
                          toEigen(des_com_pos)[3], toEigen(des_com_pos)[4], toEigen(des_com_pos)[5]+0.025;}
        else if(now.toSec()>34 )
         {  CoMPosDes << toEigen(des_com_pos)[0]+0.04*sin(toEigen(CoM)[5]),toEigen(des_com_pos)[1]-0.04*sin(toEigen(CoM)[5]), 0.39, 
                          toEigen(des_com_pos)[3], toEigen(des_com_pos)[4], toEigen(des_com_pos)[5];}

        World_bl = kinDynComp.getWorldTransform(   kinDynComp.getFrameIndex("back_left_foot")        );
	    Eigen::MatrixXd eeBLposM = toEigen(World_bl.getPosition());
        Eigen::Vector3d eeBLpos;
        eeBLpos << eeBLposM.block(0,0,3,1);


        World_bl = kinDynComp.getWorldTransform( kinDynComp.getFrameIndex("front_left_foot") );
        Eigen::MatrixXd eeFLposM = toEigen(World_bl.getPosition());
        Eigen::Vector3d eeFLpos;
        eeFLpos << eeFLposM.block(0,0,3,1);
        
        World_bl = kinDynComp.getWorldTransform( kinDynComp.getFrameIndex("front_right_foot") );
        Eigen::MatrixXd eeFRposM = toEigen(World_bl.getPosition());
        Eigen::Vector3d eeFRpos;
        eeFRpos << eeFRposM.block(0,0,3,1);

        World_bl = kinDynComp.getWorldTransform(  kinDynComp.getFrameIndex("back_right_foot") );
        Eigen::MatrixXd eeBRposM = toEigen(World_bl.getPosition());
        Eigen::Vector3d eeBRpos;
        eeBRpos << eeBRposM.block(0,0,3,1);


        Eigen::Matrix<double,6,6> Mcom;
        Mcom<< toEigen(MassMatrixCOM).block(0,0,6,6);
        if(pauseGazebo.call(pauseSrv))
            ROS_INFO("Simulation paused.");
        else
            ROS_INFO("Failed to pause simulation.");
    
         
        gait_type = 1;



      
        // Compute potential field
        
        Eigen::Vector2d e_a_bl;
        Eigen::Vector2d e_a_br;
        Eigen::Vector2d e_a_fl;
        Eigen::Vector2d e_a_fr;

        e_a_bl << saturate_x(eeBLpos(0)-goal_bl(0)),saturate_y(eeBLpos(1)-goal_bl(1));
        e_a_br << saturate_x(eeBRpos(0)-goal_br(0)),saturate_y(eeBRpos(1)-goal_br(1));
        e_a_fl << saturate_x(eeFLpos(0)-goal_fl(0)),saturate_y(eeFLpos(1)-goal_fl(1));
        e_a_fr << saturate_x(eeFRpos(0)-goal_fr(0)),saturate_y(eeFRpos(1)-goal_fr(1));

        rob_foot_bl=(0.35*rob_foot_bl+0.65*(h_bl_prev)/period_st);
        rob_foot_fr=(0.35*rob_foot_fr+0.65*(h_fr_prev)/period_st);
        rob_foot_br=(0.35*rob_foot_br+0.65*(h_br_prev)/period_st);
        rob_foot_fl=(0.35*rob_foot_fl+0.65*(h_fl_prev)/period_st);
        comb_rob=compute_fr(rob_foot_br-rob_foot_bl)+compute_fr(rob_foot_fr-rob_foot_fl)+compute_fr(abs(rob_foot_br-rob_foot_fr))+compute_fr(abs(rob_foot_bl-rob_foot_fl));
    

//repulsive fieldplot( vel_x );


        if(MIN_EXIT)
        {
            f_r_bl<<9*(rob_foot_bl)*(bl_versor)+2.2*comb_rob*lat_versor;
            f_r_br<<9*(rob_foot_br)*(br_versor)+2.2*comb_rob*lat_versor;
            f_r_fl<<9*(rob_foot_fl)*(fl_versor)+2.2*comb_rob*lat_versor;
            f_r_fr<<9*(rob_foot_fr)*(fr_versor)+2.2*comb_rob*lat_versor;
        }
        else
        {
            f_r_bl<<5*(rob_foot_bl)*(bl_versor); // 10* for case 2
            f_r_br<<5*(rob_foot_br)*(br_versor); // 10* for case 2
            f_r_fl<<5*(rob_foot_fl)*(fl_versor); // 10* for case 2
            f_r_fr<<5*(rob_foot_fr)*(fr_versor); // 10* for case 2
        }
     


        robf.data[0] = rob_foot_bl;
        robf.data[1] = rob_foot_fr;
        robf.data[2] = rob_foot_br;
        robf.data[3] = rob_foot_fl;

        //

        robf_to_mean = (rob_foot_bl + rob_foot_fr + rob_foot_br + rob_foot_fl) / 4.0;
        /*robf_list.push_back( robf_to_mean );   
        for(int i=0; i<robf_list.size(); i++ ) {
            robf_data.data += robf_list[i];
        }
        robf_data.data  /= robf_list.size();
        robf_mean_pub.publish( robf_data );
        if( robf_list.size() > 1 ) robf_list.clear(); 
        */
       robf_data.data = robf_to_mean;
       robf_mean_pub.publish( robf_data );
        

        if ( robf_data.data < 0.34 ) fake_crawl = true;
        else fake_crawl = false;
        

        if( fake_crawl )
            fake_crawl_state.data = 0;
        else
            fake_crawl_state.data = 1;
        
        fake_crawl_pub.publish(fake_crawl_state);


        compute_Kpa(e_a_bl);
        Eigen::Vector2d f_a_bl=-K_pa*e_a_bl;

        geometry_msgs::Vector3 bl_v_field;
        bl_v_field.x = f_a_bl[0];
        bl_v_field.y = f_a_bl[1];

        geometry_msgs::Vector3 bl_r_field;
        bl_r_field.x = f_r_bl[0];
        bl_r_field.y = f_r_bl[1];
    
        compute_Kpa(e_a_br);
        Eigen::Vector2d f_a_br=-K_pa*e_a_br;
        geometry_msgs::Vector3 br_v_field;
        br_v_field.x = f_a_br[0];
        br_v_field.y = f_a_br[1];


        geometry_msgs::Vector3 br_r_field;
        br_r_field.x = f_r_br[0];
        br_r_field.y = f_r_br[1];


        compute_Kpa(e_a_fl);
        Eigen::Vector2d f_a_fl=-K_pa*e_a_fl;
        geometry_msgs::Vector3 fl_v_field;
        fl_v_field.x = f_a_fl[0];
        fl_v_field.y = f_a_fl[1];

        geometry_msgs::Vector3 fl_r_field;
        fl_r_field.x = f_r_fl[0];
        fl_r_field.y = f_r_fl[1];

        compute_Kpa(e_a_fr);
        Eigen::Vector2d f_a_fr=-K_pa*e_a_fr;
        geometry_msgs::Vector3 fr_v_field;
        fr_v_field.x = f_a_fr[0];
        fr_v_field.y = f_a_fr[1];

        geometry_msgs::Vector3 fr_r_field;
        fr_r_field.x = f_r_fr[0];
        fr_r_field.y = f_r_fr[1];



        _att_field[BLindex].publish( bl_v_field ); 
        _att_field[BRindex].publish( br_v_field );
        _att_field[FLindex].publish( fl_v_field );
        _att_field[FRindex].publish( fr_v_field );

    
        _rep_field[BLindex].publish( bl_v_field ); 
        _rep_field[BRindex].publish( br_v_field );
        _rep_field[FLindex].publish( fl_v_field );
        _rep_field[FRindex].publish( fr_v_field );

        _rob_index_pub.publish( robf );

        Eigen::Vector2d bl_des; 
        Eigen::Vector2d br_des;
        Eigen::Vector2d fl_des;
        Eigen::Vector2d fr_des;


        if( REP_FIELD ) {
            bl_des<<  eeBLpos(0)+0.5*f_a_bl(0)+0.5*f_r_bl(0),  eeBLpos(1)+0.5*f_a_bl(1)+0.5*f_r_bl(1);
            br_des<<  eeBRpos(0)+0.5*f_a_br(0)+0.5*f_r_br(0),  eeBRpos(1)+0.5*f_a_br(1)+0.5*f_r_br(1);
            fl_des<<  eeFLpos(0)+0.5*f_a_fl(0)+0.5*f_r_fl(0),  eeFLpos(1)+0.5*f_a_fl(1)+0.5*f_r_fl(1);
            fr_des<<  eeFRpos(0)+0.5*f_a_fr(0)+0.5*f_r_fr(0),  eeFRpos(1)+0.5*f_a_fr(1)+0.5*f_r_fr(1);
        }
        else {
            bl_des<<  eeBLpos(0)+0.5*f_a_bl(0),  eeBLpos(1)+0.5*f_a_bl(1);
            br_des<<  eeBRpos(0)+0.5*f_a_br(0),  eeBRpos(1)+0.5*f_a_br(1);
            fl_des<<  eeFLpos(0)+0.5*f_a_fl(0),  eeFLpos(1)+0.5*f_a_fl(1);
            fr_des<<  eeFRpos(0)+0.5*f_a_fr(0),  eeFRpos(1)+0.5*f_a_fr(1);
        }

        Eigen::Vector2d com_des=(bl_des+br_des+fl_des+fr_des)/4;
        Eigen::Vector2d curr_com;
        curr_com<<toEigen(CoM)[0],toEigen(CoM)[1];

        std_msgs::Float32 norm_vel;

        Eigen::Vector2d dcom=com_des-curr_com;
        norm_vel.data = dcom.norm();


        _cmd_vel_pub.publish( norm_vel );

        CoMPosDes << saturate_xstep(com_des(0)),saturate_ystep(com_des(1)), 0.38, 
                          toEigen(des_com_pos)[3], toEigen(des_com_pos)[4], 0.0;

        time_traj=0.5;
        

        if(crawl)
        {
            if(step_num==2)
            {
                gait_type=7;
            }
            else
            {
                gait_type=4;
                step_num=0;
            }
        time_traj=1;
        }
        
            get_trajectory( toEigen( CoM ), toEigen(CoM_vel), CoMPosDes, eeBLpos, eeBRpos, eeFLpos, eeFRpos, gait_type ,time_traj, solution, formulation, Mcom(3,3),Mcom(4,4) ,Mcom(5,5) ,Mcom(3,4), Mcom(3,5), Mcom(4,5) );

        
   
        h_bl_prev=0;
        h_br_prev=0;
        h_fl_prev=0;
        h_fr_prev=0;
        h_bl_min=100;
        h_br_min=100;
        h_fr_min=100;
        h_fl_min=100;
        period_st=0;
        period_tot=0;




        unpauseGazebo.call(pauseSrv); 
        begin = ros::Time::now();
        time_prev= begin;

        while((ros::Time::now()-begin).toSec() <   formulation.params_.ee_phase_durations_.at(1)[0] && (ros::Time::now()-begin).toSec() <   formulation.params_.ee_phase_durations_.at(3)[0]  ) {

            phase_flag=0;
            
            auto t1_c = std::chrono::high_resolution_clock::now();


            // Taking Jacobian for CoM and joints
            Eigen::Matrix<double, 12, 6> Jstcom= toEigen(JacCOM_lin).block(0,0,12,6);
            Eigen::Matrix<double, 12, 12> Jstj= toEigen(JacCOM_lin).block(0,6,12,12);
            Eigen::Matrix<double, 12, 18> Jst= toEigen(JacCOM_lin);
        
            // cost function quadratic matrix
            Eigen::Matrix<double,6,30>  T_s= Jstcom.transpose()*Sigma_st;
            Eigen::Matrix<double,6,6> eigenQ1= 50*Eigen::Matrix<double,6,6>::Identity();
                
            Eigen::Matrix<double,30,30> eigenQ2= T_s.transpose()*eigenQ1*T_s;
            Eigen::Matrix<double,30,30> eigenQ= eigenQ2+Eigen::Matrix<double,30,30>::Identity();
        
            // Compute deltax, deltav
            double t = (ros::Time::now()-begin).toSec();
            Eigen::Matrix<double,6,1> CoMPosD;
            CoMPosD << solution.base_linear_->GetPoint(t).p(), solution.base_angular_->GetPoint(t).p();
            Eigen::Matrix<double,6,1> CoMVelD;
            CoMVelD << solution.base_linear_->GetPoint(t).v(), solution.base_angular_->GetPoint(t).v();
            Eigen::Matrix<double,6,1> CoMAccD;
            CoMAccD << solution.base_linear_->GetPoint(t).a(), solution.base_angular_->GetPoint(t).a();
            
            Eigen::Matrix<double,6,1> deltax = CoMPosD - toEigen( CoM );       
            deltax.block(3,0,3,1)<<_world_H_base.block(0,0,3,3)*deltax.block(3,0,3,1); 
            Eigen::Matrix<double,6,1> deltav = CoMVelD-toEigen(CoM_vel);
            Eigen::MatrixXd g_acc = Eigen::MatrixXd::Zero(6,1);
            g_acc(2,0) = 9.81;
                
            Eigen::MatrixXd M_com = toEigen(MassMatrixCOM).block(0,0,6,6);        
            Eigen::MatrixXd Kcom=3000*Eigen::MatrixXd::Identity(6,6);
            Eigen::MatrixXd Dcom=50*Eigen::MatrixXd::Identity(6,6);
                
            Eigen::Matrix<double,3,3> Tbr=getBRworldtransform();
            Eigen::Matrix<double,3,3> Tbl=getBLworldtransform();
            Eigen::Matrix<double,3,3> Tfl=getFLworldtransform();
            Eigen::Matrix<double,3,3> Tfr=getFRworldtransform();
            Fgrf<< Tbr*force_br, Tbl*force_bl,Tfl*force_fl,Tfr*force_fr;

            Eigen::Vector3d f_bl=Tbl*force_bl;
            Eigen::Vector3d f_br=Tbr*force_br;
            Eigen::Vector3d f_fl=Tfl*force_fl;
            Eigen::Vector3d f_fr=Tfr*force_fr;
       double mu=0.5;
            if(0.1<rob_foot_fl<0.25 || 0.1<rob_foot_fr<0.25 || 0.1<rob_foot_bl<0.25 || 0.1<rob_foot_br<0.25)
           {  mu=0.5;
           }
            double theta= atan(0.5);
            

            double alfa_f_bl=acos(f_bl(2)/f_bl.norm());
            double alfa_f_br=acos(f_br(2)/f_br.norm());
            double alfa_f_fl=acos(f_fl(2)/f_fl.norm());
            double alfa_f_fr=acos(f_fr(2)/f_fr.norm());
            //std::cout<<"alfa_f_bl "<<alfa_f_bl<<std::endl;
            //std::cout<<"alfa_f_bl "<<f_bl(2)/f_bl.norm()<<std::endl;

//            double h_bl= 1/((theta-alfa_f_bl)*(theta+alfa_f_bl));
//            double h_br= 1/((theta-alfa_f_br)*(theta+alfa_f_br));
//            double h_fl= 1/((theta-alfa_f_fl)*(theta+alfa_f_fl));
//            double h_fr= 1/((theta-alfa_f_fr)*(theta+alfa_f_fr));

            double h_bl= 1/(theta-alfa_f_bl);
            double h_br= 1/(theta-alfa_f_br);
            double h_fl= 1/(theta-alfa_f_fl);
            double h_fr= 1/(theta-alfa_f_fr);


            double _dt=(ros::Time::now()-time_prev).toSec();
            time_prev=ros::Time::now();
            if(!isnan(h_bl) && (1/h_bl)>0.01){
            h_bl_prev += _dt*1/h_bl;
               if((1/h_bl)<h_bl_min)
               {
                   h_bl_min=1/h_bl;}
             }
                
            if(!isnan(h_br) && (1/h_br)>0.01){
            h_br_prev += _dt*1/h_br;
             if((1/h_br)<h_br_min)
               {
                   h_br_min=1/h_br;}
            }

            if(!isnan(h_fl) && (1/h_fl)>0.01){
            h_fl_prev += _dt*1/h_fl;
            if((1/h_fl)<h_fl_min)
               {
                   h_fl_min=1/h_fl;}
            }

            if(!isnan(h_fr) && (1/h_fr)>0.01){
            h_fr_prev += _dt*1/h_fr;
            if((1/h_fr)<h_fr_min)
               {
                   h_fr_min=1/h_fr;
               }
            }


            //std::cout<<"Fgrf"<<Fgrf<<std::endl;
            // estimate();
            
            // Compute Desired vector
            Eigen::Matrix<double,6,1> Wcom_des = Kcom*deltax+Dcom*deltav+robot_mass*g_acc+toEigen(MassMatrixCOM).block(0,0,6,6)*CoMAccD;
            Eigen::Matrix<double,30,1> eigenc = -T_s.transpose()*eigenQ1.transpose()*Wcom_des;
        
            //_o->setQ( eigenQ );
            //_o->setc( eigenc );
        
            //Equality constraints
            Eigen::Matrix<double,18, 30> eigenA= Eigen::Matrix<double,18,30>::Zero();
            eigenA.block(0,0,6,6)=toEigen(MassMatrixCOM).block(0,0,6,6);
            eigenA.block(0,18,6,12)=-Jstcom.transpose();
            eigenA.block(6,0,12,6)=Jstcom;
            eigenA.block(6,6,12,12)=Jstj;

            // Known term
            Eigen::Matrix<double,18, 1> eigenb= Eigen::Matrix<double,18,1>::Zero();
            eigenb.block(0,0,6,1)=-toEigen(BiasCOM).block(0,0,6,1);
            eigenb.block(6,0,12,1)=-toEigen(JdqdCOM_lin);
        
            //Inequality Constraints
            Eigen::Matrix<double,68, 30> eigenD= Eigen::Matrix<double,68,30>::Zero();
            
            // Torque limits
            eigenD.block(20,6,12,12)=toEigen(MassMatrixCOM).block(6,6,12,12);
            eigenD.block(20,18,12,12)=-Jstj.transpose();
            eigenD.block(32,6,12,12)=-toEigen(MassMatrixCOM).block(6,6,12,12);
            eigenD.block(32,18,12,12)=Jstj.transpose();
            eigenD.block(44,6,12,12)=Eigen::Matrix<double,12,12>::Identity();
            eigenD.block(56,6,12,12)=-Eigen::Matrix<double,12,12>::Identity();
        
            //Friction
           
            Eigen::Matrix<double,3, 1> n= Eigen::Matrix<double,3,1>::Zero();
            n<< 0, 0, 1;

            Eigen::Matrix<double,3, 1> t1= Eigen::Matrix<double,3,1>::Zero();
            t1<< 1, 0, 0;
        
            Eigen::Matrix<double,3, 1> t2= Eigen::Matrix<double,3,1>::Zero();
            t2<<0, 1, 0;
        
            Eigen::Matrix<double,5,3> cfr=Eigen::Matrix<double,5,3>::Zero();
        
            cfr<<(-mu*n+t1).transpose(),
                    (-mu*n+t2).transpose(),
                    -(mu*n+t1).transpose(),
                    -(mu*n+t2).transpose(),
                    -n.transpose();
            
            Eigen::Matrix<double,20,12> Dfr=Eigen::Matrix<double,20,12>::Zero();
            for(int i=0; i<4; i++) {
                Dfr.block(0+5*i,0+3*i,5,3)=cfr;
            }
                
            eigenD.block(0,18,20,12)=Dfr;
            // Known terms for inequality
            Eigen::Matrix<double,68, 1> eigenC= Eigen::Matrix<double,68,1>::Zero();
        
            // Torque limits
            Eigen::Matrix<double,12,1> tau_max=60*Eigen::Matrix<double,12,1>::Ones();
            Eigen::Matrix<double,12,1> tau_min=-60*Eigen::Matrix<double,12,1>::Ones();
            Eigen::Matrix<double,12, 1> eigenBiascom=toEigen(BiasCOM).block(6,0,12,1);

            eigenC.block(20,0,12,1)=tau_max-eigenBiascom;
            eigenC.block(32,0,12,1)=-(tau_min-eigenBiascom);
        
            // Joints limits
            double deltat=0.025;
            Eigen::Matrix<double,12, 1> eigenq=toEigen(q).block(6,0,12,1);
            Eigen::Matrix<double,12, 1> eigendq=toEigen(dq).block(6,0,12,1);
            Eigen::Matrix<double,12, 1> eigenqmin=toEigen(qmin);
            Eigen::Matrix<double,12, 1> eigenqmax=toEigen(qmax);
            Eigen::Matrix<double,12, 1> ddqmin=(2/pow(deltat,2))*(eigenqmin-eigenq-deltat*eigendq);
            Eigen::Matrix<double,12, 1> ddqmax=(2/pow(deltat,2))*(eigenqmax-eigenq-deltat*eigendq);

            eigenC.block(44,0,12,1)=ddqmax;
            eigenC.block(56,0,12,1)=-ddqmin;          
            
            myQP = QP_SETUP_dense(30, 68, 18, eigenQ.data(), eigenA.data(), eigenD.data(), eigenc.data(), eigenC.data(), eigenb.data(), NULL, COLUMN_MAJOR_ORDERING); 
            //myQP->options->maxit  = 30;
            myQP->options->reltol = 1e-2;
            myQP->options->abstol  = 1e-2;

   
            
            qp_int ExitCode = QP_SOLVE(myQP);


            Eigen::VectorXd x_;
            x_.resize( 30 );
        
        for( int i=0; i<30; i++ ) {
            x_(i) = myQP->x[i];
        }
            //_o->opt_stance( x_ );
                    
            tau=toEigen(MassMatrixCOM).block(6,6,12,12)*x_.block(6,0,12,1)+eigenBiascom-Jstj.transpose()*x_.block(18,0,12,1);
            publish_cmd( tau );

            estimation_msg.header.stamp = ros::Time::now();
            estimation_msg.wrench.force.x = w[0][0];
            estimation_msg.wrench.force.y =w[0][1];
            estimation_msg.wrench.force.z = w[0][2];
            estimation_msg.wrench.torque.x = w[0][3];
            estimation_msg.wrench.torque.y = w[0][4];
            estimation_msg.wrench.torque.z = w[0][5];

            com_vel_p.header.stamp = ros::Time::now();
            com_vel_p.point.x=toEigen(CoM_vel)[0];
            com_vel_p.point.y=toEigen(CoM_vel)[1];
            com_vel_p.point.z=toEigen(CoM_vel)[2];
            com_vel_pub.publish(com_vel_p);


            estimation_pub.publish(estimation_msg);


            auto t2_c = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> ms_double = t2_c - t1_c;
            //std::cout <<"ctrl loop"<< ms_double.count() << "ms\n";
        
            r.sleep();
        }

        
        period_st= period_st+(ros::Time::now()-begin).toSec();

        flag_exit = false;
        
        if(gait_type ==4){
        while((ros::Time::now()-begin).toSec() <  formulation.params_.ee_phase_durations_.at(1)[1]+formulation.params_.ee_phase_durations_.at(1)[0] &&  flag_exit == false ) 
        {
         double duration= formulation.params_.ee_phase_durations_.at(1)[1]+formulation.params_.ee_phase_durations_.at(1)[0];
        qpproblemcrawl(0, solution, duration);
            }
        }
        
        if(gait_type ==7){
        while((ros::Time::now()-begin).toSec() <  formulation.params_.ee_phase_durations_.at(3)[1]+formulation.params_.ee_phase_durations_.at(3)[0] &&  flag_exit == false ) 
        {
         double duration= formulation.params_.ee_phase_durations_.at(3)[1]+formulation.params_.ee_phase_durations_.at(3)[0];
        qpproblemcrawl(3, solution, duration);
            }
        }

        if(gait_type ==1) {

            time_prev=ros::Time::now();

            while((ros::Time::now()-begin).toSec() <  formulation.params_.ee_phase_durations_.at(1)[1]+formulation.params_.ee_phase_durations_.at(1)[0] &&  flag_exit == false) {
                
                phase_flag=1;
                update_mutex.lock();
                //update(_world_H_base, _jnt_pos, _jnt_vel, _base_vel, gravity);
                update_mutex.unlock();
                double duration= formulation.params_.ee_phase_durations_.at(1)[1]+formulation.params_.ee_phase_durations_.at(1)[0];
        
                // Taking Jacobian for CoM and joints
                int swl1, swl2, stl1, stl2;
                swl1=0;
                swl2=6 ;
                stl1=3;
                stl2=9 ;
                Eigen::Matrix<double, 6, 18> Jst= Eigen::Matrix<double,6,18>::Zero();
                Jst.block(0,0,3,18)=toEigen(JacCOM_lin).block(stl1,0,3,18);
                Jst.block(3,0,3,18)=toEigen(JacCOM_lin).block(stl2,0,3,18);
        
                Eigen::Matrix<double, 6, 18> Jsw= Eigen::Matrix<double,6,18>::Zero();
                Jsw.block(0,0,3,18)=toEigen(JacCOM_lin).block(swl1,0,3,18);
                Jsw.block(3,0,3,18)=toEigen(JacCOM_lin).block(swl2,0,3,18);
        
                // cost function quadratic matrix
                
                Eigen::Matrix<double,6,30>  T_s= Jst.block(0,0,6,6).transpose()*Sigma_sw;
                
                Eigen::Matrix<double,6,6> eigenQ1= 50*Eigen::Matrix<double,6,6>::Identity();
                
                Eigen::Matrix<double,30,30> eigenQ2= T_s.transpose()*eigenQ1*T_s;
                Eigen::Matrix<double,30,30> eigenR= Eigen::Matrix<double,30,30>::Identity();
                
                eigenR.block(24,24,6,6)=100000000*Eigen::Matrix<double,6,6>::Identity();
                
                Eigen::Matrix<double,30,30> eigenQ= eigenQ2+eigenR;

                // Compute deltax, deltav
                double t = (ros::Time::now()-begin).toSec();
                Eigen::Matrix<double,6,1> CoMPosD;
                CoMPosD << solution.base_linear_->GetPoint(t).p(), solution.base_angular_->GetPoint(t).p();
                Eigen::Matrix<double,6,1> CoMVelD;
                CoMVelD << solution.base_linear_->GetPoint(t).v(), solution.base_angular_->GetPoint(t).v();
                Eigen::Matrix<double,6,1> CoMAccD;
                CoMAccD << solution.base_linear_->GetPoint(t).a(), solution.base_angular_->GetPoint(t).a();
                
                Eigen::Matrix<double,6,1> deltax = CoMPosD - toEigen( CoM );        
                deltax.block(3,0,3,1)<<_world_H_base.block(0,0,3,3)*deltax.block(3,0,3,1);  
                //cout << "Errore di posizione: " << deltax.transpose() << endl;
                //cout << "ComPosD: " << CoMPosD << endl;
                //cout << "CoM: " << toEigen( CoM ) << endl;
                Eigen::Matrix<double,6,1> deltav = CoMVelD-toEigen(CoM_vel);
                Eigen::MatrixXd g_acc = Eigen::MatrixXd::Zero(6,1);
                g_acc(2,0)=9.81;
                
                Eigen::MatrixXd M_com = toEigen(MassMatrixCOM).block(0,0,6,6);        
                Eigen::MatrixXd Kcom=3000*Eigen::MatrixXd::Identity(6,6);
                Eigen::MatrixXd Dcom=50*Eigen::MatrixXd::Identity(6,6);


                Eigen::Matrix<double,3,3> Tbr= getBRworldtransform();
                Eigen::Matrix<double,3,3> Tbl= getBLworldtransform();
                Eigen::Matrix<double,3,3> Tfl= getFLworldtransform();
                Eigen::Matrix<double,3,3> Tfr= getFRworldtransform();
                Fgrf<< Eigen::Matrix<double,3,1>::Zero(), Tbl*force_bl, Eigen::Matrix<double,3,1>::Zero(), Tfr*force_fr;

                Eigen::Vector3d f_bl=Tbl*force_bl;
                Eigen::Vector3d f_fr=Tfr*force_fr;
        double mu=0.5;
            if(0.1<rob_foot_fl<0.25 || 0.1<rob_foot_fr<0.25 || 0.1<rob_foot_bl<0.25 || 0.1<rob_foot_br<0.25)
           {  mu=0.5;
           }
            double theta= atan(0.5);        
                double alfa_f_bl=acos(f_bl(2)/f_bl.norm());
                double alfa_f_fr=acos(f_fr(2)/f_fr.norm());
        
                double h_bl= 1/((theta-alfa_f_bl)*(theta+alfa_f_bl));
                double h_fr= 1/((theta-alfa_f_fr)*(theta+alfa_f_fr));

                
                double _dt=(ros::Time::now()-time_prev).toSec();
                time_prev=ros::Time::now();

               // if(!isnan(h_bl) && (1/h_bl)>0.1){
               //    h_bl_prev += _dt*1/h_bl;
                  //    if((1/h_bl)<h_bl_min)
                  // {
                  //     h_bl_min=1/h_bl;
                  //   //  f_bl_prev<<-f_bl(0),-f_bl(1);
                  //     //std::cout<<"f_bl if "<<f_bl<<std::endl;
   //
                  //     }
    
              //  }
                
               // f_bl_prev<<f_bl_prev(0)+_dt*f_bl(0),f_bl_prev(1)+_dt*f_bl(1);}


           //     if(!isnan(h_fr) && (1/h_fr)>0.1){
           // h_fr_prev += _dt*1/h_fr;
            
             //if((1/h_fr)<h_fr_min)
             //      {
             //          h_fr_min=1/h_fr;
             //        //  f_bl_prev<<-f_bl(0),-f_bl(1);
             //          //std::cout<<"f_bl if "<<f_bl<<std::endl;
   //
             //          }
            
            //}
        
              //  std::cout<<"f_bl "<<f_bl<<std::endl;
//
              //  std::cout<<"h_bl"<<1/h_bl<<std::endl;
                
                // estimate();
                
                // Compute Desired vector
                Eigen::Matrix<double,6,1> Wcom_des = Kcom*deltax+Dcom*deltav+robot_mass*g_acc+toEigen(MassMatrixCOM).block(0,0,6,6)*CoMAccD;
                Eigen::Matrix<double,30,1> eigenc = -T_s.transpose()*eigenQ1.transpose()*Wcom_des;
        
               // _o->setQ( eigenQ );
               // _o->setc( eigenc );
        
        
                //Equality constraints
                Eigen::Matrix<double,12, 30> eigenA= Eigen::Matrix<double,12,30>::Zero();
                eigenA.block(0,0,6,6)=toEigen(MassMatrixCOM).block(0,0,6,6);
                eigenA.block(0,18,6,6)=-Jst.block(0,0,6,6).transpose();
                eigenA.block(6,0,6,6)=Jst.block(0,0,6,6);
                eigenA.block(6,6,6,12)=Jst.block(0,6,6,12);
        
                // Known term
                Eigen::Matrix<double,12, 1> eigenb= Eigen::Matrix<double,12,1>::Zero();
                Eigen::Matrix<double,6,1> Jdqdst= Eigen::Matrix<double,6,1>::Zero();
                Jdqdst<<toEigen(JdqdCOM_lin).block(stl1,0,3,1),
                        toEigen(JdqdCOM_lin).block(stl2,0,3,1);
            
                //Inequality Constraints
                Eigen::Matrix<double,70,30> eigenD= Eigen::Matrix<double,70,30>::Zero();
        
                    // Torque limits
                eigenD.block(10,6,12,12)=toEigen(MassMatrixCOM).block(6,6,12,12);   
                eigenD.block(10,18,12,6)=-Jst.block(0,6,6,12).transpose();
                eigenD.block(22,6,12,12)=-toEigen(MassMatrixCOM).block(6,6,12,12);
                eigenD.block(22,18,12,6)=Jst.block(0,6,6,12).transpose();
                eigenD.block(34,0,3,6)=Jsw.block(0,0,3,6);        
                eigenD.block(34,6,3,12)=Jsw.block(0,6,3,12);
                eigenD.block(37,0,3,6)=Jsw.block(3,0,3,6);
                eigenD.block(37,6,3,12)=Jsw.block(3,6,3,12);
                eigenD.block(34,24,3,3)=-Eigen::Matrix<double,3,3>::Identity();
                eigenD.block(37,27,3,3)=-Eigen::Matrix<double,3,3>::Identity();
                eigenD.block(40,0,3,6)=-Jsw.block(0,0,3,6);
                eigenD.block(40,6,3,12)=-Jsw.block(0,6,3,12);
                eigenD.block(43,0,3,6)=-Jsw.block(3,0,3,6);
                eigenD.block(43,6,3,12)=-Jsw.block(3,6,3,12);
                eigenD.block(40,24,3,3)=-Eigen::Matrix<double,3,3>::Identity();
                eigenD.block(43,27,3,3)=-Eigen::Matrix<double,3,3>::Identity();
                eigenD.block(46,6,12,12)=Eigen::Matrix<double,12,12>::Identity();
                eigenD.block(58,6,12,12)=-Eigen::Matrix<double,12,12>::Identity();
                //Friction
                
                Eigen::Matrix<double,3, 1> n= Eigen::Matrix<double,3,1>::Zero();
                n<< 0,
                    0,
                    1;
        
                Eigen::Matrix<double,3, 1> t1= Eigen::Matrix<double,3,1>::Zero();
                t1<< 1,
                    0,
                    0;
        
                Eigen::Matrix<double,3, 1> t2= Eigen::Matrix<double,3,1>::Zero();
                t2<<0,
                    1,
                    0;
        
                Eigen::Matrix<double,5,3> cfr=Eigen::Matrix<double,5,3>::Zero();
        
                cfr<<(-mu*n+t1).transpose(),
                    (-mu*n+t2).transpose(),
                    -(mu*n+t1).transpose(),
                    -(mu*n+t2).transpose(),
                    -n.transpose();
                
                Eigen::Matrix<double,10,6> Dfr=Eigen::Matrix<double,10,6>::Zero();
        
                for(int i=0; i<2; i++)
                {
                    Dfr.block(0+5*i,0+3*i,5,3)=cfr;
                }
                    
            
                eigenD.block(0,18,10,6)=Dfr;
            
                
                // Known terms for inequality
                Eigen::Matrix<double,70, 1> eigenC= Eigen::Matrix<double,70,1>::Zero();
                
                // Torque limits
                Eigen::Matrix<double,12,1> tau_max=60*Eigen::Matrix<double,12,1>::Ones();
                Eigen::Matrix<double,12,1> tau_min=-60*Eigen::Matrix<double,12,1>::Ones();
            
                Eigen::Matrix<double,12, 1> eigenBiascom=toEigen(BiasCOM).block(6,0,12,1);
        
            
                eigenC.block(10,0,12,1)=tau_max-eigenBiascom;
                eigenC.block(22,0,12,1)=-(tau_min-eigenBiascom);
            
                // Joints limits
                double deltat=0.025;
                Eigen::Matrix<double,12, 1> eigenq=toEigen(q).block(6,0,12,1);
                Eigen::Matrix<double,12, 1> eigendq=toEigen(dq).block(6,0,12,1);
                Eigen::Matrix<double,12, 1> eigenqmin=toEigen(qmin);
                Eigen::Matrix<double,12, 1> eigenqmax=toEigen(qmax);
                Eigen::Matrix<double,12, 1> ddqmin=(2/pow(deltat,2))*(eigenqmin-eigenq-deltat*eigendq);
                Eigen::Matrix<double,12, 1> ddqmax=(2/pow(deltat,2))*(eigenqmax-eigenq-deltat*eigendq);
        
                eigenC.block(46,0,12,1)=ddqmax;
                eigenC.block(58,0,12,1)=-ddqmin;
        
        
                Eigen::Matrix<double,6,1> Jdqdsw= Eigen::Matrix<double,6,1>::Zero();
                Jdqdsw<<toEigen(JdqdCOM_lin).block(swl1,0,3,1),
                        toEigen(JdqdCOM_lin).block(swl2,0,3,1);    
                    
                Eigen::Matrix<double,6,1> accdes;

                accdes<< solution.ee_motion_.at(1)->GetPoint(t).a(),
                            solution.ee_motion_.at(2)->GetPoint(t).a();

                World_bl = kinDynComp.getWorldTransform("back_right_foot");
                Eigen::MatrixXd eeBRposM = toEigen(World_bl.getPosition());
                Eigen::Vector3d eeBRpos;
                eeBRpos << eeBRposM.block(0,0,3,1);

                World_bl = kinDynComp.getWorldTransform("front_left_foot");
                Eigen::MatrixXd eeFLposM = toEigen(World_bl.getPosition());
                Eigen::Vector3d eeFLpos;
                eeFLpos << eeFLposM.block(0,0,3,1);

                iDynTree::Twist br_vel;
                br_vel=kinDynComp.getFrameVel("back_right_foot");
                Eigen::MatrixXd eeBRvelM = toEigen(br_vel.getLinearVec3() );
                Eigen::Vector3d eeBRvel;
                eeBRvel << eeBRvelM.block(0,0,3,1);

                iDynTree::Twist fl_vel;
                fl_vel=kinDynComp.getFrameVel("front_left_foot");
                Eigen::MatrixXd eeFLvelM = toEigen(fl_vel.getLinearVec3() );
                Eigen::Vector3d eeFLvel;
                eeFLvel << eeFLvelM.block(0,0,3,1);


                Eigen::Matrix<double,6,1> posdelta;
                posdelta<< solution.ee_motion_.at(1)->GetPoint(t).p()-eeBRpos,
                        solution.ee_motion_.at(2)->GetPoint(t).p()-eeFLpos;

                //cout<<"posdes sw 1"<< solution.ee_motion_.at(1)->GetPoint(t).p()<<endl;
                //cout<<"posdes sw 2"<< solution.ee_motion_.at(2)->GetPoint(t).p()<<endl;
        
                Eigen::Matrix<double,6,1> veldelta;
                veldelta<< solution.ee_motion_.at(1)->GetPoint(t).v()-eeBRvel,
                        solution.ee_motion_.at(2)->GetPoint(t).v()-eeFLvel;
                
                Eigen::MatrixXd Kp;
                Kp=300*Eigen::MatrixXd::Identity(6,6);
                Eigen::MatrixXd Kd;
                Kd=20*Eigen::MatrixXd::Identity(6,6);
        
                Eigen::Matrix<double,6,1> vdotswdes=accdes+Kd*veldelta+Kp*posdelta;


                eigenC.block(34,0,6,1)= vdotswdes-Jdqdsw;
                eigenC.block(40,0,6,1)= -vdotswdes+Jdqdsw;
        
                
                //Linear constraints matrix
               // Eigen::Matrix<double,82, 31> eigenL= Eigen::Matrix<double,82,31>::Zero();
        //
               // eigenL<< eigenA,eigenb,
               //             eigenD, eigenC;
        //

                
            
            
            myQP = QP_SETUP_dense(30, 70, 12, eigenQ.data(), eigenA.data(), eigenD.data(), eigenc.data(), eigenC.data(), eigenb.data(), NULL, COLUMN_MAJOR_ORDERING); 
            //myQP->options->maxit  = 30;
            myQP->options->reltol = 1e-2;
            myQP->options->abstol  = 1e-2;


            qp_int ExitCode = QP_SOLVE(myQP);
        
                Eigen::VectorXd x_;
                x_.resize( 30 );

                   for( int i=0; i<30; i++ ) {
            x_(i) = myQP->x[i];
        }
               // _o->opt_swing( x_ );
                
        //////////////////////////////
                
                tau=toEigen(MassMatrixCOM).block(6,6,12,12)*x_.block(6,0,12,1)+eigenBiascom-Jst.block(0,6,6,12).transpose()*x_.block(18,0,6,1);
                publish_cmd( tau );


                if( (contact_br==true || contact_fl==true) && t>duration-0.05)
                    {flag_exit=true;}
                
                estimation_msg.header.stamp = ros::Time::now();

                estimation_msg.wrench.force.x = w[0][0];
                estimation_msg.wrench.force.y =w[0][1];
                estimation_msg.wrench.force.z = w[0][2];
        
        
                estimation_msg.wrench.torque.x = w[0][3];
                estimation_msg.wrench.torque.y = w[0][4];
                estimation_msg.wrench.torque.z = w[0][5];
        
                com_vel_p.header.stamp = ros::Time::now();
                com_vel_p.point.x=toEigen(CoM_vel)[0];
                com_vel_p.point.y=toEigen(CoM_vel)[1];
                com_vel_p.point.z=toEigen(CoM_vel)[2];
                com_vel_pub.publish(com_vel_p);

                estimation_pub.publish(estimation_msg);

                r.sleep();

            
            } 
        }

        step_num+=1;
        period_tot=period_tot+(ros::Time::now()-begin).toSec();

        CoMPosDes << toEigen(des_com_pos)[0], toEigen(des_com_pos)[1],  toEigen(des_com_pos)[2], 
            toEigen(des_com_pos)[3], toEigen(des_com_pos)[4], toEigen(des_com_pos)[5];
        
        now = ros::Time::now();

    
        p.header.stamp.sec=now.sec;
        p.header.stamp.nsec=now.nsec;  
        
        p.point.x = CoMPosDes[0];
        p.point.y =  CoMPosDes[1];
        p.point.z =  CoMPosDes[2];
    
        traj_des_pub.publish( p );
        p.point.x = toEigen(CoM)[0];
        p.point.y =  toEigen(CoM)[1];
        p.point.z =  toEigen(CoM)[2];

        traj_pub.publish( p );
      

        des_com_pos = CoM;
        
        toEigen(des_com_pos)[2] = 0.4;
        toEigen(des_com_pos)[3] = 0.0;
        toEigen(des_com_pos)[4] = 0.0;
        


        World_bl = kinDynComp.getWorldTransform("back_left_foot");
        eeBLposM = toEigen(World_bl.getPosition());
        eeBLpos = eeBLposM.block(0,0,3,1);
        //cout << "Eeblpos " << eeBLpos << endl;

        World_bl = kinDynComp.getWorldTransform("front_left_foot");
        eeFLposM = toEigen(World_bl.getPosition());
        eeFLpos = eeFLposM.block(0,0,3,1);
        //cout << "Eeflpos " << eeFLpos << endl;

        World_bl = kinDynComp.getWorldTransform("front_right_foot");
        eeFRposM = toEigen(World_bl.getPosition());
        eeFRpos = eeFRposM.block(0,0,3,1);

        //cout << "Eefrpos " << eeFRpos << endl;

        World_bl = kinDynComp.getWorldTransform("back_right_foot");
        eeBRposM = toEigen(World_bl.getPosition());
        eeBRpos = eeBRposM.block(0,0,3,1);
        //cout << "Eebrpos " << eeBRpos << endl;

        Mcom<< toEigen(MassMatrixCOM).block(0,0,6,6);
        

        if(pauseGazebo.call(pauseSrv))
            ROS_INFO("Simulation paused.");
        else
            ROS_INFO("Failed to pause simulation.");

        double period= (ros::Time::now()-begin).toSec();

        

        rob_foot_bl=(0.35*rob_foot_bl+0.65*(h_bl_prev)/period_st);
        rob_foot_fr=(0.35*rob_foot_fr+0.65*(h_fr_prev)/period_st);
        rob_foot_br=(0.35*rob_foot_br+0.65*(h_br_prev)/period_st);
        rob_foot_fl=(0.35*rob_foot_fl+0.65*(h_fl_prev)/period_st);


        std::cout<<"comb_rob"<<comb_rob<<std::endl;
        gait_type=2;


        e_a_bl << eeBLpos(0)-goal_bl(0),eeBLpos(1)-goal_bl(1);
        e_a_br << eeBRpos(0)-goal_br(0),eeBRpos(1)-goal_br(1);
        e_a_fl << eeFLpos(0)-goal_fl(0),eeFLpos(1)-goal_fl(1);
        e_a_fr << eeFRpos(0)-goal_fr(0),eeFRpos(1)-goal_fr(1);



        if( REP_FIELD ) {
            bl_des<<  eeBLpos(0)+0.5*f_a_bl(0)+0.5*f_r_bl(0),  eeBLpos(1)+0.5*f_a_bl(1)+0.5*f_r_bl(1);
            br_des<<  eeBRpos(0)+0.5*f_a_br(0)+0.5*f_r_br(0),  eeBRpos(1)+0.5*f_a_br(1)+0.5*f_r_br(1);
            fl_des<<  eeFLpos(0)+0.5*f_a_fl(0)+0.5*f_r_fl(0),  eeFLpos(1)+0.5*f_a_fl(1)+0.5*f_r_fl(1);
            fr_des<<  eeFRpos(0)+0.5*f_a_fr(0)+0.5*f_r_fr(0),  eeFRpos(1)+0.5*f_a_fr(1)+0.5*f_r_fr(1);
        }
        else {
            bl_des<<  eeBLpos(0)+0.5*f_a_bl(0),  eeBLpos(1)+0.5*f_a_bl(1);
            br_des<<  eeBRpos(0)+0.5*f_a_br(0),  eeBRpos(1)+0.5*f_a_br(1);
            fl_des<<  eeFLpos(0)+0.5*f_a_fl(0),  eeFLpos(1)+0.5*f_a_fl(1);
            fr_des<<  eeFRpos(0)+0.5*f_a_fr(0),  eeFRpos(1)+0.5*f_a_fr(1);
        }
        com_des<<(bl_des+br_des+fl_des+fr_des)/4;

        //Eigen::Vector2d curr_com;
        curr_com<<toEigen(CoM)[0],toEigen(CoM)[1];

        //std_msgs::Float32 norm_vel;

        //Eigen::Vector2d 
        dcom=com_des-curr_com;
        norm_vel.data = dcom.norm();


        _cmd_vel_pub.publish( norm_vel );


        CoMPosDes << saturate_xstep(com_des(0)),saturate_ystep(com_des(1)), 0.38, 
                          toEigen(des_com_pos)[3], toEigen(des_com_pos)[4], 0.0;  

        
         
        if(crawl)
        {
            if(step_num==3)
            {
                gait_type=6;
            }
            else
            {
                gait_type=5;
            }
        time_traj=1;
        }

        get_trajectory( toEigen( CoM), toEigen(CoM_vel), CoMPosDes, eeBLpos, eeBRpos, eeFLpos, eeFRpos, gait_type ,time_traj, solution2, formulation2, Mcom(3,3),Mcom(4,4) ,Mcom(5,5) ,Mcom(3,4), Mcom(3,5), Mcom(4,5) );

        h_bl_prev=0;
        h_br_prev=0;
        h_fl_prev=0;
        h_fr_prev=0;
        h_bl_min=100;
        h_br_min=100;
        h_fr_min=100;
        h_fl_min=100;

        period_st=0;
        period_tot=0;

    
        unpauseGazebo.call(pauseSrv); 
        begin=ros::Time::now();
        time_prev= begin;

        while((ros::Time::now()-begin).toSec() <   formulation2.params_.ee_phase_durations_.at(0)[0] && (ros::Time::now()-begin).toSec() <   formulation2.params_.ee_phase_durations_.at(2)[0] ) {

            phase_flag=0;
            // Taking Jacobian for CoM and joints
            Eigen::Matrix<double, 12, 6> Jstcom= toEigen(JacCOM_lin).block(0,0,12,6);

            
            Eigen::Matrix<double, 12, 12> Jstj= toEigen(JacCOM_lin).block(0,6,12,12);

            Eigen::Matrix<double, 12, 18> Jst= toEigen(JacCOM_lin);

            // cost function quadratic matrix
            Eigen::Matrix<double,6,30>  T_s= Jstcom.transpose()*Sigma_st;
            Eigen::Matrix<double,6,6> eigenQ1= 50*Eigen::Matrix<double,6,6>::Identity();

            Eigen::Matrix<double,30,30> eigenQ2= T_s.transpose()*eigenQ1*T_s;
            Eigen::Matrix<double,30,30> eigenQ= eigenQ2+Eigen::Matrix<double,30,30>::Identity();

            // Compute deltax, deltav
            double t = (ros::Time::now()-begin).toSec();
            Eigen::Matrix<double,6,1> CoMPosD;
            CoMPosD << solution2.base_linear_->GetPoint(t).p(), solution2.base_angular_->GetPoint(t).p();
            Eigen::Matrix<double,6,1> CoMVelD;
            CoMVelD << solution2.base_linear_->GetPoint(t).v(), solution2.base_angular_->GetPoint(t).v();
            Eigen::Matrix<double,6,1> CoMAccD;
            CoMAccD << solution2.base_linear_->GetPoint(t).a(), solution2.base_angular_->GetPoint(t).a();
            
            Eigen::Matrix<double,6,1> deltax = CoMPosD - toEigen( CoM );        
            deltax.block(3,0,3,1)<<_world_H_base.block(0,0,3,3)*deltax.block(3,0,3,1); 
            //cout << "Errore di posizione: " << deltax.transpose() << endl;
            //cout << "ComPosD: " << CoMPosD << endl;
            //cout << "CoM: " << toEigen( CoM ) << endl;
            Eigen::Matrix<double,6,1> deltav = CoMVelD-toEigen(CoM_vel);
            Eigen::MatrixXd g_acc = Eigen::MatrixXd::Zero(6,1);
            g_acc(2,0)=9.81;
            
            Eigen::MatrixXd M_com = toEigen(MassMatrixCOM).block(0,0,6,6);        
            Eigen::MatrixXd Kcom=3000*Eigen::MatrixXd::Identity(6,6);
            Eigen::MatrixXd Dcom=50*Eigen::MatrixXd::Identity(6,6);
            
            Eigen::Matrix<double,3,3> Tbr=getBRworldtransform();
            Eigen::Matrix<double,3,3> Tbl=getBLworldtransform();
            Eigen::Matrix<double,3,3> Tfl=getFLworldtransform();
            Eigen::Matrix<double,3,3> Tfr=getFRworldtransform();
            Fgrf<< Tbr*force_br, Tbl*force_bl,Tfl*force_fl,Tfr*force_fr;

            Eigen::Vector3d f_bl=Tbl*force_bl;
            Eigen::Vector3d f_br=Tbr*force_br;
            Eigen::Vector3d f_fl=Tfl*force_fl;
            Eigen::Vector3d f_fr=Tfr*force_fr;
       double mu=0.5;
            if(0.1<rob_foot_fl<0.25 || 0.1<rob_foot_fr<0.25 || 0.1<rob_foot_bl<0.25 || 0.1<rob_foot_br<0.25)
           {  mu=0.5;
           }
            double theta= atan(0.5);    
            double alfa_f_bl=acos(f_bl(2)/f_bl.norm());
            double alfa_f_br=acos(f_br(2)/f_br.norm());
            double alfa_f_fl=acos(f_fl(2)/f_fl.norm());
            double alfa_f_fr=acos(f_fr(2)/f_fr.norm());

//            double h_bl= 1/((theta-alfa_f_bl)*(theta+alfa_f_bl));
//            double h_br= 1/((theta-alfa_f_br)*(theta+alfa_f_br));
//            double h_fl= 1/((theta-alfa_f_fl)*(theta+alfa_f_fl));
//            double h_fr= 1/((theta-alfa_f_fr)*(theta+alfa_f_fr));

            double h_bl= 1/(theta-alfa_f_bl);
            double h_br= 1/(theta-alfa_f_br);
            double h_fl= 1/(theta-alfa_f_fl);
            double h_fr= 1/(theta-alfa_f_fr);


            double _dt=(ros::Time::now()-time_prev).toSec();
            time_prev=ros::Time::now();
            if(!isnan(h_bl) && (1/h_bl)>0.01){
            h_bl_prev += _dt*1/h_bl;
               if((1/h_bl)<h_bl_min)
               {
                   h_bl_min=1/h_bl;
               }
            }
        
                
            if(!isnan(h_br) && (1/h_br)>0.01){
            h_br_prev += _dt*1/h_br;
             if((1/h_br)<h_br_min)
               {
                   h_br_min=1/h_br;}
            }

            if(!isnan(h_fl) && (1/h_fl)>0.01){
            h_fl_prev += _dt*1/h_fl;
            if((1/h_fl)<h_fl_min)
               {
                   h_fl_min=1/h_fl;}
            }

            if(!isnan(h_fr) && (1/h_fr)>0.01){
            h_fr_prev += _dt*1/h_fr;
            if((1/h_fr)<h_fr_min)
               {
                   h_fr_min=1/h_fr;
               }
            }



            //std::cout<<"h_bl"<<1/h_bl<<std::endl;
            // estimate();

            // Compute Desired vector
            Eigen::Matrix<double,6,1> Wcom_des = Kcom*deltax+Dcom*deltav+robot_mass*g_acc+toEigen(MassMatrixCOM).block(0,0,6,6)*CoMAccD;
            Eigen::Matrix<double,30,1> eigenc = -T_s.transpose()*eigenQ1.transpose()*Wcom_des;

            _o->setQ( eigenQ );
            _o->setc( eigenc );


            //Equality constraints
            Eigen::Matrix<double,18, 30> eigenA= Eigen::Matrix<double,18,30>::Zero();
            eigenA.block(0,0,6,6)=toEigen(MassMatrixCOM).block(0,0,6,6);
            eigenA.block(0,18,6,12)=-Jstcom.transpose();
            eigenA.block(6,0,12,6)=Jstcom;
            eigenA.block(6,6,12,12)=Jstj;

            // Known term
            Eigen::Matrix<double,18, 1> eigenb= Eigen::Matrix<double,18,1>::Zero();
            eigenb.block(0,0,6,1)=-toEigen(BiasCOM).block(0,0,6,1);
            eigenb.block(6,0,12,1)=-toEigen(JdqdCOM_lin);
        
            //Inequality Constraints
            Eigen::Matrix<double,68, 30> eigenD= Eigen::Matrix<double,68,30>::Zero();
        
            // Torque limits
            eigenD.block(20,6,12,12)=toEigen(MassMatrixCOM).block(6,6,12,12);
            eigenD.block(20,18,12,12)=-Jstj.transpose();
            eigenD.block(32,6,12,12)=-toEigen(MassMatrixCOM).block(6,6,12,12);
            eigenD.block(32,18,12,12)=Jstj.transpose();
            eigenD.block(44,6,12,12)=Eigen::Matrix<double,12,12>::Identity();
            eigenD.block(56,6,12,12)=-Eigen::Matrix<double,12,12>::Identity();
        
            //Friction
           
            Eigen::Matrix<double,3, 1> n= Eigen::Matrix<double,3,1>::Zero();
            n<< 0, 0, 1;

            Eigen::Matrix<double,3, 1> t1= Eigen::Matrix<double,3,1>::Zero();
            t1<< 1, 0, 0;

            Eigen::Matrix<double,3, 1> t2= Eigen::Matrix<double,3,1>::Zero();
            t2<<0, 1, 0;

            Eigen::Matrix<double,5,3> cfr=Eigen::Matrix<double,5,3>::Zero();
        
            cfr<<(-mu*n+t1).transpose(),
                    (-mu*n+t2).transpose(),
                    -(mu*n+t1).transpose(),
                    -(mu*n+t2).transpose(),
                    -n.transpose();
            
            Eigen::Matrix<double,20,12> Dfr=Eigen::Matrix<double,20,12>::Zero();

            for(int i=0; i<4; i++)
            {
                Dfr.block(0+5*i,0+3*i,5,3)=cfr;
            }
            

            eigenD.block(0,18,20,12)=Dfr;

            // Known terms for inequality
            Eigen::Matrix<double,68, 1> eigenC= Eigen::Matrix<double,68,1>::Zero();
        
            // Torque limits
            Eigen::Matrix<double,12,1> tau_max=60*Eigen::Matrix<double,12,1>::Ones();
            Eigen::Matrix<double,12,1> tau_min=-60*Eigen::Matrix<double,12,1>::Ones();
            Eigen::Matrix<double,12, 1> eigenBiascom=toEigen(BiasCOM).block(6,0,12,1);

            eigenC.block(20,0,12,1)=tau_max-eigenBiascom;
            eigenC.block(32,0,12,1)=-(tau_min-eigenBiascom);
        
            // Joints limits
            double deltat=0.025;
            Eigen::Matrix<double,12, 1> eigenq=toEigen(q).block(6,0,12,1);
            Eigen::Matrix<double,12, 1> eigendq=toEigen(dq).block(6,0,12,1);
            Eigen::Matrix<double,12, 1> eigenqmin=toEigen(qmin);
            Eigen::Matrix<double,12, 1> eigenqmax=toEigen(qmax);
            Eigen::Matrix<double,12, 1> ddqmin=(2/pow(deltat,2))*(eigenqmin-eigenq-deltat*eigendq);
            Eigen::Matrix<double,12, 1> ddqmax=(2/pow(deltat,2))*(eigenqmax-eigenq-deltat*eigendq);

            eigenC.block(44,0,12,1)=ddqmax;
            eigenC.block(56,0,12,1)=-ddqmin;
        

            Eigen::Matrix<double,18,18> Si;
            Si<<Eigen::Matrix<double,6,18>::Zero(),
                Eigen::Matrix<double,12,6>::Zero(),Eigen::Matrix<double,12,12>::Identity();

        
            //Linear constraints matrix
            Eigen::Matrix<double,86, 31> eigenL= Eigen::Matrix<double,86,31>::Zero();

            eigenL<< eigenA,eigenb,
                        eigenD, eigenC;


            ////////////////////////////////            
            
            myQP = QP_SETUP_dense(30, 68, 18, eigenQ.data(), eigenA.data(), eigenD.data(), eigenc.data(), eigenC.data(), eigenb.data(), NULL, COLUMN_MAJOR_ORDERING); 
            //myQP->options->maxit  = 30;
            myQP->options->reltol = 1e-2 ;
            myQP->options->abstol  = 1e-2;

            
            qp_int ExitCode = QP_SOLVE(myQP);


            Eigen::VectorXd x_;
            x_.resize( 30 );

               for( int i=0; i<30; i++ ) {
            x_(i) = myQP->x[i];
        }
           
        //////////////////////////////////////////

            //_o->opt_stance( x_ );
            
            
            tau=toEigen(MassMatrixCOM).block(6,6,12,12)*x_.block(6,0,12,1)+eigenBiascom-Jstj.transpose()*x_.block(18,0,12,1);
            publish_cmd( tau );

            estimation_msg.header.stamp = ros::Time::now();

            estimation_msg.wrench.force.x = w[0][0];
            estimation_msg.wrench.force.y =w[0][1];
            estimation_msg.wrench.force.z = w[0][2];


            estimation_msg.wrench.torque.x = w[0][3];
            estimation_msg.wrench.torque.y = w[0][4];
            estimation_msg.wrench.torque.z = w[0][5];


                        com_vel_p.header.stamp = ros::Time::now();
            com_vel_p.point.x=toEigen(CoM_vel)[0];
            com_vel_p.point.y=toEigen(CoM_vel)[1];
            com_vel_p.point.z=toEigen(CoM_vel)[2];
            com_vel_pub.publish(com_vel_p);

            estimation_pub.publish(estimation_msg);

            r.sleep();
        }
         
        period_st= period_st+(ros::Time::now()-begin).toSec();


        flag_exit=false;

        if(gait_type ==5){
        while((ros::Time::now()-begin).toSec() <  formulation2.params_.ee_phase_durations_.at(0)[1]+formulation2.params_.ee_phase_durations_.at(0)[0] &&  flag_exit == false ) 
        {
         double duration= formulation2.params_.ee_phase_durations_.at(0)[1]+formulation2.params_.ee_phase_durations_.at(0)[0];
        qpproblemcrawl(2, solution2, duration);
            }
        }
        
        if(gait_type ==6){
        while((ros::Time::now()-begin).toSec() <  formulation2.params_.ee_phase_durations_.at(2)[1]+formulation2.params_.ee_phase_durations_.at(2)[0] &&  flag_exit == false ) 
        {
         double duration= formulation2.params_.ee_phase_durations_.at(2)[1]+formulation2.params_.ee_phase_durations_.at(2)[0];
        qpproblemcrawl(1, solution2, duration);
            }
        }

        if(gait_type==2) {
            while((ros::Time::now()-begin).toSec() <  formulation2.params_.ee_phase_durations_.at(0)[0]+formulation2.params_.ee_phase_durations_.at(0)[1] &  flag_exit == false) {
            
                phase_flag=2;
                update_mutex.lock();
                //update(_world_H_base, _jnt_pos, _jnt_vel, _base_vel, gravity);
                update_mutex.unlock();
                double duration=formulation2.params_.ee_phase_durations_.at(0)[0]+formulation2.params_.ee_phase_durations_.at(0)[1] ;
                // Taking Jacobian for CoM and joints
                int swl1, swl2, stl1, stl2;
                swl1=3;
                swl2=9 ;
                stl1=0;
                stl2=6 ;
                Eigen::Matrix<double, 6, 18> Jst= Eigen::Matrix<double,6,18>::Zero();
                Jst.block(0,0,3,18)=toEigen(JacCOM_lin).block(stl1,0,3,18);
                Jst.block(3,0,3,18)=toEigen(JacCOM_lin).block(stl2,0,3,18);

                Eigen::Matrix<double, 6, 18> Jsw= Eigen::Matrix<double,6,18>::Zero();
                Jsw.block(0,0,3,18)=toEigen(JacCOM_lin).block(swl1,0,3,18);
                Jsw.block(3,0,3,18)=toEigen(JacCOM_lin).block(swl2,0,3,18);

                // cost function quadratic matrix
                Eigen::Matrix<double,6,30>  T_s= Jst.block(0,0,6,6).transpose()*Sigma_sw;
        
                Eigen::Matrix<double,6,6> eigenQ1= 50*Eigen::Matrix<double,6,6>::Identity();
                Eigen::Matrix<double,30,30> eigenQ2= T_s.transpose()*eigenQ1*T_s;
                Eigen::Matrix<double,30,30> eigenR= Eigen::Matrix<double,30,30>::Identity();
                
                eigenR.block(24,24,6,6)=100000000*Eigen::Matrix<double,6,6>::Identity();
                
                Eigen::Matrix<double,30,30> eigenQ= eigenQ2+eigenR;

                // Compute deltax, deltav
                double t = (ros::Time::now()-begin).toSec();
                Eigen::Matrix<double,6,1> CoMPosD;
                CoMPosD << solution2.base_linear_->GetPoint(t).p(), solution2.base_angular_->GetPoint(t).p();
                Eigen::Matrix<double,6,1> CoMVelD;
                CoMVelD << solution2.base_linear_->GetPoint(t).v(), solution2.base_angular_->GetPoint(t).v();
                Eigen::Matrix<double,6,1> CoMAccD;
                CoMAccD << solution2.base_linear_->GetPoint(t).a(), solution2.base_angular_->GetPoint(t).a();
                
                Eigen::Matrix<double,6,1> deltax = CoMPosD - toEigen( CoM );      
                deltax.block(3,0,3,1)<<_world_H_base.block(0,0,3,3)*deltax.block(3,0,3,1);  
                //cout << "Errore di posizione: " << deltax.transpose() << endl;
                //cout << "ComPosD: " << CoMPosD << endl;
                //cout << "CoM: " << toEigen( CoM ) << endl;

                Eigen::Matrix<double,6,1> deltav = CoMVelD-toEigen(CoM_vel);
                Eigen::MatrixXd g_acc = Eigen::MatrixXd::Zero(6,1);
                g_acc(2,0)=9.81;
                
                Eigen::MatrixXd M_com = toEigen(MassMatrixCOM).block(0,0,6,6);        
                Eigen::MatrixXd Kcom=3000*Eigen::MatrixXd::Identity(6,6);
                Eigen::MatrixXd Dcom=50*Eigen::MatrixXd::Identity(6,6);
                
                Eigen::Matrix<double,3,3> Tbr=getBRworldtransform();
                Eigen::Matrix<double,3,3> Tbl=getBLworldtransform();
                Eigen::Matrix<double,3,3> Tfl=getFLworldtransform();
                Eigen::Matrix<double,3,3> Tfr=getFRworldtransform();

                Fgrf<<Tbr*force_br, Eigen::Matrix<double,3,1>::Zero(),  Tfl*force_fl, Eigen::Matrix<double,3,1>::Zero();
                
                // estimate();
                
                // Compute Desired vector
                Eigen::Matrix<double,6,1> Wcom_des = Kcom*deltax+Dcom*deltav+robot_mass*g_acc+toEigen(MassMatrixCOM).block(0,0,6,6)*CoMAccD;
                Eigen::Matrix<double,30,1> eigenc = -T_s.transpose()*eigenQ1.transpose()*Wcom_des;
        
                _o->setQ( eigenQ );
                _o->setc( eigenc );
        
        
                //Equality constraints
                Eigen::Matrix<double,12, 30> eigenA= Eigen::Matrix<double,12,30>::Zero();
                eigenA.block(0,0,6,6)=toEigen(MassMatrixCOM).block(0,0,6,6);
                eigenA.block(0,18,6,6)=-Jst.block(0,0,6,6).transpose();
                eigenA.block(6,0,6,6)=Jst.block(0,0,6,6);
                eigenA.block(6,6,6,12)=Jst.block(0,6,6,12);
        
                // Known term
                Eigen::Matrix<double,12, 1> eigenb= Eigen::Matrix<double,12,1>::Zero();
                Eigen::Matrix<double,6,1> Jdqdst= Eigen::Matrix<double,6,1>::Zero();
                Jdqdst<<toEigen(JdqdCOM_lin).block(stl1,0,3,1),
                        toEigen(JdqdCOM_lin).block(stl2,0,3,1);
            
                //Inequality Constraints
                Eigen::Matrix<double,70,30> eigenD= Eigen::Matrix<double,70,30>::Zero();
        
                // Torque limits
                eigenD.block(10,6,12,12)=toEigen(MassMatrixCOM).block(6,6,12,12);
                eigenD.block(10,18,12,6)=-Jst.block(0,6,6,12).transpose();
                eigenD.block(22,6,12,12)=-toEigen(MassMatrixCOM).block(6,6,12,12);
                eigenD.block(22,18,12,6)=Jst.block(0,6,6,12).transpose();
                eigenD.block(34,0,3,6)=Jsw.block(0,0,3,6);
                eigenD.block(34,6,3,12)=Jsw.block(0,6,3,12);
                eigenD.block(37,0,3,6)=Jsw.block(3,0,3,6);
                eigenD.block(37,6,3,12)=Jsw.block(3,6,3,12);
                eigenD.block(34,24,3,3)=-Eigen::Matrix<double,3,3>::Identity();
                eigenD.block(37,27,3,3)=-Eigen::Matrix<double,3,3>::Identity();
                eigenD.block(40,0,3,6)=-Jsw.block(0,0,3,6);
                eigenD.block(40,6,3,12)=-Jsw.block(0,6,3,12);
                eigenD.block(43,0,3,6)=-Jsw.block(3,0,3,6);
                eigenD.block(43,6,3,12)=-Jsw.block(3,6,3,12);
                eigenD.block(40,24,3,3)=-Eigen::Matrix<double,3,3>::Identity();
                eigenD.block(43,27,3,3)=-Eigen::Matrix<double,3,3>::Identity();
                eigenD.block(46,6,12,12)=Eigen::Matrix<double,12,12>::Identity();
                eigenD.block(58,6,12,12)=-Eigen::Matrix<double,12,12>::Identity();
            
                //Friction
         double mu=0.5;
            if(0.1<rob_foot_fl<0.25 || 0.1<rob_foot_fr<0.25 || 0.1<rob_foot_bl<0.25 || 0.1<rob_foot_br<0.25)
           {  mu=0.5;
           }
                Eigen::Matrix<double,3, 1> n= Eigen::Matrix<double,3,1>::Zero();
                n<< 0, 
                    0,
                    1;
        
                Eigen::Matrix<double,3, 1> t1= Eigen::Matrix<double,3,1>::Zero();
                t1<< 1,
                    0,
                    0;
        
                Eigen::Matrix<double,3, 1> t2= Eigen::Matrix<double,3,1>::Zero();
                t2<<0,
                    1,
                    0;
        
                Eigen::Matrix<double,5,3> cfr=Eigen::Matrix<double,5,3>::Zero();
        
                cfr << (-mu*n+t1).transpose(),
                            (-mu*n+t2).transpose(),
                            -(mu*n+t1).transpose(),
                            -(mu*n+t2).transpose(),
                            -n.transpose();
                        
                Eigen::Matrix<double,10,6> Dfr=Eigen::Matrix<double,10,6>::Zero();
                for(int i=0; i<2; i++) {
                    Dfr.block(0+5*i,0+3*i,5,3)=cfr;
                }
                eigenD.block(0,18,10,6)=Dfr;
                // Known terms for inequality
                Eigen::Matrix<double,70, 1> eigenC= Eigen::Matrix<double,70,1>::Zero();
            
                // Torque limits
                Eigen::Matrix<double,12,1> tau_max=60*Eigen::Matrix<double,12,1>::Ones();
                Eigen::Matrix<double,12,1> tau_min=-60*Eigen::Matrix<double,12,1>::Ones(); 
                Eigen::Matrix<double,12, 1> eigenBiascom=toEigen(BiasCOM).block(6,0,12,1);
                eigenC.block(10,0,12,1)=tau_max-eigenBiascom;
                eigenC.block(22,0,12,1)=-(tau_min-eigenBiascom);
            
                // Joints limits
                double deltat=0.025;
                Eigen::Matrix<double,12, 1> eigenq=toEigen(q).block(6,0,12,1);
                Eigen::Matrix<double,12, 1> eigendq=toEigen(dq).block(6,0,12,1);
                Eigen::Matrix<double,12, 1> eigenqmin=toEigen(qmin);
                Eigen::Matrix<double,12, 1> eigenqmax=toEigen(qmax);
                Eigen::Matrix<double,12, 1> ddqmin=(2/pow(deltat,2))*(eigenqmin-eigenq-deltat*eigendq);
                Eigen::Matrix<double,12, 1> ddqmax=(2/pow(deltat,2))*(eigenqmax-eigenq-deltat*eigendq);
        
                eigenC.block(46,0,12,1)=ddqmax;
                eigenC.block(58,0,12,1)=-ddqmin;
                
                
                Eigen::Matrix<double,6,1> Jdqdsw= Eigen::Matrix<double,6,1>::Zero();
                Jdqdsw<<toEigen(JdqdCOM_lin).block(swl1,0,3,1),toEigen(JdqdCOM_lin).block(swl2,0,3,1);    
                Eigen::Matrix<double,6,1> accdes;
        
                accdes<< solution2.ee_motion_.at(0)->GetPoint(t).a(), solution2.ee_motion_.at(3)->GetPoint(t).a();
                World_bl = kinDynComp.getWorldTransform("back_left_foot");
                Eigen::MatrixXd eeBLposM = toEigen(World_bl.getPosition());
                Eigen::Vector3d eeBLpos;
                eeBLpos << eeBLposM.block(0,0,3,1);
        
                World_bl = kinDynComp.getWorldTransform("front_right_foot");
                Eigen::MatrixXd eeFRposM = toEigen(World_bl.getPosition());
                Eigen::Vector3d eeFRpos;
                eeFRpos << eeFRposM.block(0,0,3,1);

                iDynTree::Twist br_vel;
                br_vel=kinDynComp.getFrameVel("back_left_foot");
                Eigen::MatrixXd eeBLvelM = toEigen(br_vel.getLinearVec3() );
                Eigen::Vector3d eeBLvel;
                eeBLvel << eeBLvelM.block(0,0,3,1);

                iDynTree::Twist fl_vel;
                fl_vel=kinDynComp.getFrameVel("front_right_foot");
                Eigen::MatrixXd eeFRvelM = toEigen(fl_vel.getLinearVec3() );
                Eigen::Vector3d eeFRvel;
                eeFRvel << eeFRvelM.block(0,0,3,1);

        
                Eigen::Matrix<double,6,1> posdelta;
                posdelta<< solution2.ee_motion_.at(0)->GetPoint(t).p()-eeBLpos,
                        solution2.ee_motion_.at(3)->GetPoint(t).p()-eeFRpos;
        
                Eigen::Matrix<double,6,1> veldelta;
                veldelta<< solution2.ee_motion_.at(0)->GetPoint(t).v()-eeBLvel,
                        solution2.ee_motion_.at(3)->GetPoint(t).v()-eeFRvel;
                
                Eigen::MatrixXd Kp;
                Kp=300*Eigen::MatrixXd::Identity(6,6);
                Eigen::MatrixXd Kd;
                Kd=20*Eigen::MatrixXd::Identity(6,6);       
                Eigen::Matrix<double,6,1> vdotswdes=accdes+Kd*veldelta+Kp*posdelta;
        
                eigenC.block(34,0,6,1)= vdotswdes-Jdqdsw;
                eigenC.block(40,0,6,1)= -vdotswdes+Jdqdsw;
            
                //Linear constraints matrix
                Eigen::Matrix<double,82, 31> eigenL= Eigen::Matrix<double,82,31>::Zero();
            
                eigenL<< eigenA,eigenb,
                            eigenD, eigenC;


            //////////////////////////////////////////////            
            
            myQP = QP_SETUP_dense(30, 70, 12, eigenQ.data(), eigenA.data(), eigenD.data(), eigenc.data(), eigenC.data(), eigenb.data(), NULL, COLUMN_MAJOR_ORDERING); 
            //myQP->options->maxit  = 30;
            myQP->options->reltol = 1e-2 ;
            myQP->options->abstol  = 1e-2;

            
            qp_int ExitCode = QP_SOLVE(myQP);
            
                Eigen::VectorXd x_;
                x_.resize( 30 );

        for( int i=0; i<30; i++ ) {
            x_(i) = myQP->x[i];
        }
                    
            Eigen::VectorXd tau= Eigen::VectorXd::Zero(12);
            tau=toEigen(MassMatrixCOM).block(6,6,12,12)*x_.block(6,0,12,1)+eigenBiascom-Jst.block(0,6,6,12).transpose()*x_.block(18,0,6,1);
            publish_cmd( tau );   
            estimation_msg.header.stamp = ros::Time::now();
            estimation_msg.wrench.force.x = w[0][0];
            estimation_msg.wrench.force.y =w[0][1];
            estimation_msg.wrench.force.z = w[0][2];
                    
            estimation_msg.wrench.torque.x = w[0][3];
            estimation_msg.wrench.torque.y = w[0][4];
            estimation_msg.wrench.torque.z = w[0][5];


            com_vel_p.header.stamp = ros::Time::now();
            com_vel_p.point.x=toEigen(CoM_vel)[0];
            com_vel_p.point.y=toEigen(CoM_vel)[1];
            com_vel_p.point.z=toEigen(CoM_vel)[2];
            com_vel_pub.publish(com_vel_p);
                
                estimation_pub.publish(estimation_msg);

            
                r.sleep();
            
                if( (contact_fr==true || contact_bl==true) && t>duration-0.05) {
                    flag_exit=true;
                }       
            } 
        }
        
        step_num+=1;
        period_tot=period_tot+(ros::Time::now()-begin).toSec();
    }
}

double DOGCTRL::compute_fr(double v)
{
    if (abs(v)<0.07)
    {return 0;}
    else{
        return abs(v);
    }


}

double DOGCTRL::saturate_x(double v)
{
    if(abs(v)>2)
    { 
      return copysign(2,v);
    }
    else{
        return v;
     }
}

double DOGCTRL::saturate_xstep(double v)
{
    double step_x=toEigen(CoM)[0]-v;
    if(abs(step_x)>0.06)
    { 
      return toEigen(CoM)[0]-copysign(0.06,step_x);
    }
    else{
        return v;
     }
}

double DOGCTRL::saturate_ystep(double v)
{
    double step_y=toEigen(CoM)[1]-v;
    if(abs(step_y)>0.06)
    { 
      return toEigen(CoM)[1]-copysign(0.06,step_y);
    }
    else{
        return v;
     }
}

double DOGCTRL::saturate_y(double v)
{
    if(abs(v)>2)
    { 
      return copysign(2,v);
    }
    else{
        return v;
    }
}


void DOGCTRL::compute_Kpa(Eigen::Vector2d e_a)
{
    if(abs(e_a(0))<0.4)
        {
             if(fake_crawl)
            { K_pa(0,0)=0.01;}
            else{
              K_pa(0,0)=0.3;
            }
        }
        else{
        if(MIN_EXIT){
            K_pa(0,0)=0.1;
        }
        else{
            if(fake_crawl)
            { K_pa(0,0)=0.01;}
            else{
              K_pa(0,0)=0.3;
            }
        }
        }
        if(abs(e_a(1))<0.4)
        {    if(fake_crawl)
            { K_pa(1,1)=0.01;}
            else{
              K_pa(1,1)=0.4;
            }
         // K_pa(1,1)=0.3; // 0.4 case 2
        }
        else{
        if(MIN_EXIT)
            K_pa(1,1)=0.2; // 0.2 case 2
        else{
              if(fake_crawl)
            { K_pa(1,1)=0.01;}
            else{
              K_pa(1,1)=0.4;
            }
           // K_pa(1,1)=0.1;
          }
        }
}

void DOGCTRL::run() {
    ros::ServiceClient set_model_configuration_srv = _nh.serviceClient<gazebo_msgs::SetModelConfiguration>("/gazebo/set_model_configuration");
    ros::ServiceClient set_model_state_srv = _nh.serviceClient<gazebo_msgs::SetModelState>("/gazebo/set_model_state");
    
    gazebo_msgs::SetModelConfiguration robot_init_config;
    robot_init_config.request.model_name = "dogbot";
    robot_init_config.request.urdf_param_name = "robot_description";
    robot_init_config.request.joint_names.push_back("back_left_roll_joint");
    robot_init_config.request.joint_names.push_back("back_left_pitch_joint");
    robot_init_config.request.joint_names.push_back("back_left_knee_joint");
    robot_init_config.request.joint_names.push_back("back_right_roll_joint");
    robot_init_config.request.joint_names.push_back("back_right_pitch_joint");
    robot_init_config.request.joint_names.push_back("back_right_knee_joint");
    robot_init_config.request.joint_names.push_back("front_left_roll_joint");
    robot_init_config.request.joint_names.push_back("front_left_pitch_joint");
    robot_init_config.request.joint_names.push_back("front_left_knee_joint");
    robot_init_config.request.joint_names.push_back("front_right_roll_joint");
    robot_init_config.request.joint_names.push_back("front_right_pitch_joint");
    robot_init_config.request.joint_names.push_back("front_right_knee_joint");
    robot_init_config.request.joint_positions.push_back( 0.0004875394147498824);
    robot_init_config.request.joint_positions.push_back( -0.884249947977489);
    robot_init_config.request.joint_positions.push_back(-1.6039026405138666);
    robot_init_config.request.joint_positions.push_back( 0.0006243098169198547);
    robot_init_config.request.joint_positions.push_back(0.8861978063639038);
    robot_init_config.request.joint_positions.push_back(1.6032646991719783);
    robot_init_config.request.joint_positions.push_back(-3.197670677312914e-05);
    robot_init_config.request.joint_positions.push_back(-0.8848124990461947);
    robot_init_config.request.joint_positions.push_back(-1.6039627256817717);
    robot_init_config.request.joint_positions.push_back(-0.0005127385581351618);
    robot_init_config.request.joint_positions.push_back(0.886353788084274);
    robot_init_config.request.joint_positions.push_back( 1.60361055049274);

    if(set_model_configuration_srv.call(robot_init_config))
        ROS_INFO("Robot configuration set.");
    else
        ROS_INFO("Failed to set robot configuration.");



    while (!_first_jpos) {
        gazebo_msgs::SetModelState robot_init_state;
        robot_init_state.request.model_state.model_name = "dogbot";
        robot_init_state.request.model_state.reference_frame = "world";
        robot_init_state.request.model_state.pose.position.x = 0.0;
        robot_init_state.request.model_state.pose.position.y =0;
        robot_init_state.request.model_state.pose.position.z = 0.430159040502;
        robot_init_state.request.model_state.pose.orientation.x=0;
        robot_init_state.request.model_state.pose.orientation.y=0;
        robot_init_state.request.model_state.pose.orientation.z=0;
        robot_init_state.request.model_state.pose.orientation.w=1;
        if(set_model_state_srv.call(robot_init_state))
            ROS_INFO("Robot state set.");
        else
            ROS_INFO("Failed to set robot state.");

        ros::spinOnce();
        usleep(0.1*1e6);
    }


    boost::thread ctrl_loop_t( &DOGCTRL::ctrl_loop, this );
    boost::thread update_loop_t( &DOGCTRL::update_loop, this );
    //boost::thread estimate_loop_t( &DOGCTRL::estimate_loop, this );


    ros::spin();
}

void DOGCTRL::qpproblemcrawl(   int swingleg, towr::SplineHolder solution,  double duration)
{ 
	int swl1, stl1, stl2, stl3;
    switch(swingleg){
		case 0: swl1=0; //BR
		stl1=3;
		stl2=6 ; 
		stl3=9;
		break;
		case 1: swl1=6; //FL
		stl1=0;
		stl2=3 ; 
		stl3=9;
		break;
		case 2: swl1=3; // BL
		stl1=0;
		stl2=6; 
		stl3=9;
		break;
		case 3: swl1=9; //FR
		stl1=0;
		stl2=3; 
		stl3=6;
		 break;
	}


   // Taking Jacobian for CoM and joints
   Eigen::Matrix<double, 9, 6> Jstcom= Eigen::Matrix<double,9,6>::Zero();
    Jstcom.block(0,0,3,6)= toEigen(JacCOM_lin).block(stl1,0,3,6);
	Jstcom.block(3,0,3,6)= toEigen(JacCOM_lin).block(stl2,0,3,6);
    Jstcom.block(6,0,3,6)= toEigen(JacCOM_lin).block(stl3,0,3,6);

   Eigen::Matrix<double, 9, 12> Jstj= Eigen::Matrix<double,9,12>::Zero();
    Jstj.block(0,0,3,12)=toEigen(JacCOM_lin).block(stl1,6,3,12);
    Jstj.block(3,0,3,12)=toEigen(JacCOM_lin).block(stl2,6,3,12);
    Jstj.block(6,0,3,12)=toEigen(JacCOM_lin).block(stl3,6,3,12);

    Eigen::Matrix<double, 9, 18> Jst= Eigen::Matrix<double,9,18>::Zero();
    Jst.block(0,0,3,18)=toEigen(JacCOM_lin).block(stl1,0,3,18);
    Jst.block(3,0,3,18)=toEigen(JacCOM_lin).block(stl2,0,3,18);
	Jst.block(6,0,3,18)=toEigen(JacCOM_lin).block(stl3,0,3,18);


   Eigen::Matrix<double, 3, 6> Jswcom= Eigen::Matrix<double,3,6>::Zero();
    Jswcom.block(0,0,3,6)= toEigen(JacCOM_lin).block(swl1,0,3,6);
	
   Eigen::Matrix<double, 3, 12> Jswj=  Eigen::Matrix<double,3,12>::Zero();
   Jswj.block(0,0,3,12)=toEigen(JacCOM_lin).block(swl1,6,3,12);


   // cost function quadratic matrix
   Eigen::Matrix<double,9,30> Sigma= Eigen::Matrix<double,9,30>::Zero();
   Sigma.block(0,18,9,9)= Eigen::Matrix<double,9,9>::Identity();
   
   Eigen::Matrix<double,6,30>  T_s= Jstcom.transpose()*Sigma;

   Eigen::Matrix<double,6,6> eigenQ1= 50*Eigen::Matrix<double,6,6>::Identity();
   Eigen::Matrix<double,30,30> eigenQ2= T_s.transpose()*eigenQ1*T_s;
   Eigen::Matrix<double,30,30> eigenR= Eigen::Matrix<double,30,30>::Identity();
  
   eigenR.block(27,27,3,3)=10000*Eigen::Matrix<double,3,3>::Identity();
   
   Eigen::Matrix<double,30,30> eigenQ= eigenQ2+eigenR;
 
    // cost function linear matrix
    double t = (ros::Time::now()-begin).toSec();
    Eigen::Matrix<double,6,1> CoMPosD;
    CoMPosD << solution.base_linear_->GetPoint(t).p(), solution.base_angular_->GetPoint(t).p();
    Eigen::Matrix<double,6,1> CoMVelD;
    CoMVelD << solution.base_linear_->GetPoint(t).v(), solution.base_angular_->GetPoint(t).v();
    Eigen::Matrix<double,6,1> CoMAccD;
    CoMAccD << solution.base_linear_->GetPoint(t).a(), solution.base_angular_->GetPoint(t).a();

    Eigen::Matrix<double,6,1> deltax = CoMPosD - toEigen( CoM );      
    deltax.block(3,0,3,1)<<_world_H_base.block(0,0,3,3)*deltax.block(3,0,3,1);  
    Eigen::Matrix<double,6,1> deltav = CoMVelD-toEigen(CoM_vel);
    Eigen::MatrixXd g_acc = Eigen::MatrixXd::Zero(6,1);
    g_acc(2,0)=9.81;
    
    Eigen::MatrixXd M_com = toEigen(MassMatrixCOM).block(0,0,6,6);        
    Eigen::MatrixXd Kcom=3000*Eigen::MatrixXd::Identity(6,6);
    Eigen::MatrixXd Dcom=50*Eigen::MatrixXd::Identity(6,6);
    
    Eigen::Matrix<double,3,3> Tbr=getBRworldtransform();
    Eigen::Matrix<double,3,3> Tbl=getBLworldtransform();
    Eigen::Matrix<double,3,3> Tfl=getFLworldtransform();
    Eigen::Matrix<double,3,3> Tfr=getFRworldtransform();

    if(swl1==0)
    {Fgrf<<Eigen::Matrix<double,3,1>::Zero(), Tbl*force_bl,  Tfl*force_fl, Tfr*force_fr;}
    else if(swl1==6)
    {Fgrf<<Tbr*force_br, Tbl*force_bl,  Eigen::Matrix<double,3,1>::Zero(), Tfr*force_fr;}
    else if(swl1==3)
    {Fgrf<<Tbr*force_br, Eigen::Matrix<double,3,1>::Zero(),  Tfl*force_fl, Tfr*force_fr;}
    else if(swl1==9)
    {Fgrf<<Tbr*force_br, Tbl*force_bl,  Tfl*force_fl, Eigen::Matrix<double,3,1>::Zero();}
    
    // Compute Desired vector
    Eigen::Matrix<double,6,1> Wcom_des = Kcom*deltax+Dcom*deltav+robot_mass*g_acc+toEigen(MassMatrixCOM).block(0,0,6,6)*CoMAccD;
    Eigen::Matrix<double,30,1> eigenc= -T_s.transpose()*eigenQ1.transpose()*Wcom_des; 
    double mu=0.5;
    if(0.1<rob_foot_fl<0.25 || 0.1<rob_foot_fr<0.25 || 0.1<rob_foot_bl<0.25 || 0.1<rob_foot_br<0.25)
    {  mu=0.5;}

	//Equality constraints
	Eigen::Matrix<double,15, 30> eigenA= Eigen::Matrix<double,15,30>::Zero();
	
	eigenA.block(0,0,6,6)=toEigen(MassMatrixCOM).block(0,0,6,6);

	eigenA.block(0,18,6,9)=-Jstcom.transpose();

    eigenA.block(6,0,9,6)=Jstcom;

    eigenA.block(6,6,9,12)=Jstj;

    // Known term
    Eigen::Matrix<double,15, 1> eigenb= Eigen::Matrix<double,15,1>::Zero();

    Eigen::Matrix<double,9,1> Jdqdst= Eigen::Matrix<double,9,1>::Zero();
	Jdqdst<<toEigen(JdqdCOM_lin).block(stl1,0,3,1),
	        toEigen(JdqdCOM_lin).block(stl2,0,3,1),
			toEigen(JdqdCOM_lin).block(stl3,0,3,1);
			 
    Eigen::Matrix<double,1,6> grav;
	grav<<0,0,9.8,0,0,0;

	eigenb.block(0,0,6,1)=-toEigen(BiasCOM).block(0,0,6,1);

	eigenb.block(6,0,9,1)=-Jdqdst;    
    
    //Inequality Constraints

	Eigen::Matrix<double,69,30> eigenD= Eigen::Matrix<double,69,30>::Zero();
	
	// Torque limits
	eigenD.block(15,6,12,12)=toEigen(MassMatrixCOM).block(6,6,12,12);

    eigenD.block(15,18,12,9)=-Jstj.transpose();

	eigenD.block(27,6,12,12)=-toEigen(MassMatrixCOM).block(6,6,12,12);

    eigenD.block(27,18,12,9)=Jstj.transpose();
    
    eigenD.block(39,0,3,6)=Jswcom;

    eigenD.block(39,6,3,12)=Jswj;

	eigenD.block(39,27,3,3)=-Eigen::Matrix<double,3,3>::Identity();

	eigenD.block(42,0,3,6)=-Jswcom;

    eigenD.block(42,6,3,12)=-Jswj;

    eigenD.block(42,27,3,3)=-Eigen::Matrix<double,3,3>::Identity();

	eigenD.block(45,6,12,12)=Eigen::Matrix<double,12,12>::Identity();

    eigenD.block(57,6,12,12)=-Eigen::Matrix<double,12,12>::Identity();
    
	//Friction
	   Eigen::Matrix<double,3, 1> n= Eigen::Matrix<double,3,1>::Zero();
	   n<< 0,
	       0,
		   1;

	   Eigen::Matrix<double,3, 1> t1= Eigen::Matrix<double,3,1>::Zero();
	   t1<< 1,
	       0,
		   0;

       Eigen::Matrix<double,3, 1> t2= Eigen::Matrix<double,3,1>::Zero();
	   t2<<0,
	       1,
		   0;

	   Eigen::Matrix<double,5,3> cfr=Eigen::Matrix<double,5,3>::Zero();
  
	   cfr<<(-mu*n+t1).transpose(),
	        (-mu*n+t2).transpose(),
			-(mu*n+t1).transpose(),
			-(mu*n+t2).transpose(),
			-n.transpose();
     
	    Eigen::Matrix<double,15,9> Dfr=Eigen::Matrix<double,15,9>::Zero();

		for(int i=0; i<3; i++)
		{
			Dfr.block(0+5*i,0+3*i,5,3)=cfr;
		}
		

    eigenD.block(0,18,15,9)=Dfr;

    // Known terms for inequality
	Eigen::Matrix<double,69, 1> eigenC= Eigen::Matrix<double,69,1>::Zero();
	
	// Torque limits
    Eigen::Matrix<double,12,1> tau_max=60*Eigen::Matrix<double,12,1>::Ones();
	Eigen::Matrix<double,12,1> tau_min=-60*Eigen::Matrix<double,12,1>::Ones();

    Eigen::Matrix<double,12, 1> eigenBiascom=toEigen(BiasCOM).block(6,0,12,1);
	eigenC.block(15,0,12,1)=tau_max-eigenBiascom;
	eigenC.block(27,0,12,1)=-(tau_min-eigenBiascom);
    
      // Joints limits
    double deltat=0.025;
    Eigen::Matrix<double,12, 1> eigenq=toEigen(q).block(6,0,12,1);
	Eigen::Matrix<double,12, 1> eigendq=toEigen(dq).block(6,0,12,1);
	Eigen::Matrix<double,12, 1> eigenqmin=toEigen(qmin);
	Eigen::Matrix<double,12, 1> eigenqmax=toEigen(qmax);
	Eigen::Matrix<double,12, 1> ddqmin=(2/pow(deltat,2))*(eigenqmin-eigenq-deltat*eigendq);
	Eigen::Matrix<double,12, 1> ddqmax=(2/pow(deltat,2))*(eigenqmax-eigenq-deltat*eigendq);

    eigenC.block(45,0,12,1)=ddqmax;
	eigenC.block(57,0,12,1)=-ddqmin;

	Eigen::Matrix<double,3,1> Jdqdsw= Eigen::Matrix<double,3,1>::Zero();
	Jdqdsw<<toEigen(JdqdCOM_lin).block(swl1,0,3,1);

    
	Eigen::Matrix<double,3,1> fext_lambda= Eigen::Matrix<double,3,1>::Zero();

	Eigen::Matrix<double,18,18> Si;
	Si<<Eigen::Matrix<double,6,18>::Zero(),
	    Eigen::Matrix<double,12,6>::Zero(),Eigen::Matrix<double,12,12>::Identity();
	
    Eigen::Matrix<double,3,1> accdes;
    Eigen::Matrix<double,3,1> veldelta;
    Eigen::Matrix<double,3,1> posdelta;


    if(swl1==0)
    {
    accdes<< solution.ee_motion_.at(1)->GetPoint(t).a();

    World_bl = kinDynComp.getWorldTransform("back_right_foot");
    Eigen::MatrixXd eeBRposM = toEigen(World_bl.getPosition());
    Eigen::Vector3d eeBRpos;
    eeBRpos << eeBRposM.block(0,0,3,1);

    iDynTree::Twist br_vel;
    br_vel=kinDynComp.getFrameVel("back_right_foot");
    Eigen::MatrixXd eeBRvelM = toEigen(br_vel.getLinearVec3() );
    Eigen::Vector3d eeBRvel;
    eeBRvel << eeBRvelM.block(0,0,3,1);
  
    posdelta<< solution.ee_motion_.at(1)->GetPoint(t).p()-eeBRpos;
    veldelta<< solution.ee_motion_.at(1)->GetPoint(t).v()-eeBRvel;
    }

    if(swl1==6)
    {
    accdes<< solution.ee_motion_.at(2)->GetPoint(t).a();

    World_bl = kinDynComp.getWorldTransform("front_left_foot");
    Eigen::MatrixXd eeFLposM = toEigen(World_bl.getPosition());
    Eigen::Vector3d eeFLpos;
    eeFLpos << eeFLposM.block(0,0,3,1);

    iDynTree::Twist fl_vel;
    fl_vel=kinDynComp.getFrameVel("front_left_foot");
    Eigen::MatrixXd eeFLvelM = toEigen(fl_vel.getLinearVec3() );
    Eigen::Vector3d eeFLvel;
    eeFLvel << eeFLvelM.block(0,0,3,1);
  
    posdelta<< solution.ee_motion_.at(2)->GetPoint(t).p()-eeFLpos;
    veldelta<< solution.ee_motion_.at(2)->GetPoint(t).v()-eeFLvel;
    }

    if(swl1==3)
    {
    accdes<< solution.ee_motion_.at(1)->GetPoint(t).a();

    World_bl = kinDynComp.getWorldTransform("back_left_foot");
    Eigen::MatrixXd eeBLposM = toEigen(World_bl.getPosition());
    Eigen::Vector3d eeBLpos;
    eeBLpos << eeBLposM.block(0,0,3,1);

    iDynTree::Twist bl_vel;
    bl_vel=kinDynComp.getFrameVel("back_left_foot");
    Eigen::MatrixXd eeBLvelM = toEigen(bl_vel.getLinearVec3() );
    Eigen::Vector3d eeBLvel;
    eeBLvel << eeBLvelM.block(0,0,3,1);
  
    posdelta<< solution.ee_motion_.at(0)->GetPoint(t).p()-eeBLpos;
    veldelta<< solution.ee_motion_.at(0)->GetPoint(t).v()-eeBLvel;
    }

    if(swl1==9)
    {
    accdes<< solution.ee_motion_.at(3)->GetPoint(t).a();

    World_bl = kinDynComp.getWorldTransform("front_right_foot");
    Eigen::MatrixXd eeFRposM = toEigen(World_bl.getPosition());
    Eigen::Vector3d eeFRpos;
    eeFRpos << eeFRposM.block(0,0,3,1);

    iDynTree::Twist fr_vel;
    fr_vel=kinDynComp.getFrameVel("front_right_foot");
    Eigen::MatrixXd eeFRvelM = toEigen(fr_vel.getLinearVec3() );
    Eigen::Vector3d eeFRvel;
    eeFRvel << eeFRvelM.block(0,0,3,1);
  
    posdelta<< solution.ee_motion_.at(3)->GetPoint(t).p()-eeFRpos;
    veldelta<< solution.ee_motion_.at(3)->GetPoint(t).v()-eeFRvel;
    }

    
    Eigen::MatrixXd Kp;
    Kp=300*Eigen::MatrixXd::Identity(3,3);
    Eigen::MatrixXd Kd;
    Kd=20*Eigen::MatrixXd::Identity(3,3);       
    Eigen::Matrix<double,3,1> vdotswdes=accdes+Kd*veldelta+Kp*posdelta;
	eigenC.block(39,0,3,1)= vdotswdes-Jdqdsw;
	eigenC.block(42,0,3,1)= -vdotswdes+Jdqdsw;
	
    myQP = QP_SETUP_dense(30, 69, 15, eigenQ.data(), eigenA.data(), eigenD.data(), eigenc.data(), eigenC.data(), eigenb.data(), NULL, COLUMN_MAJOR_ORDERING); 
    //myQP->options->maxit  = 30;
    myQP->options->reltol = 1e-2 ;
    myQP->options->abstol  = 1e-2;     
    
    qp_int ExitCode = QP_SOLVE(myQP);
    
        Eigen::VectorXd x_;
        x_.resize( 30 );   
    for( int i=0; i<30; i++ ) {
        x_(i) = myQP->x[i];
    }
            
    Eigen::VectorXd tau= Eigen::VectorXd::Zero(12);
    tau=toEigen(MassMatrixCOM).block(6,6,12,12)*x_.block(6,0,12,1)+eigenBiascom-Jst.block(0,6,6,12).transpose()*x_.block(18,0,6,1);
    publish_cmd( tau );   

    if(swl1==0)
    {
    if( contact_br==true && t>duration-0.08 )
                    {flag_exit=true;}}
    if(swl1==6)
    {
    if( contact_fl==true && t>duration-0.08 )
                    {flag_exit=true;}}
    if(swl1==3)
    {
    if( contact_bl==true && t>duration-0.08 )
                    {flag_exit=true;}}
    if(swl1==9)
    {
    if( contact_fr==true && t>duration-0.08 )
                    {flag_exit=true;}}

}

int main(int argc, char** argv) {

    ros::init( argc, argv, "popt");
    DOGCTRL dc;
    dc.run();

}
