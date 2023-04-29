#include "dogbot_model/estimator_sem.h"



ESTIMATOR_SEM::ESTIMATOR_SEM()
{

}

ESTIMATOR_SEM::ESTIMATOR_SEM(QUADRUPED &quadruped_)

{
    dogbot = &quadruped_;
   
    yw.resize(4);
    w.resize(4);
    yd.resize(4);
    ygamma.resize(4);
}




void ESTIMATOR_SEM::estimate(Eigen::Matrix<double,6,1> &eigencomVel, Eigen::VectorXd tauj, Eigen::Matrix<double,12,1> Fgrf, std::vector<Eigen::Matrix<double,6,1>>  yd_prev,
                   std::vector<Eigen::Matrix<double,6,1>>  yw_prev, std::vector<Eigen::Matrix<double,6,1>>  w_prev, std::vector<Eigen::Matrix<double,6,1>>  ygamma_prev)
{
Eigen::MatrixXd Mcom=dogbot->getMassMatrixCOM_com();
Eigen::Matrix<double,6,1> q_dot=eigencomVel;
Eigen::Matrix<double,12,6> J=dogbot->getJacobianCOM_linear().block(0,0,12,6);
Eigen::Matrix<double,6,1> Ctq=dogbot->getCtq().block(0,0,6,1);
Eigen::Matrix<double,6,1> fc=J.transpose()*Fgrf;
Eigen::Matrix<double,6,1> g=dogbot->getGravityMatrixCOM().block(0,0,6,1);

Eigen::MatrixXd p1 = Mcom*q_dot;

std::cout<<"ctq"<<Ctq<<std::endl;
std::cout<<"g"<<g<<std::endl;
  double mass_robot=dogbot->getMass();

  Eigen::MatrixXd g_acc=Eigen::MatrixXd::Zero(6,1);
  g_acc(2,0)=9.81;
Eigen::MatrixXd intg=Ctq-mass_robot*g_acc+fc;
 

Eigen::MatrixXd rho=p1;
Eigen::MatrixXd d=intg;

Eigen::VectorXd coeffs=Eigen::VectorXd::Zero(2);
coeffs<< 10, 1;
std::vector<Eigen::Matrix<double,6,6>> k(2);

for (int i=0; i<1; i++)
{
   k[i]=coeffs[i]*Eigen::Matrix<double,6,6>::Identity();
}

double T=0.001;
Eigen::Matrix<double,6,6> m=(Eigen::Matrix<double,6,6>::Identity()+k[0]*T).inverse()*k[0];


  yd[0]=yd_prev[0]+(d*T);
  
  w[0] = k[0]*(rho-yw_prev[0]-yd[0]);
  
  yw[0]=yw_prev[0]+w[0]*T;

  std::cout<<"wex"<<w[0]<<std::endl;

 
}

Eigen::Matrix<double,12,1> ESTIMATOR_SEM::getw3()
{   Eigen::Matrix<double,12,6> J=dogbot->getJacobianCOM_linear().block(0,0,12,6);
    Eigen::Matrix<double,6,6> M=dogbot->getMassMatrixCOM_com();
    Eigen::MatrixXd Jinv=(dogbot->getJacobianCOM_linear().block(0,0,12,6)).completeOrthogonalDecomposition().pseudoInverse();
    Eigen::Matrix<double,12,1> w3=Jinv.transpose()*w[0];
    return w3;
}

std::vector<Eigen::Matrix<double,6,1>> ESTIMATOR_SEM::getyd()
{ 
    return yd;
}

std::vector<Eigen::Matrix<double,6,1>> ESTIMATOR_SEM::getyw()
{   
    return yw;
}

std::vector<Eigen::Matrix<double,6,1>> ESTIMATOR_SEM::getw()
{  
    return w;
}

std::vector<Eigen::Matrix<double,6,1>> ESTIMATOR_SEM::getygamma()
{   
    return ygamma;
}

Eigen::Matrix<double,6,1> ESTIMATOR_SEM::getwext()
{   Eigen::Matrix<double,12,6> J=dogbot->getJacobianCOM_linear().block(0,0,12,6);
    Eigen::Matrix<double,6,6> M=dogbot->getMassMatrixCOM_com();
    Eigen::MatrixXd Jinv=M.inverse()*J.transpose()*(J*M.inverse()*J.transpose()).inverse();
    Eigen::Matrix<double,6,1> w3=w[0];
    return w3;
}