#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd::Zero(5);

  // initial covariance matrix
  P_ = MatrixXd::Zero(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.8; //30; //was 0.2 in lesson 

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.6; //30; //was 0.2 in lesson 

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  H_laser_ = MatrixXd(2,5);
  H_laser_ << 1, 0, 0, 0, 0,
              0, 1, 0, 0 ,0;

  R_laser_ = MatrixXd(2,2);
  R_laser_ << std_laspx_*std_laspx_, 0,
              0, std_laspy_*std_laspy_;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3; //was 0.3 in lesson

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03; //was 0.0175 in lesson

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3; //was 0.1 in lesson

  R_radar_ = MatrixXd::Zero(3,3);
  R_radar_(0,0) = std_radr_ * std_radr_;
  R_radar_(1,1) = std_radphi_ * std_radphi_;
  R_radar_(2,2) = std_radrd_ * std_radrd_;

  n_x_ = x_.size();

  n_aug_ = n_x_ + 2;

  lambda_ = 3 - n_aug_;

  weights_ = VectorXd(2*n_aug_ + 1);
  //set weights
  weights_.setOnes();
  weights_ *= 1/(2*(lambda_ + n_aug_));
  weights_(0) *= 2*lambda_;

  is_initialized_ = false;
}

UKF::~UKF() {}

MatrixXd UKF::AugmentedSigmaPoints() {

  //create augmented mean vector
  VectorXd x_aug = VectorXd(7);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  //create augmented mean state
  x_aug.setZero(7,1);
  x_aug.head(5) = x_;
  
  //create augmented covariance matrix
  P_aug.setZero(7,7);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;
  
  //create square root matrix
  MatrixXd P_aug_scaled = P_aug.llt().matrixL();
  P_aug_scaled *= sqrt(lambda_+n_aug_);
  
  //create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; i++){
      Xsig_aug.col(i+1) = x_aug + P_aug_scaled.col(i);
      Xsig_aug.col(i+1+n_aug_) = x_aug - P_aug_scaled.col(i);
  }
  return Xsig_aug;
}

MatrixXd UKF::SigmaPointPrediction( double delta_t) {

  MatrixXd Xsig_aug = AugmentedSigmaPoints();
  //create matrix with predicted sigma points as columns
  MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);

  float dt_sq_2 = delta_t*delta_t*0.5;
  
  static double px,py,v,psi,psi_dot,neu_a,neu_psi_dot_dot;
  for (int i = 0; i < Xsig_pred.cols(); i ++)
  {
      px = Xsig_aug(0,i);
      py = Xsig_aug(1,i);
      v = Xsig_aug(2,i);
      psi = Xsig_aug(3,i);
      psi_dot = Xsig_aug(4,i);
      neu_a = Xsig_aug(5,i);
      neu_psi_dot_dot = Xsig_aug(6,i);
      
      Xsig_pred(0,i) = px + dt_sq_2*cos(psi)*neu_a;
      Xsig_pred(1,i) = py + dt_sq_2*sin(psi)*neu_a;
      Xsig_pred(2,i) = v + delta_t*neu_a;
      Xsig_pred(3,i) = psi + psi_dot*delta_t + dt_sq_2*neu_psi_dot_dot;
      Xsig_pred(4,i) = psi_dot + delta_t*neu_psi_dot_dot;

      //angle normalization
      NormalizeAngle(Xsig_pred(3));
      
      if (fabs(psi_dot) <= 0.001) {
          Xsig_pred(0,i) += v*cos(psi)*delta_t;
          Xsig_pred(1,i) += v*sin(psi)*delta_t;
      }
      else {
          Xsig_pred(0,i) += (v/psi_dot)*(sin(psi + psi_dot*delta_t) - sin(psi));
          Xsig_pred(1,i) += (v/psi_dot)*(-cos(psi + psi_dot*delta_t) + cos(psi));
      }
  }

  return Xsig_pred;
}

void UKF::PredictMeanAndCovariance() {
  //predict state mean
  x_ = Xsig_pred_ * weights_;
  NormalizeAngle(x_(3));
  
  //predict state covariance matrix
  MatrixXd P_h1 = MatrixXd(n_x_, 2 * n_aug_ + 1);
  MatrixXd P_h2 = MatrixXd(n_x_, 2 * n_aug_ + 1); //helper matricies for calculation
  VectorXd diff = VectorXd(n_x_);
  for (int i = 0 ; i < Xsig_pred_.cols() ; i++){
      diff = Xsig_pred_.col(i) - x_;
      
      //angle normalization
      NormalizeAngle(diff(3));

      P_h1.col(i) = diff * weights_(i);
      P_h2.col(i) = diff;
  }
  P_ = P_h1 * P_h2.transpose();
}

bool UKF::PredictRadarMeasurement(VectorXd* z_pred_out, MatrixXd* Zsig_pred_out, MatrixXd* S_pred_out) {

  //set measurement dimension, radar can measure r, phi, and r_dot
  static int n_z = 3;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);

  //transform sigma points into measurement space
  for (int i = 0; i < Zsig.cols() ; i++){
      double x = Xsig_pred_(0,i);
      double y = Xsig_pred_(1,i);
      double v = Xsig_pred_(2,i);
      double psi = Xsig_pred_(3,i);
      double psi_dot = Xsig_pred_(4,i);

      if (x == 0 && y == 0)
      {
        //can't predict measurement for this state due to
        //divide by zero
        return false;
      }
      
      Zsig(0,i) = sqrt(x*x+y*y);
      Zsig(1,i) = atan2(y,x);
      Zsig(2,i) = (x*cos(psi)*v + y*sin(psi)*v) / Zsig(0,i);
  }
  //calculate mean predicted measurement
  z_pred = Zsig*weights_;
  
  //calculate measurement covariance matrix S
  MatrixXd S_h1 = MatrixXd(n_z,2 * n_aug_ + 1);
  MatrixXd S_h2 = MatrixXd(n_z,2 * n_aug_ + 1);
  
  VectorXd diff(n_z);
  
  for (int i = 0; i < Zsig.cols(); i++)
  {
      //angle normalization
      diff = Zsig.col(i) - z_pred;
      NormalizeAngle(diff(1));
      
      S_h1.col(i) = weights_(i) * diff;
      S_h2.col(i) = diff;
  }
  
  S = S_h1 * S_h2.transpose();
  S(0,0) += std_radr_*std_radr_;
  S(1,1) += std_radphi_*std_radphi_;
  S(2,2) += std_radrd_ * std_radrd_;
 
  *z_pred_out = z_pred;
  *Zsig_pred_out = Zsig;
  *S_pred_out = S;
  return true;
}

void UKF::RadarUpdateState(VectorXd &z_pred, MatrixXd& Zsig_pred, MatrixXd& S_pred, VectorXd &z) {

  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  //create copy of Xsig_pred
  MatrixXd Xsig_pred = Xsig_pred_;

  //calculate cross correlation matrix
  for (int i = 0; i < Xsig_pred.cols(); i++) {
      Xsig_pred.col(i) = weights_(i)*(Xsig_pred.col(i) - x_);
      Zsig_pred.col(i) -= z_pred;
  }
  MatrixXd Tc = Xsig_pred*Zsig_pred.transpose();
  //calculate Kalman gain K;
  MatrixXd K = Tc*S_pred.inverse();

  //update state mean and covariance matrix
  x_ += K*(z-z_pred);
  P_ -= K*S_pred*K.transpose();
}


void UKF::NormalizeAngle(double &angle)
{
  while (angle > M_PI) angle -= 2.0*M_PI;
  while (angle < -M_PI) angle += 2.0*M_PI;
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (is_initialized_)
  {
    double dt = (meas_package.timestamp_ - time_us_) * 1e-6;
    //if dt is large, do prediction in small steps to prevent numerical instability
    //https://discussions.udacity.com/t/numerical-instability-of-the-implementation/230449/3
    while (dt > 0.05) {
      Prediction(0.05);
      dt -= 0.05;
    }
    Prediction(dt);
    if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
      UpdateLidar(meas_package);
    }
    else if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
      UpdateRadar(meas_package);
    }
  }
  else
  {
    //initialize
    if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
      x_ = H_laser_.transpose()*meas_package.raw_measurements_;
      
      P_ = MatrixXd::Identity(5,5);
      P_(0,0) = H_laser_(0,0);
      P_(1,1) = H_laser_(1,1);
      is_initialized_ = true;
    }
    else if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
      double rho, psi, rhod;
      rho = meas_package.raw_measurements_(0);
      psi = meas_package.raw_measurements_(1);
      rhod = meas_package.raw_measurements_(2);
      x_(0) = rho * cos(psi);
      x_(1) = rho * sin(psi);
      x_(2) = rhod;
      x_(3) = 0;
      x_(4) = 0;
      P_ = MatrixXd::Identity(5,5);
      is_initialized_ = true;
    }
  }
  time_us_ = meas_package.timestamp_;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  Xsig_pred_ = SigmaPointPrediction(delta_t);
  PredictMeanAndCovariance();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  VectorXd y = meas_package.raw_measurements_ - H_laser_ * x_;
  MatrixXd S = H_laser_*P_*H_laser_.transpose() + R_laser_;
  MatrixXd K = P_*H_laser_.transpose()*S.inverse();
  x_ += K*y;
  P_ = ( MatrixXd::Identity(5,5) - K*H_laser_) * P_;

  NIS_laser_ = y.transpose() * S.inverse() * y;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  VectorXd z_pred;
  MatrixXd Zsig_pred;
  MatrixXd S_pred;

  if (PredictRadarMeasurement(&z_pred, &Zsig_pred, &S_pred)) {
    RadarUpdateState(z_pred, Zsig_pred, S_pred, meas_package.raw_measurements_);
  
  //calculate NIS
  VectorXd y = meas_package.raw_measurements_ - z_pred;
  NIS_radar_ = y.transpose() * S_pred.inverse() * y;
  }
}
