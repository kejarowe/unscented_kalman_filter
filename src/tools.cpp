#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
	VectorXd rmse(4);
	rmse << 0,0,0,0;

	if (estimations.size() == 0 || estimations.size() != ground_truth.size())
	    return rmse;

	//accumulate squared residuals
	VectorXd this_term(4);
	for(int i=0; i < estimations.size(); ++i){
	    this_term << 0,0,0,0;
	    this_term = estimations[i] - ground_truth[i];
	    this_term = this_term.array()*this_term.array();
        rmse += this_term;
	}

	//calculate the mean
	rmse /= estimations.size();

	//calculate the squared root
	rmse = rmse.array().sqrt();

	//return the result
	return rmse;
}
