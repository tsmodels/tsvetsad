#include <Eigen/Eigenvalues>

#ifndef VETS_LN_2PI
#define VETS_LN_2PI 1.837877066409345483560659472811235279722794947275566825634
#endif

#ifndef VETS_LARGE_POS_NUM
#define VETS_LARGE_POS_NUM 1.0E8
#endif

namespace vetsfun {
template<class Type>
matrix<Type> covariance(matrix<Type> x){
    Type n = x.rows() - Type(1);
    matrix<Type> centered = x.rowwise() - x.colwise().mean();
    matrix<Type> cov = (centered.transpose() * centered)/n;
    return cov;
}

template<class Type>
Type power_iterations(matrix<Type> inp_mat, Type err_tol, const int max_iter){
    vector<Type> b(inp_mat.cols());
    b.setConstant(Type(0.98));
    int iter = 0;
    Type err = Type(2.0) * err_tol;
    vector<Type> tmp = inp_mat * b;
    Type spec_rad = (b*tmp).sum() / (b*b).sum();
    Type spec_rad_old = spec_rad;
    while (err > err_tol && iter < max_iter) {
        iter++;
        b = inp_mat * b;
        b /= sqrt((b * b).sum());
        if ((iter > 100) && (iter % 20 == 0)) {
            tmp = inp_mat * b;
            spec_rad = (b*tmp).sum() / (b*b).sum();
            err = fabs(spec_rad - spec_rad_old);
            spec_rad_old = spec_rad;
        }
    }

    if (iter >= max_iter) {
        tmp = inp_mat * b;
        spec_rad = (b*tmp).sum() / (b*b).sum();
    }
    return spec_rad;
}

template<class Type>
Type power_iterations_fast(matrix<Type> inp_mat, Type err_tol, const int max_iter)
{
    vector<Type> b(inp_mat.cols());
    b.setConstant(Type(0.98));
    for (int i = 0; i < max_iter; ++i) {
        b = inp_mat * b;
        b /= b.matrix().norm();
    }
    vector<Type> tmp = inp_mat * b;
    Type spec_rad = (b*tmp).sum() / b.matrix().squaredNorm();
    return spec_rad;
}

template<class Type>
Type vetsshrink_lik(matrix<Type> Error, Type rho, Type t, Type n)
{
    matrix<Type> V = covariance(Error);
    Type k = (rho * V.trace()/Error.rows());
    matrix<Type> S =  (Type(1.0) - rho) * V;
    S = (S.array() + k).matrix();
    Eigen::SelfAdjointEigenSolver<Matrix<Type,Eigen::Dynamic,Eigen::Dynamic> > es(S);
    matrix<Type> VV = es.eigenvectors();
    vector<Type> EV = es.eigenvalues();
    matrix<Type> EX = (Error * VV).array().pow(2);
    vector<Type> tmp = Type(1.0)/EV.array();
    Type sL = (EX * tmp).sum();
    Type ldet = (EV.array().log()).matrix().sum();
    Type lconstant = Type(-0.5) * t * (n * VETS_LN_2PI + ldet);
    // Negative log likelihood
    Type loglik = - (lconstant - Type(0.5) * sL);
    return loglik;
}

template<class Type>
Type vetsequicor_lik(matrix<Type> Error, Type rho, Type t, Type n)
{
    vector<Type> sd = covariance(Error).diagonal().array().sqrt();
    matrix<Type> R(sd.size(), sd.size());
    R.setConstant(rho);
    R.diagonal().setOnes();
    matrix<Type> S(sd.size(), sd.size());
    S.setZero();
    S.diagonal() = sd;
    matrix<Type> SS = S * R * S.transpose();
    Eigen::SelfAdjointEigenSolver<Matrix<Type,Eigen::Dynamic,Eigen::Dynamic> > es(SS);
    matrix<Type> VV = es.eigenvectors();
    vector<Type> EV = es.eigenvalues();
    matrix<Type> EX = (Error * VV).array().pow(2);
    vector<Type> tmp = Type(1.0)/EV.array();
    Type sL = (EX * tmp).sum();
    Type ldet = (EV.array().log()).matrix().sum();
    Type lconstant = Type(-0.5) * t * (n * VETS_LN_2PI + ldet);
    // Negative log likelihood
    Type loglik = - (lconstant - Type(0.5) * sL);
    return loglik;
}

template<class Type>
Type vetsdiag_lik(matrix<Type> &Error, Type n, Type t)
{
    vector<Type> V = covariance(Error).diagonal();
    matrix<Type> S(Error.cols(), Error.cols());
    S.setZero();
    S.diagonal() = V.array();
    Type ldet = (V.array().log()).matrix().sum();
    Type lconstant = Type(-0.5) * t * (n * VETS_LN_2PI + ldet);
    Eigen::SelfAdjointEigenSolver<Matrix<Type,Eigen::Dynamic,Eigen::Dynamic> > es(S);
    matrix<Type> VV = es.eigenvectors();
    vector<Type> EV = es.eigenvalues();
    matrix<Type> EX = (Error * VV).array().pow(2);
    vector<Type> tmp = Type(1.0)/EV.array();
    Type sL = (EX * tmp).sum();
    // Negative log likelihood
    Type loglik = Type(-1.0) * (lconstant - Type(0.5) * sL);
    return loglik;
}

template<class Type>
Type vetsfull_lik(matrix<Type> Error, Type n, Type t)
{
    matrix<Type> S = covariance(Error);
    Eigen::SelfAdjointEigenSolver<Matrix<Type,Eigen::Dynamic,Eigen::Dynamic> > es(S);
    matrix<Type> VV = es.eigenvectors();
    vector<Type> EV = es.eigenvalues();
    matrix<Type> EX = (Error * VV).array().pow(2);
    vector<Type> tmp = Type(1.0)/EV.array();
    Type sL = (EX * tmp).sum();
    Type ldet = (EV.array().log()).matrix().sum();
    Type lconstant = Type(-0.5) * t * (n * VETS_LN_2PI + ldet);
    // Negative log likelihood
    Type loglik = - (lconstant - Type(0.5) * sL);
    return loglik;
}
}
