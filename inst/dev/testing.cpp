#include <TMB.hpp>
#include <Eigen/Eigenvalues>

#ifndef VETS_LN_2PI
#define VETS_LN_2PI 1.837877066409345483560659472811235279722794947275566825634
#endif

#ifndef VETS_LARGE_POS_NUM
#define VETS_LARGE_POS_NUM 1.0E8
#endif

#ifndef VETS_MAX_EIGVAL_TOL
#define VETS_MAX_EIGVAL_TOL 1.01
#endif

template<class Type>
matrix<Type> covariance(matrix<Type> x) {
    Type n = x.rows() - Type(1);
    matrix<Type> centered = x.rowwise() - x.colwise().mean();
    matrix<Type> cov = (centered.transpose() * centered)/n;
    return cov;
}

template<class Type>
matrix<Type> shrinkcov(matrix<Type> E, Type rho)
{
    matrix<Type> V = covariance(E);
    Type k = (rho * V.trace()/E.rows());
    matrix<Type> S =  (Type(1.0) - rho) * V;
    S = (S.array() + k).matrix();
    return S;
}

template<class Type>
matrix<Type> equicorr(matrix<Type> E, Type rho)
{
    vector<Type> sd = covariance(E).diagonal().array().sqrt();
    matrix<Type> R(sd.size(), sd.size());
    R.setConstant(rho);
    R.diagonal().setOnes();
    matrix<Type> S(sd.size(), sd.size());
    S.setZero();
    S.diagonal() = sd;
    return(S * R * S.transpose());
}
template<class Type>
Type power_iterations(matrix<Type> inp_mat, Type err_tol, const int max_iter)
{
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
        b /= sqrt((b * b).sum());
    }
    vector<Type> tmp = inp_mat * b;
    Type spec_rad = (b*tmp).sum() / (b*b).sum();
    return spec_rad;
}

template<class Type>
Type vetsdiagonal(matrix<Type> Error, Type n, Type t)
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
    matrix<Type> E = (Error * VV).array().pow(2);
    vector<Type> tmp = Type(1.0)/EV.array();
    Type sL = (E * tmp).sum();
    // Negative log likelihood
    Type loglik = Type(-1.0) * (lconstant - Type(0.5) * sL);
    return loglik;
}

template<class Type>
Type vetsfull(matrix<Type> Error, Type n, Type t)
{
    matrix<Type> S = covariance(Error);
    Eigen::SelfAdjointEigenSolver<Matrix<Type,Eigen::Dynamic,Eigen::Dynamic> > es(S);
    matrix<Type> VV = es.eigenvectors();
    vector<Type> EV = es.eigenvalues();
    matrix<Type> E = (Error * VV).array().pow(2);
    vector<Type> tmp = Type(1.0)/EV.array();
    Type sL = (E * tmp).sum();
    Type ldet = (EV.array().log()).matrix().sum();
    Type lconstant = Type(-0.5) * t * (n * VETS_LN_2PI + ldet);
    // Negative log likelihood
    Type loglik = - (lconstant - Type(0.5) * sL);
    return loglik;
}

template<class Type>
Type objective_function<Type>::operator() ()
{
    DATA_VECTOR(E);
    DATA_IVECTOR(edim);
    PARAMETER(rho);
    matrix<Type> A = asMatrix(E,edim(0), edim(1));
    Type n = A.cols();
    Type t = A.rows();
    //matrix<Type> S = equicorr(E, rho);
    Type llh = vetsfull(A, n, t);
    REPORT(llh);
    REPORT(A);
    return(0);
}
