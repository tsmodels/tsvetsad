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

// template<class Type>
// Type objective_function<Type>::operator() ()
// {
//   DATA_MATRIX(E);
//   PARAMETER(rho);
//   Type n = E.cols();
//   Type t = E.rows();
//   //matrix<Type> S = equicorr(E, rho);
//   Type llh = vetsfull(E, n, t);
//   REPORT(llh);
//   return(0);
// }

template<class Type>
Type objective_function<Type>::operator() ()
{
  DATA_MATRIX(Y);
  DATA_MATRIX(X);
  DATA_VECTOR(amat);
  DATA_MATRIX(states);
  DATA_VECTOR(fmat);
  DATA_MATRIX(G);
  DATA_MATRIX(H);
  DATA_IVECTOR(adim);
  DATA_IVECTOR(mindex);
  DATA_IVECTOR(pindex);
  DATA_IVECTOR(fdim);
  DATA_IVECTOR(phiindex);
  DATA_IVECTOR(findex);
  DATA_IVECTOR(xindex);
  DATA_IVECTOR(betadim);
  DATA_IVECTOR(model);
  PARAMETER_VECTOR(pars);
  //timesteps
  int t = model(0);
  //series
  int n = model(1);
  int inclx = model(2);
  for(int i=0;i<pindex.size();i++){
    amat(mindex(i)) = pars(pindex(i));
  }
  matrix<Type> tmp = asMatrix(amat, adim(0), adim(1));
  matrix<Type> A = tmp.leftCols(adim(2));
  A.transposeInPlace();
  REPORT(A);
  if (phiindex(0) >= 0)
  {
    vector<Type> vtmp = amat.segment(phiindex(0), phiindex(1));
    for(int i=0;i<vtmp.size();i++) {
      fmat(findex(i)) = vtmp(i);
    }
  }
  matrix<Type> F = asMatrix(fmat, fdim(0), fdim(1));
  REPORT(F);
  matrix<Type> beta;
  if (xindex(0)>=0) {
      vector<Type> xbeta = amat.segment(xindex(0), xindex(1));
      beta = asMatrix(xbeta, betadim(0), betadim(1));
  } else {
     beta.setZero(betadim(0), betadim(1));
  }
  REPORT(beta);
  matrix<Type> GA = G * A;
  matrix<Type> E(Y.rows(), Y.cols() - 1);
  E.setZero();
  matrix<Type> Cond = F - GA * H;
  Type spec_radius = power_iterations_fast(Cond, Type(1e-12), 1000);
  bool stability_test = (spec_radius < VETS_MAX_EIGVAL_TOL);
  REPORT(spec_radius);
  Type loglik = 0.0;
  REPORT(stability_test);
  if (!stability_test) {
     loglik = 0.0;
     return(loglik);
  }
  vector<Type> Yhat(n);
  Yhat.setZero();
  for (int i = 1; i < t; i++) {
     Yhat.array() = (H * states.col(i - 1)).array();
     if(inclx) {
       Yhat.array() += (beta * X.col(i)).array();
     }
     E.col(i-1) = Y.col(i).array() -  Yhat.array();
     states.col(i) = (F * states.col(i-1)).array() + (GA * E.col(i-1)).array();
  }
  E.transposeInPlace();
  REPORT(E);
  loglik = vetsdiagonal(E, Type(n), Type(t));
  REPORT(loglik);
  REPORT(Yhat);
  return(loglik);
}

