#include <RcppArmadillo.h>
using namespace Rcpp;

inline double power_iterations(const arma::mat& inp_mat, const double err_tol = 1.0e-04, const int max_iter = 1000)
  {
    // arma::vec b = arma::randu(inp_mat.n_cols) + 0.2;
    arma::vec b = arma::ones(inp_mat.n_cols) - 0.02;
    
    int iter = 0;
    double err = 2*err_tol;
    
    double spec_rad = arma::dot(b,inp_mat*b) / arma::dot(b,b);
    double spec_rad_old = spec_rad;
    
    while (err > err_tol && iter < max_iter) {
      iter++;
      
      b = inp_mat * b;
      b /= arma::norm(b, 2);
      
      if ((iter > 100) && (iter % 20 == 0)) {
        spec_rad = arma::dot(b,inp_mat*b) / arma::dot(b,b);
        
        err = std::abs(spec_rad - spec_rad_old);
        spec_rad_old = spec_rad;
      }
    }
    
    if (iter >= max_iter) {
      spec_rad = arma::dot(b,inp_mat*b) / arma::dot(b,b);
    }
    
    return spec_rad;
  }

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp14)]]
// [[Rcpp::export]]
Rcpp::List vtest(arma::mat X)
{
  double spec_radius = power_iterations(X);
  Rcpp::List output = Rcpp::List::create(Rcpp::Named("radius") = spec_radius,
                                         Rcpp::Named("condition") = 0);
  return(output);
  
}
  
