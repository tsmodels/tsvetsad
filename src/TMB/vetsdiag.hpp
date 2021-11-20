/// @file vetsdiag.hpp
#ifndef vetsdiag_hpp
#define vetsdiag_hpp

#undef TMB_OBJECTIVE_PTR
#define TMB_OBJECTIVE_PTR obj

template<class Type>
Type vetsdiag(objective_function<Type>* obj)
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
        DATA_IVECTOR(vmodel);
        PARAMETER_VECTOR(pars);
        //timesteps
        Eigen::SparseMatrix<Type> HH=asSparseMatrix(H);
        Eigen::SparseMatrix<Type> GG=asSparseMatrix(G);
        int t = vmodel(0);
        //series
        int n = vmodel(1);
        int inclx = vmodel(2);
        for(int i=0;i<pindex.size();i++){
            amat(mindex(i)) = pars(pindex(i));
        }
        matrix<Type> tmp = asMatrix(amat, adim(0), adim(1));
        matrix<Type> A = tmp.leftCols(adim(2));
        A.transposeInPlace();
        if (phiindex(0) >= 0)
        {
            vector<Type> vtmp = amat.segment(phiindex(0), phiindex(1));
            for(int i=0;i<vtmp.size();i++) {
                fmat(findex(i)) = vtmp(i);
            }
        }
        matrix<Type> F = asMatrix(fmat, fdim(0), fdim(1));
        Eigen::SparseMatrix<Type> FF=asSparseMatrix(F);
        matrix<Type> beta;
        if (xindex(0)>=0) {
            vector<Type> xbeta = amat.segment(xindex(0), xindex(1));
            beta = asMatrix(xbeta, betadim(0), betadim(1));
        } else {
            beta.setZero(betadim(0), betadim(1));
        }
        matrix<Type> GA = GG * A;
        matrix<Type> E(Y.rows(), Y.cols() - 1);
        E.setZero();
        matrix<Type> Cond = FF - GA * HH;
        Type spec_radius = vetsfun::power_iterations_fast(Cond, Type(1e-12), 1000);
        bool stability_test = (spec_radius < Type(1.01));
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
            Yhat = HH * states.col(i - 1);
            if(inclx) {
                Yhat.array() += (beta * X.col(i)).array();
            }
            E.col(i-1) = Y.col(i).array() -  Yhat.array();
            states.col(i) = (FF * states.col(i-1)).array() + (GA * E.col(i-1)).array();
        }
        E.transposeInPlace();
        loglik = vetsfun::vetsdiag_lik(E, Type(n), Type(t));
        return(loglik);
}

#undef TMB_OBJECTIVE_PTR
#define TMB_OBJECTIVE_PTR this

#endif
