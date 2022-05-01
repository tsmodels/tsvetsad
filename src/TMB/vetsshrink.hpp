/// @file vetsshrink.hpp
#ifndef vetsshrink_hpp
#define vetsshrink_hpp

#undef TMB_OBJECTIVE_PTR
#define TMB_OBJECTIVE_PTR obj


template<class Type>
Type vetsshrink(objective_function<Type>* obj) {
    DATA_MATRIX(Y);
    DATA_MATRIX(X);
    DATA_VECTOR(amat);
    DATA_MATRIX(states);
    DATA_VECTOR(fmat);
    DATA_MATRIX(G);
    DATA_MATRIX(H);
    DATA_MATRIX(good);
    DATA_VECTOR(select);
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
    int t = vmodel(0);
    //series
    int n = vmodel(1);
    int inclx = vmodel(2);
    Type rho = pars(pars.size() - 1);
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
    matrix<Type> Aux(Y.rows(), select.size());
    Aux.setZero();
    matrix<Type> Cond = F - GA * H;
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
    int kselect = 0;
    for (int i = 1; i < t; i++) {
        Yhat.array() = (H * states.col(i - 1)).array();
        if(inclx) {
            Yhat.array() += (beta * X.col(i)).array();
        }
        E.col(i-1) = (Y.col(i).array() -  Yhat.array()) * good.col(i-1).array();
        if (select(i-1) == 1) {
            Aux.col(kselect) = E.col(i-1).array();
            kselect+=1;
        }
        states.col(i) = (F * states.col(i-1)).array() + (GA * E.col(i-1)).array();
    }
    Aux.transposeInPlace();
    loglik = vetsfun::vetsshrink_lik(Aux, rho, Type(t), Type(n));
    return(loglik);
}

#undef TMB_OBJECTIVE_PTR
#define TMB_OBJECTIVE_PTR this

#endif
