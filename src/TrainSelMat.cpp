#include <vector>
#include <iterator>

#include <algorithm>
#include <RcppArmadillo.h>
#include <queue>
using namespace Rcpp;
using namespace std;
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppProgress)]]
#include <progress.hpp>
#include <progress_bar.hpp>


///////////////////////
NumericVector subsetNumVec(NumericVector x, IntegerVector index) {
  // Length of the index vector
  int n = index.length();
  // Initialize output vector
  NumericVector out(n);

  // Subtract 1 from index as C++ starts to count at 0
  index = index - 1;
  // Loop through index vector and extract values of x at the given positions
  for (int i = 0; i < n; i++) {
    out[i] = x[index[i]];
  }

  // Return output
  return out;
}


IntegerVector subsetIntVec(IntegerVector x, IntegerVector index) {
  // Length of the index vector
  int n = index.length();
  // Initialize output vector
  IntegerVector out(n);

  // Subtract 1 from index as C++ starts to count at 0
  index = index - 1;
  // Loop through index vector and extract values of x at the given positions
  for (int i = 0; i < n; i++) {
    out[i] = x[index[i]];
  }

  // Return output
  return out;
}



Rcpp::IntegerVector whichRcpp(const Rcpp::LogicalVector& x) {
  Rcpp::IntegerVector v = Rcpp::seq(0, x.length() - 1);
  return v[x];
}


bool contains(IntegerVector x, int b){
  bool cont=false;
  for (int i=0;i<x.length();i++){
    if (x[i]==b){
      cont=true;
    }
  }
  return cont;
}


IntegerVector orderRcpp(vector<double> v) {
  int n = v.size();
  typedef pair<double, int> Elt;
  priority_queue< Elt, vector<Elt>, greater<Elt> > pq;
  vector<int> result;
  for (int i = 0; i != v.size(); ++i) {
    if (pq.size() < n)
      pq.push(Elt(v[i], i));
    else {
      Elt elt = Elt(v[i], i);
      if (pq.top() < elt) {
        pq.pop();
        pq.push(elt);
      }
    }
  }

  result.reserve(pq.size());
  while (!pq.empty()) {
    result.push_back(pq.top().second + 1);
    pq.pop();
  }
  return wrap(result);

}



// [[Rcpp::export]]
arma::mat nearPDc(arma::mat X){
  arma::colvec d;
  arma::mat Q;
  eig_sym(d, Q, X);

  double Eps = 1e-7 * std::abs(d[X.n_cols-1]);
  if (d(0) < Eps) {
    arma::uvec d_comp = d < Eps;
    for(int i=0;i<sum(d_comp);i++){
      if(d_comp(i)){
        d(i)=Eps;
      }
    }
  }
  return Q*arma::diagmat(d)*Q.t();
}





//////////////////////////////

double crossprodRcpp(const NumericVector& x, const NumericVector& y) {
  int nx = x.length();
  int ny = y.length();
  double sumup = 0;

  if (nx == ny) {
    for (int i = 0; i < nx; i++)
      sumup += x[i] * y[i];
  } else
    sumup = NA_REAL; // NA_REAL: constant of NA value for numeric (double) values

  return sumup;
}

bool is_duplicate_row(int& r, NumericMatrix& x) {
  int i = 0, nr = x.nrow();
  const NumericMatrix::Row y = x.row(r);

  for (; i < r; i++) {
    if (is_true(all(y == x.row(i)))) {
      return true;
    }
  }
  for (i = r + 1; i < nr; i++) {
    if (is_true(all(y == x.row(i)))) {
      return true;
    }
  }

  return false;
}

LogicalVector duplicatedRcpp(NumericMatrix m) {
  int nr = m.nrow();
  int i;
  LogicalVector out(nr);
  for (i = 0; i < nr; i++) {
    out(i) = is_duplicate_row(i, m);
  }
  return out;
}

NumericMatrix cbindNM(const NumericMatrix& a, const NumericMatrix& b) {
  //the colnumber of first matrix
  int acoln = a.ncol();
  //the colnumber of second matrix
  int bcoln = b.ncol();
  //build a new matrix, the dim is a.nrow() and acoln+bcoln
  NumericMatrix out(a.nrow(), acoln + bcoln);
  for (int j = 0; j < acoln + bcoln; j++) {
    if (j < acoln) {
      out(_, j) = a(_, j);
    } else {
      //put the context in the second matrix to the new matrix
      out(_, j) = b(_, j - acoln);
    }
  }
  return out;
}

IntegerMatrix cbindIM(const IntegerMatrix& a, const IntegerMatrix& b) {
  //the colnumber of first matrix
  int acoln = a.ncol();
  //the colnumber of second matrix
  int bcoln = b.ncol();
  //build a new matrix, the dim is a.nrow() and acoln+bcoln
  IntegerMatrix out(a.nrow(), acoln + bcoln);
  for (int j = 0; j < acoln + bcoln; j++) {
    if (j < acoln) {
      out(_, j) = a(_, j);
    } else {
      //put the context in the second matrix to the new matrix
      out(_, j) = b(_, j - acoln);
    }
  }
  return out;
}

LogicalMatrix cbindLM(const LogicalMatrix& a, const LogicalMatrix& b) {
  //the colnumber of first matrix
  int acoln = a.ncol();
  //the colnumber of second matrix
  int bcoln = b.ncol();
  //build a new matrix, the dim is a.nrow() and acoln+bcoln
  LogicalMatrix out(a.nrow(), acoln + bcoln);
  for (int j = 0; j < acoln + bcoln; j++) {
    if (j < acoln) {
      out(_, j) = a(_, j);
    } else {
      //put the context in the second matrix to the new matrix
      out(_, j) = b(_, j - acoln);
    }
  }
  return out;
}

Rcpp::NumericMatrix subcolNM(
    const Rcpp::NumericMatrix& x, const Rcpp::IntegerVector& y) {

  // Determine the number of observations
  int n_cols_out = y.length();

  // Create an output matrix
  Rcpp::NumericMatrix out = Rcpp::no_init(x.nrow(), n_cols_out);

  // Loop through each column and copy the data.
  for (unsigned int z = 0; z < n_cols_out; ++z) {
    out(Rcpp::_, z) = x(Rcpp::_, y[z]);
  }

  return out;
}

Rcpp::IntegerMatrix subcolIM(
    const Rcpp::IntegerMatrix& x, const Rcpp::IntegerVector& y) {

  // Determine the number of observations
  int n_cols_out = y.length();

  // Create an output matrix
  Rcpp::IntegerMatrix out = Rcpp::no_init(x.nrow(), n_cols_out);

  // Loop through each column and copy the data.
  for (unsigned int z = 0; z < n_cols_out; ++z) {
    out(Rcpp::_, z) = x(Rcpp::_, y[z]);
  }

  return out;
}

Rcpp::IntegerVector subcolIM0(
    const Rcpp::IntegerMatrix& x, const int& y) {
  Rcpp::IntegerVector out = x(Rcpp::_, y);

  return out;
}

Rcpp::LogicalMatrix subcolLM(
    const Rcpp::LogicalMatrix& x, const Rcpp::IntegerVector& y) {

  // Determine the number of observations
  int n_cols_out = y.length();

  // Create an output matrix
  Rcpp::LogicalMatrix out = Rcpp::no_init(x.nrow(), n_cols_out);

  // Loop through each column and copy the data.
  for (unsigned int z = 0; z < n_cols_out; ++z) {
    out(Rcpp::_, z) = x(Rcpp::_, y[z]);
  }

  return out;
}






int dominates(NumericMatrix p, int i, int j, int nobj) {
  int i_flagged = 0;
  int j_flagged = 0;
  int k;
  NumericVector pi = p(_, i);
  NumericVector pj = p(_, j);
  for (k = 0; k < nobj; ++k) {
    const double p_ik = pi[k];
    const double p_jk = pj[k];
    if (p_ik > p_jk) {
      j_flagged = 1;
    } else if (p_jk > p_ik) {
      i_flagged = 1;
    }
  }
  return j_flagged - i_flagged;
}

LogicalVector do_is_dominated(NumericMatrix points) {

  int i, j;


  int d = points.nrow();
  int n = points.ncol();
  LogicalVector res(n);

  for (i = 0; i < n; ++i) {
    res(i) = false;
  }

  for (i = 0; i < n; ++i) {
    if (res(i)) {
      continue;
    }
    for (j = (i + 1); j < n; ++j) {
      if (res(j)) {
        continue;
      }
      int dom = dominates(points, i, j, d);
      if (dom > 0) {
        res[j] = true;
      } else if (dom < 0) {
        res[i] = true;
      }
    }
  }

  return res;
}




int contains(const StringVector& X, const StringVector& z) {
  int out;
  if (std::find(X.begin(), X.end(), z(0)) != X.end()) {
    out = 1L;
  } else {
    out = 0L;
  }
  return out;
}






/////////////



struct STATCLASS {
public:
  Rcpp::List Data = Rcpp::List::create();
  arma::mat G;
  arma::mat R;
  arma::mat X;
  Rcpp::IntegerVector Target;
  std::string typestat;
  int ntotal=0;
  bool CD=false;
  int AllinG=0;

  STATCLASS(const Rcpp::List& Data_) {
    Data = Data_;
    typestat = "UDD";
  }

  STATCLASS() {
    typestat = "UD";
  }

  STATCLASS(arma::mat& X_, arma::mat& G_, arma::mat& R_) {
    G = G_;
    R = R_;
    X = X_;
    typestat = "CDMEANX";
    CD=true;
  }

  STATCLASS(const Rcpp::IntegerVector& Target_, arma::mat& X_, arma::mat& G_, arma::mat& R_) {
    G = G_;
    R = R_;
    X = X_;
    Target = Target_;
    typestat = "CDMEANTX";
    CD=true;

  }

  STATCLASS(const arma::mat& G_, const arma::mat& R_) {
    G = G_;
    R = R_;
    typestat = "CDMEAN";
    CD=true;
  }

  STATCLASS(const Rcpp::IntegerVector& Target_, arma::mat& G_, arma::mat& R_) {
    G = G_;
    R = R_;
    Target = Target_;
    typestat = "CDMEANT";
    CD=true;
  }


  void setntotal(int ntotal_){
    ntotal=ntotal_;
  }

  void setAllinG(int AllinG_){
    AllinG=AllinG_;
  }


  IntegerVector getInds(IntegerVector soln_int){
    if (!CD){
      return soln_int;
    } else {
      IntegerVector Inds;

      for (int i=0;i<soln_int.length();i++){
        int  tempint=soln_int(i)%AllinG;
        if (tempint==0){tempint=AllinG;}
        Inds.push_back(tempint);

      }
      return Inds;
    }
  }

  double GetStat(const Rcpp::IntegerVector& soln_int, const Rcpp::NumericVector& soln_dbl, Rcpp::Function Stat) {
    int nunique=unique(getInds(soln_int)).length();
    if (((ntotal!=nunique) & (ntotal!=0))){
      return -1e+10*abs(ntotal-nunique);
    } else {
      if (typestat == "UD") {
        return as<double>(Stat(soln_int, soln_dbl));
      } else if (typestat == "UDD") {
        return as<double>(Stat(soln_int, soln_dbl, Data));
      } else if (typestat == "CDMEANTX") {
        arma::uvec solalluvec = as<arma::uvec>(wrap(soln_int));
        arma::uvec Targetuvec = as<arma::uvec>(wrap(Target));
        arma::mat V = G.submat(solalluvec - 1, solalluvec - 1) + R.submat(solalluvec - 1, solalluvec - 1) + 1e-15 * arma::eye(solalluvec.size(), solalluvec.size());
        arma::mat Vinv = arma::pinv(V, 1e-5, "std");
        arma::mat P = -X * arma::solve(X.t() * Vinv*X, X.t(), arma::solve_opts::likely_sympd + arma::solve_opts::fast) * Vinv + arma::eye(X.n_rows, X.n_rows);
        arma::vec D = sum(G.submat(Targetuvec - 1, solalluvec - 1) * Vinv * P % G.submat(Targetuvec - 1, solalluvec - 1), 1);
        arma::mat Num = G.submat(Targetuvec - 1, Targetuvec - 1);
        return min(D / Num.diag());
      } else if (typestat == "CDMEANX") {
        arma::uvec solalluvec = as<arma::uvec>(wrap(soln_int));
        arma::mat V = G.submat(solalluvec - 1, solalluvec - 1) + R.submat(solalluvec - 1, solalluvec - 1) + 1e-15 * arma::eye(solalluvec.size(), solalluvec.size());
        arma::mat Vinv = arma::pinv(V, 1e-5, "std");
        arma::mat P = -X * solve(X.t() * Vinv*X, X.t(), arma::solve_opts::likely_sympd + arma::solve_opts::fast) * Vinv + arma::eye(X.n_rows, X.n_rows);
        arma::vec D = sum(G.cols(solalluvec - 1) * Vinv * P % G.cols(solalluvec - 1), 1);
        return min(D / G.diag());
      } else if (typestat == "CDMEAN") {
        arma::uvec solalluvec = as<arma::uvec>(wrap(soln_int));
        arma::mat V = G.submat(solalluvec - 1, solalluvec - 1) + R.submat(solalluvec - 1, solalluvec - 1) + 1e-15 * arma::eye(solalluvec.size(), solalluvec.size());
        arma::vec D = sum(G.cols(solalluvec - 1) * arma::pinv(V, 1e-5, "std") % G.cols(solalluvec - 1), 1);
        return min(D / G.diag());
      } else if (typestat == "CDMEANT") {
        arma::uvec solalluvec = as<arma::uvec>(wrap(soln_int));
        arma::uvec Targetuvec = as<arma::uvec>(wrap(Target));
        arma::mat V = G.submat(solalluvec - 1, solalluvec - 1) + R.submat(solalluvec - 1, solalluvec - 1) + 1e-15 * arma::eye(solalluvec.size(), solalluvec.size());
        arma::vec D = sum(G.submat(Targetuvec - 1, solalluvec - 1) * arma::pinv(V, 1e-5, "std") % G.submat(Targetuvec - 1, solalluvec - 1), 1);
        arma::mat Num = G.submat(Targetuvec - 1, Targetuvec - 1);
        return min(D / Num.diag());
      } else {
        return 0;
      }
    }
  }


  double GetStat(const Rcpp::IntegerVector& soln_int, Rcpp::Function Stat) {
    int nunique=unique(getInds(soln_int)).length();
    if (((ntotal!=nunique) & (ntotal!=0))){
      return -1e+10*abs(ntotal-nunique);
    } else {
      if (typestat == "UD") {
        return as<double>(Stat(soln_int));
      } else if (typestat == "UDD") {
        return as<double>(Stat(soln_int, Data));
      } else if (typestat == "CDMEANTX") {
        arma::uvec solalluvec = as<arma::uvec>(wrap(soln_int));
        arma::uvec Targetuvec = as<arma::uvec>(wrap(Target));
        arma::mat V = G.submat(solalluvec - 1, solalluvec - 1) + R.submat(solalluvec - 1, solalluvec - 1) + 1e-15 * arma::eye(solalluvec.size(), solalluvec.size());
        arma::mat Vinv = arma::pinv(V, 1e-5, "std");
        arma::mat P = -X * arma::solve(X.t() * Vinv*X, X.t(), arma::solve_opts::likely_sympd + arma::solve_opts::fast) * Vinv + arma::eye(X.n_rows, X.n_rows);
        arma::vec D = sum(G.submat(Targetuvec - 1, solalluvec - 1) * Vinv * P % G.submat(Targetuvec - 1, solalluvec - 1), 1);
        arma::mat Num = G.submat(Targetuvec - 1, Targetuvec - 1);
        return min(D / Num.diag());
      } else if (typestat == "CDMEANX") {
        arma::uvec solalluvec = as<arma::uvec>(wrap(soln_int));
        arma::mat V = G.submat(solalluvec - 1, solalluvec - 1) + R.submat(solalluvec - 1, solalluvec - 1) + 1e-15 * arma::eye(solalluvec.size(), solalluvec.size());
        arma::mat Vinv = arma::pinv(V, 1e-5, "std");
        arma::mat P = -X * solve(X.t() * Vinv*X, X.t(), arma::solve_opts::likely_sympd + arma::solve_opts::fast) * Vinv + arma::eye(X.n_rows, X.n_rows);
        arma::vec D = sum(G.cols(solalluvec - 1) * Vinv * P % G.cols(solalluvec - 1), 1);
        return min(D / G.diag());
      } else if (typestat == "CDMEAN") {
        arma::uvec solalluvec = as<arma::uvec>(wrap(soln_int));
        arma::mat V = G.submat(solalluvec - 1, solalluvec - 1) + R.submat(solalluvec - 1, solalluvec - 1) + 1e-15 * arma::eye(solalluvec.size(), solalluvec.size());
        arma::vec D = sum(G.cols(solalluvec - 1) * arma::pinv(V, 1e-5, "std") % G.cols(solalluvec - 1), 1);
        return min(D / G.diag());
      } else if (typestat == "CDMEANT") {
        arma::uvec solalluvec = as<arma::uvec>(wrap(soln_int));
        arma::uvec Targetuvec = as<arma::uvec>(wrap(Target));
        arma::mat V = G.submat(solalluvec - 1, solalluvec - 1) + R.submat(solalluvec - 1, solalluvec - 1) + 1e-15 * arma::eye(solalluvec.size(), solalluvec.size());
        arma::vec D = sum(G.submat(Targetuvec - 1, solalluvec - 1) * arma::pinv(V, 1e-5, "std") % G.submat(Targetuvec - 1, solalluvec - 1), 1);
        arma::mat Num = G.submat(Targetuvec - 1, Targetuvec - 1);
        return min(D / Num.diag());
      } else {
        return 0;
      }
    }
  }

  double GetStat(const Rcpp::NumericVector& soln_dbl, Rcpp::Function Stat) {

    if (typestat == "UD") {
      return as<double>(Stat(soln_dbl));
    } else if (typestat == "UDD") {
      return as<double>(Stat(soln_dbl, Data));
    } else {
      return 0;
    }

  }



};




class Population{
private:
  int npop;
  int nelite;
  int nchrom;
  IntegerVector chromsizes;
  CharacterVector chromtypes;

  STATCLASS StatClass;
  vector<double> FitnessVals;

  vector<IntegerMatrix> BOOL;
  vector<IntegerMatrix> OS;
  vector<IntegerMatrix> UOS;
  vector<IntegerMatrix> OMS;
  vector<IntegerMatrix> UOMS;
  vector<NumericMatrix> DBL;

  int nBOOL=0;
  int nOS=0;
  int nUOS=0;
  int nOMS=0;
  int nUOMS=0;
  int nDBL=0;

  IntegerVector nvecBOOL;
  IntegerVector nvecOS;
  IntegerVector nvecUOS;
  IntegerVector nvecOMS;
  IntegerVector nvecUOMS;
  IntegerVector nvecDBL;

  vector<IntegerVector> CandOS;
  vector<IntegerVector>  CandUOS;
  vector<IntegerVector>  CandOMS;
  vector<IntegerVector>  CandUOMS;
  vector<NumericVector> CandDBL;


public:
  //////////////
  IntegerVector CandBOOL={1,0};

  IntegerVector OrderPop;

  Population(){
  }

  void set_npop(int npop_){
    npop=npop_;
  }

  void set_nelite(int nelite_){
    nelite=nelite_;
  }
  void set_nchrom(int nchrom_){
    nchrom=nchrom_;
  }
  void set_chromsizes(IntegerVector chromsizes_){
    chromsizes=chromsizes_;
  }
  void set_chromtypes(CharacterVector chromtypes_){
    chromtypes=chromtypes_;
  }
  /////////////////
  int get_npop( ){
    return npop;
  }

  int get_nelite( ){
    return nelite;
  }
  int get_nchrom( ){
    return nchrom;
  }
  IntegerVector get_chromsizes( ){
    return chromsizes;
  }
  CharacterVector get_chromtypes(){
    return chromtypes;
  }



  /////////////

  void push_back_CandOS(IntegerVector Cand){
    CandOS.push_back(Cand);
  }
  void push_back_CandUOS(IntegerVector Cand){
    CandUOS.push_back(Cand);
  }
  void push_back_CandOMS(IntegerVector Cand){
    CandOMS.push_back(Cand);
  }
  void push_back_CandUOMS(IntegerVector Cand){
    CandUOMS.push_back(Cand);
  }
  void push_back_CandDBL(NumericVector Cand){
    CandDBL.push_back(Cand);
  }

 IntegerMatrix get_BOOL(int i){
    return BOOL.at(i);
  }

  IntegerMatrix get_OS(int i){
    return OS.at(i);
  }
  IntegerMatrix get_UOS(int i){
    return UOS.at(i);
  }
  IntegerMatrix get_OMS(int i){
    return OMS.at(i);
  }
  IntegerMatrix get_UOMS(int i){
    return UOMS.at(i);
  }
  NumericMatrix get_DBL(int i){
    return DBL.at(i);
  }


  //////////////////////////
  void InitRand(){
    int iiBOOL=0;
    for (int i=0;i<nchrom;i++)
      if (chromtypes[i]=="BOOL"){
        IntegerMatrix TempMat(chromsizes[i],npop+nelite);
        for (int j=0;j<(npop+nelite);j++){
          TempMat(_,j)=sample(CandBOOL, chromsizes[i],true);
        }
        BOOL.push_back(TempMat);
        iiBOOL++;
      }
      nBOOL=iiBOOL;
    int iiOS=0;
    for (int i=0;i<nchrom;i++)
      if (chromtypes[i]=="OS"){
        IntegerMatrix TempMat(chromsizes[i],npop+nelite);
        for (int j=0;j<(npop+nelite);j++){
          TempMat(_,j)=sample(CandOS.at(iiOS), chromsizes[i]);
        }
        OS.push_back(TempMat);
        iiOS++;
      }
      nOS=iiOS;
      int iiUOS=0;
      for (int i=0;i<nchrom;i++)
        if (chromtypes[i]=="UOS"){
          IntegerMatrix TempMat(chromsizes[i],(npop+nelite));
          for (int j=0;j<(npop+nelite);j++){
            TempMat(_,j)=sample(CandUOS.at(iiUOS), chromsizes[i]).sort();
          }
          UOS.push_back(TempMat);
          iiUOS++;
        }
        nUOS=iiUOS;
        int iiOMS=0;
        for (int i=0;i<nchrom;i++)
          if (chromtypes[i]=="OMS"){
            IntegerMatrix TempMat(chromsizes[i],(npop+nelite));
            for (int j=0;j<(npop+nelite);j++){
              TempMat(_,j)=sample(CandOMS.at(iiOMS), chromsizes[i], true);
            }
            OMS.push_back(TempMat);
            iiOMS++;
          }
          nOMS=iiOMS;
          int iiUOMS=0;
          for (int i=0;i<nchrom;i++)
            if (chromtypes[i]=="UOMS"){
              IntegerMatrix TempMat(chromsizes[i],(npop+nelite));
              for (int j=0;j<(npop+nelite);j++){
                TempMat(_,j)=sample(CandUOMS.at(iiUOMS), chromsizes[i], true).sort();
              }
              UOMS.push_back(TempMat);
              iiUOMS++;
            }
            nUOMS=iiUOMS;
            int iiDBL=0;
            for (int i=0;i<nchrom;i++)
              if (chromtypes[i]=="DBL"){
                NumericMatrix TempMat(chromsizes[i],(npop+nelite));
                for (int j=0;j<(npop+nelite);j++){
                  TempMat(_,j)=runif(chromsizes[i])*(CandDBL.at(iiDBL)(1)-CandDBL.at(iiDBL)(0))+CandDBL.at(iiDBL)(0);
                }
                DBL.push_back(TempMat);
                iiDBL++;
              }
              nDBL=iiDBL;
  }

  void Init(vector<IntegerMatrix> BOOL_, vector<IntegerMatrix> OS_,vector<IntegerMatrix> UOS_,vector<IntegerMatrix> OMS_,vector<IntegerMatrix> UOMS_, vector<NumericMatrix>DBL_){
    BOOL=BOOL_;
    OS=OS_;
    UOS=UOS_;
    OMS=OMS_;
    UOMS=UOMS_;
    DBL=DBL_;
  }




  void MoveInd(int from, int to){
    if (nBOOL>0){
      for (int i=0;i<nBOOL;i++){
        BOOL.at(i)(_, to)=BOOL.at(i)(_, from);
      }
    }
    if (nOS>0){
      for (int i=0;i<nOS;i++){
        OS.at(i)(_, to)=OS.at(i)(_, from);
      }
    }
    if (nUOS>0){
      for (int i=0;i<nUOS;i++){
        UOS.at(i)(_, to)=UOS.at(i)(_, from);
      }
    }
    if (nOMS>0){
      for (int i=0;i<nOMS;i++){
        OMS.at(i)(_, to)=OMS.at(i)(_, from);
      }
    }
    if (nUOMS>0){
      for (int i=0;i<nUOMS;i++){
        UOMS.at(i)(_, to)=UOMS.at(i)(_, from);
      }
    }
    if (nDBL>0){
      for (int i=0;i<nDBL;i++){
        DBL.at(i)(_, to)=DBL.at(i)(_, from);
      }
    }
  }



  void MakeCross(int p1, int p2, int child){

    if (nBOOL>0){
      for (int i=0;i<nBOOL;i++){
        int BOOLirows=BOOL.at(i).nrow();

        IntegerVector TempSol;
        for (int j=0;j<BOOLirows;j++){
          int sampleint=sample(2,1)[0];

          if (sampleint==1){
            TempSol.push_back(BOOL.at(i)(j,p1));
          } else {
            TempSol.push_back(BOOL.at(i)(j,p2));
          }
        }
        BOOL.at(i)(_,child)=TempSol;
      }
    }

    if (nOS>0){
      for (int i=0;i<nOS;i++){
        int OSirows=OS.at(i).nrow();

        IntegerVector TempSol;
        for (int j=0;j<OSirows;j++){
          int sampleint=sample(2,1)[0];

          if (sampleint==1 && !contains(TempSol,OS.at(i)(j,p1))){
            TempSol.push_back(OS.at(i)(j,p1));
          } else if (sampleint==2 && !contains(TempSol,OS.at(i)(j,p2))){
            TempSol.push_back(OS.at(i)(j,p2));
          } else {
            TempSol.push_back(sample(setdiff(CandOS.at(i),TempSol),1)[0]);
          }
        }
        OS.at(i)(_,child)=TempSol;
      }
    }

    if (nUOS>0){
      for (int i=0;i<nUOS;i++){
        int UOSirows=UOS.at(i).nrow();
        IntegerVector p1vec=UOS.at(i)(_,p1);
        IntegerVector p2vec=UOS.at(i)(_,p2);
        UOS.at(i)(_,child)=sample(union_(p1vec,p2vec),UOSirows, false).sort();
      }
    }

    if (nOMS>0){
      for (int i=0;i<nOMS;i++){
        int OMSirows=OMS.at(i).nrow();
        IntegerVector TempSol;
        for (int j=0;j<OMSirows;j++){
          int sampleint=sample(2,1)[0];
          if (sampleint==1){
            TempSol.push_back(OMS.at(i)(j,p1));
          } else{
            TempSol.push_back(OMS.at(i)(j,p2));
          }
        }
        OMS.at(i)(_,child)=TempSol;
      }
    }

    if (nUOMS>0){
      for (int i=0;i<nUOMS;i++){
        int UOMSirows=UOMS.at(i).nrow();
        IntegerVector TempSol;
        for (int j=0;j<UOMSirows;j++){
          TempSol.push_back(UOMS.at(i)(j,p1));
          TempSol.push_back(UOMS.at(i)(j,p2));
        }
        UOMS.at(i)(_,child)=sample(TempSol, UOMSirows, true).sort();
      }
    }

    if (nDBL>0){
      for (int i=0;i<nDBL;i++){
        int DBLirows=DBL.at(i).nrow();
        NumericVector TempSol;
        for (int j=0;j<DBLirows;j++){
          TempSol.push_back(.5*(DBL.at(i)(j,p1)+DBL.at(i)(j,p2)));
        }
        DBL.at(i)(_,child)=TempSol;
      }
    }

  }


  void  Mutate(int ind, double MUTPROB){


    if (nBOOL>0){
      for (int i=0;i<nBOOL;i++){
        int BOOLirows=BOOL.at(i).nrow();
        IntegerVector IndSol=BOOL.at(i)(_,ind);
        for (int j=0;j<BOOLirows;j++){
          double mutp=runif(1)(0);
          if (mutp<MUTPROB){
            IntegerVector totakeout;
            totakeout.push_back(IndSol(j));
            int replacement=sample(setdiff(CandBOOL,totakeout),1)(0);
            IndSol(j)=replacement;
          }
        }
        BOOL.at(i)(_,ind)=IndSol;
      }
    }

    if (nOS>0){
      for (int i=0;i<nOS;i++){
        int OSirows=OS.at(i).nrow();
        IntegerVector IndSol=OS.at(i)(_,ind);
        for (int j=0;j<OSirows;j++){
          double mutp=runif(1)(0);
          if (mutp<MUTPROB){
            IntegerVector totakeout;
            totakeout.push_back(IndSol(j));
            int replacement=sample(setdiff(CandOS.at(i),setdiff(IndSol,totakeout)),1)(0);
            IndSol(j)=replacement;
          }
        }
        double swapp=runif(1)(0);
        if (swapp<MUTPROB){
          int i1=sample(OSirows,1)(0)-1;
          int i2=sample(OSirows,1)(0)-1;
          int ii1=IndSol(i1);
          int ii2=IndSol(i2);

          IndSol(i1)= ii2;
          IndSol(i2)= ii1;

        }

        double slidep=runif(1)(0);
        if (slidep<MUTPROB){
          int movedirection=sample(2,1)(0);
          if (movedirection==1){
            std::rotate(IndSol.begin(), IndSol.begin() + 1, IndSol.end());
          } else{
            std::rotate(IndSol.begin(), IndSol.end(), IndSol.end());
          }
        }
        OS.at(i)(_,ind)=IndSol;
      }

    }

    if (nUOS>0){
      for (int i=0;i<nUOS;i++){
        int UOSirows=UOS.at(i).nrow();
        IntegerVector IndSol=UOS.at(i)(_,ind);
        for (int j=0;j<UOSirows;j++){
          double mutp=runif(1)(0);
          if (mutp<MUTPROB){
            int replacement=sample(setdiff(CandUOS.at(i),IndSol),1)(0);
            IndSol(j)=replacement;
          }
        }
        UOS.at(i)(_,ind)=IndSol.sort();
      }
    }

    if (nOMS>0){
      for (int i=0;i<nOMS;i++){
        int OMSirows=OMS.at(i).nrow();
        IntegerVector IndSol=OMS.at(i)(_,ind);
        for (int j=0;j<OMSirows;j++){
          double mutp=runif(1)(0);
          if (mutp<MUTPROB){
            int replacement=sample(CandOMS.at(i),1)(0);
            IndSol(j)=replacement;
          }
        }
        double swapp=runif(1)(0);
        if (swapp<MUTPROB){
          int i1=sample(OMSirows,1)(0)-1;
          int i2=sample(OMSirows,1)(0)-1;
          int ii1=IndSol(i1);
          int ii2=IndSol(i2);

          IndSol(i1)= ii2;
          IndSol(i2)= ii1;

        }

        double slidep=runif(1)(0);
        if (slidep<MUTPROB){
          int movedirection=sample(2,1)(0);
          if (movedirection==1){
            std::rotate(IndSol.begin(), IndSol.begin() + 1, IndSol.end());
          } else{
            std::rotate(IndSol.begin(), IndSol.end(), IndSol.end());
          }
        }
        OMS.at(i)(_,ind)=IndSol;
      }

    }


    if (nUOMS>0){
      for (int i=0;i<nUOMS;i++){
        int UOMSirows=UOMS.at(i).nrow();
        IntegerVector IndSol=UOMS.at(i)(_,ind);
        for (int j=0;j<UOMSirows;j++){
          double mutp=runif(1)(0);
          if (mutp<MUTPROB){
            int replacement=sample(CandUOMS.at(i),1)(0);
            IndSol(j)=replacement;
          }
        }
        UOMS.at(i)(_,ind)=IndSol.sort();
      }
    }
    if (nDBL>0){
      for (int i=0;i<nDBL;i++){
        int DBLirows=DBL.at(i).nrow();
        NumericVector IndSol=DBL.at(i)(_,ind);
        for (int j=0;j<DBLirows;j++){
          double mutp=runif(1)(0);
          double tempsold;
          if (mutp<MUTPROB){
            tempsold=IndSol[j]+rnorm(1)(0)*sd(DBL.at(i)(j,_))*.01;
            if (tempsold<CandDBL.at(i)(0)){tempsold=tempsold<CandDBL.at(i)(0);}
            if (tempsold>CandDBL.at(i)(1)){tempsold=tempsold<CandDBL.at(i)(1);}

            IndSol[j]=tempsold;
          }
        }
        DBL.at(i)(_,ind)=IndSol;
      }
    }

  }



/*
void  MutatetowardsNTotal(int ind, int ninG=1, int ntotal=0){
  if (ntotal>0){
  IntegerVector soln=getSolnInt(ind);
  IntegerVector AllIndsinSoln=clone(soln);
  if (ninG>1){
    for (int i=0;i<AllIndsinSoln.length();i++){
      int  tempint=AllIndsinSoln(i)%ninG;
      if (tempint==0){tempint=ninG;}
       AllIndsinSoln(i)=tempint;

    }

  }

  if (unique(AllIndsinSoln).length()>ntotal){
  int samplepos=sample(AllIndsinSoln.length(),1)(0);
    IntegerVector totakeout;
    totakeout.push_back(AllIndsinSoln[samplepos-1]);
  IntegerVector CumSumchromsizes=cumsum(chromsizes);

  IntegerVector smallerpart=CumSumchromsizes[CumSumchromsizes<samplepos];
  int setint= smallerpart.length()+1;

  int toreplace =sample(setdiff(AllIndsinSoln, totakeout),1)(0);
  if (ninG>1){
  toreplace=toreplace+(setint-1)*ninG;
  }
  soln[samplepos-1]=toreplace;
  putSolnInt(ind, soln);

  }
  }
}
*/



  IntegerVector getSolnInt(int ind){
    IntegerVector soln;
    int iBOOL=0;
    int iOS=0;
    int iUOS=0;
    int iOMS=0;
    int iUOMS=0;
    for (int i=0;i<chromsizes.length();i++){

      if (chromtypes[i]=="BOOL"){
        for (int j=0;j<chromsizes[i];j++){
          soln.push_back(BOOL.at(iBOOL)(j,ind));
        }
        iBOOL++;
      }
      if (chromtypes[i]=="OS"){
        for (int j=0;j<chromsizes[i];j++){
          soln.push_back(OS.at(iOS)(j,ind));
        }
        iOS++;
      }
      if (chromtypes[i]=="UOS"){
        for (int j=0;j<chromsizes[i];j++){

          soln.push_back(UOS.at(iUOS)(j,ind));
        }
        iUOS++;
      }
      if (chromtypes[i]=="OMS"){
        for (int j=0;j<chromsizes[i];j++){
          soln.push_back(OMS.at(iOMS)(j,ind));
        }
        iOMS++;
      }
      if (chromtypes[i]=="UOMS"){
        for (int j=0;j<chromsizes[i];j++){
          soln.push_back(UOMS.at(iUOMS)(j,ind));
        }
        iUOMS++;
      }
    }
    return soln;
  }



  void putSolnInt(int ind, IntegerVector soln){
    int iBOOL=0;
    int iOS=0;
    int iUOS=0;
    int iOMS=0;
    int iUOMS=0;
    int jj=0;
    for (int i=0;i<chromsizes.length();i++){
      if (chromtypes[i]=="BOOL"){
        for (int j=0;j<chromsizes[i];j++){
          BOOL.at(iBOOL)(j,ind)=soln(jj);
          jj++;
        }
        iBOOL++;
      }

      if (chromtypes[i]=="OS"){
        for (int j=0;j<chromsizes[i];j++){
          OS.at(iOS)(j,ind)=soln(jj);
          jj++;
        }
        iOS++;
      }
      if (chromtypes[i]=="UOS"){
        for (int j=0;j<chromsizes[i];j++){

          UOS.at(iUOS)(j,ind)=soln[jj];
          jj++;
        }
        IntegerVector TempVec=UOS.at(iUOS)(_,ind);
        TempVec.sort();
        UOS.at(iUOS)(_,ind)=TempVec;
        iUOS++;
      }
      if (chromtypes[i]=="OMS"){
        for (int j=0;j<chromsizes[i];j++){
          OMS.at(iOMS)(j,ind)=soln[jj];
          jj++;
        }
        iOMS++;
      }
      if (chromtypes[i]=="UOMS"){
        for (int j=0;j<chromsizes[i];j++){
          UOMS.at(iUOMS)(j,ind)=soln[jj];
          jj++;
        }
        IntegerVector TempVec=UOMS.at(iUOMS)(_,ind);
        TempVec.sort();
        UOS.at(iUOMS)(_,ind)=TempVec;
        iUOMS++;
      }
    }
  }













  NumericVector getSolnDbl(int ind){
    NumericVector soln;
    int iDBL=0;
    for (int i=0;i<chromsizes.length();i++){
      if (chromtypes[i]=="DBL"){
        for (int j=0;j<chromsizes[i];j++){
          soln.push_back(DBL.at(iDBL)(j,ind));
        }
        iDBL++;
      }
    }
    return soln;
  }





  void putSolnDbl(int ind, NumericVector soln){
    int iDBL=0;
    int jj=0;
    for (int i=0;i<chromsizes.length();i++){
      if (chromtypes[i]=="DBL"){
        for (int j=0;j<chromsizes[i];j++){
          DBL.at(iDBL)(j,ind)=soln[jj];
          jj++;
        }
        iDBL++;
      }
    }
  }

  ///////////


  void set_STATCLASS(STATCLASS STATC_){
    StatClass=STATC_;
  }


  void init_Fitness(){
    FitnessVals =vector<double>(npop+nelite);
  }

  void set_Fitness(int ind, Function Stat){
    if (nDBL>0){
      FitnessVals[ind]=StatClass.GetStat(getSolnInt(ind),getSolnDbl(ind), Stat);
    } else {
      FitnessVals[ind]=StatClass.GetStat(getSolnInt(ind), Stat);
    }
  }
  void set_Fitness(int ind, double val){
    FitnessVals[ind]=val;
  }


  double get_Fitness(int ind, Function Stat){
    double out;
    if (nDBL>0){
      out=StatClass.GetStat(getSolnInt(ind),getSolnDbl(ind), Stat);
    } else {
      out=StatClass.GetStat(getSolnInt(ind), Stat);
    }
    return out;
  }


  double get_Fitness(IntegerVector soln_int, NumericVector soln_dbl, Function Stat){
    double out;
    if (nDBL>0){
      out=StatClass.GetStat(soln_int,soln_dbl, Stat);
    } else {
      out=StatClass.GetStat(soln_int, Stat);
    }
    return StatClass.GetStat(soln_int,soln_dbl, Stat);
  }

  vector<double> get_Fitness(){
    vector<double> out={FitnessVals.begin(),FitnessVals.begin()+npop};
    return out;
  }
  double get_Fitness(int i){
    return FitnessVals[i];
  }



};











/////////////////////////////////////////////////////////////////////////////////////



/////////////////////////////////////////////////////////////////////////////////////////////////////////
class OutTrainSel {
public:
  Rcpp::IntegerVector Best_Sol_Int;
  Rcpp::NumericVector Best_Sol_DBL;
  double Best_Val = -1;
  NumericVector maxvec;
  int convergence = -1;


  OutTrainSel() {
    Best_Sol_Int.push_back(-1);
    Best_Sol_DBL.push_back(-1);
    Best_Val = -1;
    maxvec.push_back(-1);
    convergence = -1;
  }

  OutTrainSel(List Data,
              List CANDIDATES,
              Rcpp::IntegerVector setsizes,
              Rcpp::CharacterVector settypes,
              Rcpp::Function Stat,
              bool CD,
              Rcpp::IntegerVector Target,
              List control,
              int ntotal) {


    int NPOP = as<int>(control["npop"]);
    int NELITE = as<int>(control["nelite"]);
    int NITERGA = as<int>(control["niterations"]);
    double MUTPROB = as<double>(control["mutprob"]);
    int NITERSANN = as<int>(control["niterSANN"]);
    double STEPSANN = as<double>(control["stepSANN"]);
    double TOLCONV = as<double>(control["tolconv"]);
    int MINITBEFSTOP = as<int>(control["minitbefstop"]);
    bool PROGRESS = as<bool>(control["progress"]);


    bool maxiter = false;
    bool minitcond = false;
    double maxmeans, minmeans, meansdiff;

    bool CheckData = false;
    bool CheckDataMM = false;
    if (Data.size() > 0) {
      CheckData = true;
    }

    if (CheckData) {
      if (Data.containsElementNamed("class")) {
        string DataClass = as<string>(Data["class"]);
        string MMClass = "TrainSel_Data";
        if (DataClass == MMClass) {
          CheckDataMM = true;
        }
      }
    }

    bool CheckTarget = Target.length() > 0;
    /////errors
    if (CheckData & !CheckDataMM) {
      if (CD) {
        stop("error");
      }
    }
    /////errors
    if (!CheckData) {
      if (CD) {
        stop("error");
      }
    }
    STATCLASS STATc;

    ///
    if (!CheckData) {
      if (!CD) {
        STATc = STATCLASS();
      }
    }

    if (CheckData) {
      if (!CD) {
        STATc = STATCLASS(Data);
      }
    }
    if (CheckData & CheckDataMM) {
      if (CD) {
        if (Data.containsElementNamed("X") & (CheckTarget)) {
          arma::mat X = as<arma::mat>(Data["X"]);
          arma::mat G = as<arma::mat>(Data["G"]);
          arma::mat R = as<arma::mat>(Data["R"]);
          STATc = STATCLASS(Target, X, G, R);
          STATc.setAllinG(as<int>(Data["Nind"]));
        }
        if (!Data.containsElementNamed("X") & (CheckTarget)) {
          arma::mat G = as<arma::mat>(Data["G"]);
          arma::mat R = as<arma::mat>(Data["R"]);
          STATc = STATCLASS(Target, G, R);
          STATc.setAllinG(as<int>(Data["Nind"]));

        }
        if (Data.containsElementNamed("X") & !(CheckTarget)) {

          arma::mat X = as<arma::mat>(Data["X"]);
          arma::mat G = as<arma::mat>(Data["G"]);
          arma::mat R = as<arma::mat>(Data["R"]);
          STATc = STATCLASS(X, G, R);
          STATc.setAllinG(as<int>(Data["Nind"]));

        }
        if (!Data.containsElementNamed("X")&!(CheckTarget)) {
          const arma::mat G = as<arma::mat>(Data["G"]);
          const arma::mat R = as<arma::mat>(Data["R"]);
          STATc = STATCLASS(G, R);
          STATc.setAllinG(as<int>(Data["Nind"]));
        }

      } else {
        STATc = STATCLASS(Data);
      }
    }

    if (ntotal>0){
      STATc.setntotal(ntotal);
    }



    ////////////////


    Population pop;
    pop.set_npop(NPOP);

    pop.set_nelite(NELITE);

    pop.set_nchrom(setsizes.length());
    pop.set_chromsizes(setsizes);
    pop.set_chromtypes(settypes);
    for (int i=0;i<pop.get_nchrom();i++){
      if (settypes[i]=="OS"){
        pop.push_back_CandOS(as<IntegerVector>(CANDIDATES[i]));
      }
      if (settypes[i]=="UOS"){
        pop.push_back_CandUOS(as<IntegerVector>(CANDIDATES[i]));

      }
      if (settypes[i]=="OMS"){
        pop.push_back_CandOMS(as<IntegerVector>(CANDIDATES[i]));
      }
      if (settypes[i]=="UOMS"){
        pop.push_back_CandUOMS(as<IntegerVector>(CANDIDATES[i]));
      }
      if (settypes[i]=="DBL"){
        pop.push_back_CandDBL(as<NumericVector>(CANDIDATES[i]));
      }
    }



    pop.set_STATCLASS(STATc);
    pop.init_Fitness();

    pop.InitRand();

    for (int i=0;i<pop.get_npop();i++){
      pop.set_Fitness(i, Stat);
    }
    int Generation = 0;
    Progress p(NITERGA, PROGRESS);
    int tryCount=0;
    while (!maxiter & !minitcond) {
      NumericVector GoodSols=maxvec[maxvec>=-1e+10];
      if (GoodSols.length()>=1){
      p.increment();
      Generation++;
      } else {
        tryCount++;
      }
      if (tryCount==10000){
        Rcout << "No feasible solution found in 10000 (warmup) iterations! \n Try restart??." << std::endl;
      }
      R_CheckUserInterrupt();
      if (Generation > MINITBEFSTOP) {
        maxmeans = max(maxvec[Range(maxvec.length() - MINITBEFSTOP - 1, maxvec.length() - 1)]);
        minmeans = min(maxvec[Range(maxvec.length() - MINITBEFSTOP - 1, maxvec.length() - 1)]);
        meansdiff = maxmeans - minmeans;
        if (meansdiff < TOLCONV) {
          Rcout << "Convergence Achieved \n (no improv in the last 'minitbefstop' iters)." << std::endl;
          minitcond = true;
        }
      }
      if (Generation == NITERGA) {
        Rcout << "Maximum number of iterations reached." << std::endl;
        maxiter = true;
      }

      pop.OrderPop=orderRcpp(pop.get_Fitness())-1;
      IntegerVector bestsols(pop.get_nelite());
      std::copy(pop.OrderPop.end()-pop.get_nelite(), pop.OrderPop.end(),bestsols.begin());
      IntegerVector worstsols(pop.get_npop()-pop.get_nelite());
      std::copy(pop.OrderPop.begin(), pop.OrderPop.begin()+worstsols.length(),worstsols.begin());

      /////SANN
      if ( NITERSANN> 0) {
        for (int i = 0; i < pop.get_nelite(); i++) {
          pop.MoveInd(bestsols[i],worstsols[0]);
          pop.set_Fitness(worstsols[0],pop.get_Fitness(bestsols[i])); //sb
          pop.MoveInd(bestsols[i],worstsols[1]);
          pop.set_Fitness(worstsols[1],pop.get_Fitness(bestsols[i])); //sn
          pop.MoveInd(bestsols[i],worstsols[2]);
          pop.set_Fitness(worstsols[2],pop.get_Fitness(bestsols[i])); //sc
          double Temp;


          double f_b=pop.get_Fitness(worstsols[0]);
          double f_c = f_b;
          double f_n = f_c;

          GetRNGstate();
          for (int k = 0; k < NITERSANN - 1; k++) {
            R_CheckUserInterrupt();

            Temp = powf(1 - STEPSANN / log(NITERSANN + 1), k);
            pop.Mutate(worstsols[1], MUTPROB);

            pop.set_Fitness(worstsols[1], Stat);
            f_n=pop.get_Fitness(worstsols[1]);
            if ((f_n > f_c) | (runif(1, 0, 1)(0) < exp(-(f_n - f_c) / Temp))) {
              pop.MoveInd(worstsols[1],worstsols[2]);
              pop.set_Fitness(worstsols[2],pop.get_Fitness(worstsols[1]));
              f_c = f_n;
            }
            if (f_n > f_b) {
              pop.MoveInd(worstsols[1],worstsols[0]);
              pop.set_Fitness(worstsols[0],pop.get_Fitness(worstsols[1]));
              f_b = pop.get_Fitness(worstsols[0]);
            }
            if (Temp < 1e-15) {
              break;
            }
          }

          PutRNGstate();

          pop.MoveInd(worstsols[0],bestsols[i]);

          pop.set_Fitness(worstsols[0],pop.get_Fitness(bestsols[i]));

        }
      }

      int bestsol= pop.OrderPop(pop.get_npop()-1);


      maxvec.push_back(pop.get_Fitness(bestsol));

      for (int i=0;i<worstsols.length();i++){
        int p1=sample(bestsols,1)(0);
        int p2=sample(bestsols,1)(0);
        pop.MakeCross(p1,p2,worstsols[i]);
        pop.Mutate(worstsols[i],MUTPROB);

        pop.set_Fitness(worstsols[i], Stat);
      }
    }

    pop.OrderPop=orderRcpp(pop.get_Fitness())-1;
    int bestsol= pop.OrderPop(pop.get_npop()-1);
    Best_Sol_Int=pop.getSolnInt(bestsol);
    Best_Sol_DBL=pop.getSolnDbl(bestsol);
    Best_Val=pop.get_Fitness(bestsol);

    if (minitcond) {
      convergence = 1;
    }
    if (maxiter) {
      convergence = 0;
    }


  }

  List getSol() {
    return Rcpp::List::create(Rcpp::Named("BestSol_int") = Best_Sol_Int,
                              Rcpp::Named("BestSol_DBL") =  Best_Sol_DBL,
                              Rcpp::Named("BestVal") =Best_Val,
                              Rcpp::Named("maxvec") = maxvec,
                              Rcpp::Named("convergence") = convergence);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////






// [[Rcpp::export]]
List TrainSelC(List Data,
               List CANDIDATES,
               Rcpp::IntegerVector setsizes,
               Rcpp::CharacterVector settypes,
               Rcpp::Function Stat,
               bool CD,
               Rcpp::IntegerVector Target,
               List control,
               int ntotal) {

  OutTrainSel out(Data,
                  CANDIDATES,
                  setsizes,
                  settypes,
                  Stat,
                  CD,
                  Target,
                  control,
                  ntotal);
  return out.getSol();

}


/////////////////////////











struct STATCLASSMOO {
public:
  Rcpp::List Data = Rcpp::List::create();
  std::string typestat;
  int numstat;
  STATCLASSMOO(const Rcpp::List& Data_) {
    Data = Data_;
    typestat = "UDD";
  }

  STATCLASSMOO() {
    typestat = "UD";
  }

  void set_numstat(int numstat_){
    numstat=numstat_;
  }

  int get_numstat(){
    return numstat;
  }



  IntegerVector getInds(IntegerVector soln_int){
    return soln_int;
  }

  NumericVector GetStat(const Rcpp::IntegerVector& soln_int, const Rcpp::NumericVector& soln_dbl, Rcpp::Function Stat) {

    NumericVector out;
    if (typestat == "UD") {
      out= as<NumericVector>(Stat(soln_int, soln_dbl));
    }

    if (typestat == "UDD") {
      out=as<NumericVector>(Stat(soln_int, soln_dbl, Data));
    }
    return out;
  }


  NumericVector GetStat(const Rcpp::IntegerVector& soln_int, Rcpp::Function Stat) {
    NumericVector out;
    if (typestat == "UD") {
      out= as<NumericVector>(Stat(soln_int));
    }
    if (typestat == "UDD") {
      out= as<NumericVector>(Stat(soln_int, Data));
    }
    return out;

  }

  NumericVector GetStat(const Rcpp::NumericVector& soln_dbl, Rcpp::Function Stat) {
    NumericVector out;
    if (typestat == "UD") {
      out= as<NumericVector>(Stat(soln_dbl));
    }
    if (typestat == "UDD") {
      out = as<NumericVector>(Stat(soln_dbl, Data));
    }
    return out;
  }
};




class PopulationMOO{
private:
  int nchrom;
  int npop;

  IntegerVector chromsizes;
  CharacterVector chromtypes;

  STATCLASSMOO StatClass;
  NumericMatrix FitnessVals;
  vector<IntegerMatrix> BOOL;
  vector<IntegerMatrix> OS;
  vector<IntegerMatrix> UOS;
  vector<IntegerMatrix> OMS;
  vector<IntegerMatrix> UOMS;
  vector<NumericMatrix> DBL;
  int nBOOL=0;
  int nOS=0;
  int nUOS=0;
  int nOMS=0;
  int nUOMS=0;
  int nDBL=0;

  IntegerVector nvecBOOL;
  IntegerVector nvecOS;
  IntegerVector nvecUOS;
  IntegerVector nvecOMS;
  IntegerVector nvecUOMS;
  IntegerVector nvecDBL;

  vector<IntegerVector> CandOS;
  vector<IntegerVector>  CandUOS;
  vector<IntegerVector>  CandOMS;
  vector<IntegerVector>  CandUOMS;
  vector<NumericVector> CandDBL;

public:
  //////////////
  IntegerVector CandBOOL={1,0};


  PopulationMOO(){
  }

  void set_npop(int npop_){
    npop=npop_;
  }


  void set_nchrom(int nchrom_){
    nchrom=nchrom_;
  }
  void set_chromsizes(IntegerVector chromsizes_){
    chromsizes=chromsizes_;
  }
  void set_chromtypes(CharacterVector chromtypes_){
    chromtypes=chromtypes_;
  }
  /////////////////
  int get_npop( ){
    return npop;
  }

  int get_nchrom( ){
    return nchrom;
  }
  IntegerVector get_chromsizes( ){
    return chromsizes;
  }
  CharacterVector get_chromtypes(){
    return chromtypes;
  }



  /////////////

  void push_back_CandOS(IntegerVector Cand){
    CandOS.push_back(Cand);
  }
  void push_back_CandUOS(IntegerVector Cand){
    CandUOS.push_back(Cand);
  }
  void push_back_CandOMS(IntegerVector Cand){
    CandOMS.push_back(Cand);
  }
  void push_back_CandUOMS(IntegerVector Cand){
    CandUOMS.push_back(Cand);
  }
  void push_back_CandDBL(NumericVector Cand){
    CandDBL.push_back(Cand);
  }

  IntegerMatrix get_BOOL(int i){
    return BOOL.at(i);
  }

  IntegerMatrix get_OS(int i){
    return OS.at(i);
  }
  IntegerMatrix get_UOS(int i){
    return UOS.at(i);
  }
  IntegerMatrix get_OMS(int i){
    return OMS.at(i);
  }
  IntegerMatrix get_UOMS(int i){
    return UOMS.at(i);
  }
  NumericMatrix get_DBL(int i){
    return DBL.at(i);
  }


  //////////////////////////
  void InitRand(){
    int iiBOOL=0;
    for (int i=0;i<nchrom;i++)
      if (chromtypes[i]=="BOOL"){
        IntegerMatrix TempMat(chromsizes[i],npop);
        for (int j=0;j<(npop);j++){
          TempMat(_,j)=sample(CandBOOL, chromsizes[i], true);
        }
        BOOL.push_back(TempMat);
        iiBOOL++;
      }
      nBOOL=iiBOOL;

    int iiOS=0;
    for (int i=0;i<nchrom;i++)
      if (chromtypes[i]=="OS"){
        IntegerMatrix TempMat(chromsizes[i],npop);
        for (int j=0;j<(npop);j++){
          TempMat(_,j)=sample(CandOS.at(iiOS), chromsizes[i]);
        }
        OS.push_back(TempMat);
        iiOS++;
      }
      nOS=iiOS;
      int iiUOS=0;
      for (int i=0;i<nchrom;i++)
        if (chromtypes[i]=="UOS"){
          IntegerMatrix TempMat(chromsizes[i],(npop));
          for (int j=0;j<(npop);j++){
            TempMat(_,j)=sample(CandUOS.at(iiUOS), chromsizes[i]).sort();
          }
          UOS.push_back(TempMat);
          iiUOS++;
        }
        nUOS=iiUOS;
        int iiOMS=0;
        for (int i=0;i<nchrom;i++)
          if (chromtypes[i]=="OMS"){
            IntegerMatrix TempMat(chromsizes[i],(npop));
            for (int j=0;j<(npop);j++){
              TempMat(_,j)=sample(CandOMS.at(iiOMS), chromsizes[i], true);
            }
            OMS.push_back(TempMat);
            iiOMS++;
          }
          nOMS=iiOMS;
          int iiUOMS=0;
          for (int i=0;i<nchrom;i++)
            if (chromtypes[i]=="UOMS"){
              IntegerMatrix TempMat(chromsizes[i],(npop));
              for (int j=0;j<(npop);j++){
                TempMat(_,j)=sample(CandUOMS.at(iiUOMS), chromsizes[i], true).sort();
              }
              UOMS.push_back(TempMat);
              iiUOMS++;
            }
            nUOMS=iiUOMS;
            int iiDBL=0;
            for (int i=0;i<nchrom;i++)
              if (chromtypes[i]=="DBL"){
                NumericMatrix TempMat(chromsizes[i],(npop));
                for (int j=0;j<(npop);j++){
                  TempMat(_,j)=runif(chromsizes[i])*(CandDBL.at(iiDBL)(1)-CandDBL.at(iiDBL)(0))+CandDBL.at(iiDBL)(0);
                }
                DBL.push_back(TempMat);
                iiDBL++;
              }
              nDBL=iiDBL;
  }

  void Init(vector<IntegerMatrix> BOOL_,vector<IntegerMatrix> OS_,vector<IntegerMatrix> UOS_,vector<IntegerMatrix> OMS_,vector<IntegerMatrix> UOMS_, vector<NumericMatrix>DBL_){
    BOOL=BOOL_;
    OS=OS_;
    UOS=UOS_;
    OMS=OMS_;
    UOMS=UOMS_;
    DBL=DBL_;
  }




  void MoveInd(int from, int to){
    if (nBOOL>0){
      for (int i=0;i<nBOOL;i++){
        BOOL.at(i)(_, to)=BOOL.at(i)(_, from);
      }
    }

    if (nOS>0){
      for (int i=0;i<nOS;i++){
        OS.at(i)(_, to)=OS.at(i)(_, from);
      }
    }
    if (nUOS>0){
      for (int i=0;i<nUOS;i++){
        UOS.at(i)(_, to)=UOS.at(i)(_, from);
      }
    }
    if (nOMS>0){
      for (int i=0;i<nOMS;i++){
        OMS.at(i)(_, to)=OMS.at(i)(_, from);
      }
    }
    if (nUOMS>0){
      for (int i=0;i<nUOMS;i++){
        UOMS.at(i)(_, to)=UOMS.at(i)(_, from);
      }
    }
    if (nDBL>0){
      for (int i=0;i<nDBL;i++){
        DBL.at(i)(_, to)=DBL.at(i)(_, from);
      }
    }
  }



  void MakeCross(int p1, int p2, int child){

    if (nBOOL>0){
      for (int i=0;i<nBOOL;i++){
        int BOOLirows=BOOL.at(i).nrow();

        IntegerVector TempSol;
        for (int j=0;j<BOOLirows;j++){
          int sampleint=sample(2,1)[0];

          if (sampleint==1){
            TempSol.push_back(BOOL.at(i)(j,p1));
          } else {
            TempSol.push_back(BOOL.at(i)(j,p2));
          }
        }
        BOOL.at(i)(_,child)=TempSol;
      }
    }


    if (nOS>0){
      for (int i=0;i<nOS;i++){
        int OSirows=OS.at(i).nrow();

        IntegerVector TempSol;
        for (int j=0;j<OSirows;j++){
          int sampleint=sample(2,1)[0];

          if (sampleint==1 && !contains(TempSol,OS.at(i)(j,p1))){
            TempSol.push_back(OS.at(i)(j,p1));
          } else if (sampleint==2 && !contains(TempSol,OS.at(i)(j,p2))){
            TempSol.push_back(OS.at(i)(j,p2));
          } else {
            TempSol.push_back(sample(setdiff(CandOS.at(i),TempSol),1)[0]);
          }
        }
        OS.at(i)(_,child)=TempSol;
      }
    }

    if (nUOS>0){
      for (int i=0;i<nUOS;i++){
        int UOSirows=UOS.at(i).nrow();
        IntegerVector p1vec=UOS.at(i)(_,p1);
        IntegerVector p2vec=UOS.at(i)(_,p2);
        UOS.at(i)(_,child)=sample(union_(p1vec,p2vec),UOSirows, false).sort();
      }
    }

    if (nOMS>0){
      for (int i=0;i<nOMS;i++){
        int OMSirows=OMS.at(i).nrow();
        IntegerVector TempSol;
        for (int j=0;j<OMSirows;j++){
          int sampleint=sample(2,1)[0];
          if (sampleint==1){
            TempSol.push_back(OMS.at(i)(j,p1));
          } else{
            TempSol.push_back(OMS.at(i)(j,p2));
          }
        }
        OMS.at(i)(_,child)=TempSol;
      }
    }

    if (nUOMS>0){
      for (int i=0;i<nUOMS;i++){
        int UOMSirows=UOMS.at(i).nrow();
        IntegerVector TempSol;
        for (int j=0;j<UOMSirows;j++){
          TempSol.push_back(UOMS.at(i)(j,p1));
          TempSol.push_back(UOMS.at(i)(j,p2));
        }
        UOMS.at(i)(_,child)=sample(TempSol, UOMSirows, true).sort();
      }
    }

    if (nDBL>0){
      for (int i=0;i<nDBL;i++){
        int DBLirows=DBL.at(i).nrow();
        NumericVector TempSol;
        for (int j=0;j<DBLirows;j++){
          TempSol.push_back(.5*(DBL.at(i)(j,p1)+DBL.at(i)(j,p2)));
        }
        DBL.at(i)(_,child)=TempSol;
      }
    }

  }


  void  Mutate(int ind, double MUTPROB){

    if (nBOOL>0){
      for (int i=0;i<nBOOL;i++){
        int BOOLirows=BOOL.at(i).nrow();
        IntegerVector IndSol=BOOL.at(i)(_,ind);
        for (int j=0;j<BOOLirows;j++){
          double mutp=runif(1)(0);
          if (mutp<MUTPROB){
            IntegerVector totakeout;
            totakeout.push_back(IndSol(j));
            int replacement=sample(setdiff(CandBOOL,totakeout),1)(0);
            IndSol(j)=replacement;
          }
        }

        BOOL.at(i)(_,ind)=IndSol;
      }

    }


    if (nOS>0){
      for (int i=0;i<nOS;i++){
        int OSirows=OS.at(i).nrow();
        IntegerVector IndSol=OS.at(i)(_,ind);
        for (int j=0;j<OSirows;j++){
          double mutp=runif(1)(0);
          if (mutp<MUTPROB){
            IntegerVector totakeout;
            totakeout.push_back(IndSol(j));
            int replacement=sample(setdiff(CandOS.at(i),setdiff(IndSol,totakeout)),1)(0);
            IndSol(j)=replacement;
          }
        }
        double swapp=runif(1)(0);
        if (swapp<MUTPROB){
          int i1=sample(OSirows,1)(0)-1;
          int i2=sample(OSirows,1)(0)-1;
          int ii1=IndSol(i1);
          int ii2=IndSol(i2);

          IndSol(i1)= ii2;
          IndSol(i2)= ii1;

        }

        double slidep=runif(1)(0);
        if (slidep<MUTPROB){
          int movedirection=sample(2,1)(0);
          if (movedirection==1){
            std::rotate(IndSol.begin(), IndSol.begin() + 1, IndSol.end());
          } else{
            std::rotate(IndSol.begin(), IndSol.end(), IndSol.end());
          }
        }
        OS.at(i)(_,ind)=IndSol;
      }

    }

    if (nUOS>0){
      for (int i=0;i<nUOS;i++){
        int UOSirows=UOS.at(i).nrow();
        IntegerVector IndSol=UOS.at(i)(_,ind);
        for (int j=0;j<UOSirows;j++){
          double mutp=runif(1)(0);
          if (mutp<MUTPROB){
            int replacement=sample(setdiff(CandUOS.at(i),IndSol),1)(0);
            IndSol(j)=replacement;
          }
        }
        UOS.at(i)(_,ind)=IndSol.sort();
      }
    }

    if (nOMS>0){
      for (int i=0;i<nOMS;i++){
        int OMSirows=OMS.at(i).nrow();
        IntegerVector IndSol=OMS.at(i)(_,ind);
        for (int j=0;j<OMSirows;j++){
          double mutp=runif(1)(0);
          if (mutp<MUTPROB){
            int replacement=sample(CandOMS.at(i),1)(0);
            IndSol(j)=replacement;
          }
        }
        double swapp=runif(1)(0);
        if (swapp<MUTPROB){
          int i1=sample(OMSirows,1)(0)-1;
          int i2=sample(OMSirows,1)(0)-1;
          int ii1=IndSol(i1);
          int ii2=IndSol(i2);

          IndSol(i1)= ii2;
          IndSol(i2)= ii1;

        }

        double slidep=runif(1)(0);
        if (slidep<MUTPROB){
          int movedirection=sample(2,1)(0);
          if (movedirection==1){
            std::rotate(IndSol.begin(), IndSol.begin() + 1, IndSol.end());
          } else{
            std::rotate(IndSol.begin(), IndSol.end(), IndSol.end());
          }
        }
        OMS.at(i)(_,ind)=IndSol;
      }

    }


    if (nUOMS>0){
      for (int i=0;i<nUOMS;i++){
        int UOMSirows=UOMS.at(i).nrow();
        IntegerVector IndSol=UOMS.at(i)(_,ind);
        for (int j=0;j<UOMSirows;j++){
          double mutp=runif(1)(0);
          if (mutp<MUTPROB){
            int replacement=sample(CandUOMS.at(i),1)(0);
            IndSol(j)=replacement;
          }
        }
        UOMS.at(i)(_,ind)=IndSol.sort();
      }
    }
    if (nDBL>0){
      for (int i=0;i<nDBL;i++){
        int DBLirows=DBL.at(i).nrow();
        NumericVector IndSol=DBL.at(i)(_,ind);
        for (int j=0;j<DBLirows;j++){
          double mutp=runif(1)(0);
          double tempsold;
          if (mutp<MUTPROB){
            tempsold=IndSol[j]+rnorm(1)(0)*sd(DBL.at(i)(j,_))*.01;
            if (tempsold<CandDBL.at(i)(0)){tempsold=tempsold<CandDBL.at(i)(0);}
            if (tempsold>CandDBL.at(i)(1)){tempsold=tempsold<CandDBL.at(i)(1);}

            IndSol[j]=tempsold;
          }
        }
        DBL.at(i)(_,ind)=IndSol;
      }
    }

  }


  ///



  IntegerVector getSolnInt(int ind){
    IntegerVector soln;
    int iBOOL=0;
    int iOS=0;
    int iUOS=0;
    int iOMS=0;
    int iUOMS=0;
    for (int i=0;i<chromsizes.length();i++){
      if (chromtypes[i]=="BOOL"){
        for (int j=0;j<chromsizes[i];j++){
          soln.push_back(BOOL.at(iBOOL)(j,ind));
        }
        iBOOL++;
      }

      if (chromtypes[i]=="OS"){
        for (int j=0;j<chromsizes[i];j++){
          soln.push_back(OS.at(iOS)(j,ind));
        }
        iOS++;
      }
      if (chromtypes[i]=="UOS"){
        for (int j=0;j<chromsizes[i];j++){

          soln.push_back(UOS.at(iUOS)(j,ind));
        }
        iUOS++;
      }
      if (chromtypes[i]=="OMS"){
        for (int j=0;j<chromsizes[i];j++){
          soln.push_back(OMS.at(iOMS)(j,ind));
        }
        iOMS++;
      }
      if (chromtypes[i]=="UOMS"){
        for (int j=0;j<chromsizes[i];j++){
          soln.push_back(UOMS.at(iUOMS)(j,ind));
        }
        iUOMS++;
      }
    }
    return soln;
  }




  IntegerMatrix getSolnInt(IntegerVector inds){


    int ind=inds[0];
    IntegerVector soln;
    int iBOOL=0;
    int iOS=0;
    int iUOS=0;
    int iOMS=0;
    int iUOMS=0;
    for (int i=0;i<chromsizes.length();i++){
      if (chromtypes[i]=="BOOL"){
        for (int j=0;j<chromsizes[i];j++){
          soln.push_back(BOOL.at(iBOOL)(j,ind));
        }
        iBOOL++;
      }
      if (chromtypes[i]=="OS"){
        for (int j=0;j<chromsizes[i];j++){
          soln.push_back(OS.at(iOS)(j,ind));
        }
        iOS++;
      }
      if (chromtypes[i]=="UOS"){
        for (int j=0;j<chromsizes[i];j++){

          soln.push_back(UOS.at(iUOS)(j,ind));
        }
        iUOS++;
      }
      if (chromtypes[i]=="OMS"){
        for (int j=0;j<chromsizes[i];j++){
          soln.push_back(OMS.at(iOMS)(j,ind));
        }
        iOMS++;
      }
      if (chromtypes[i]=="UOMS"){
        for (int j=0;j<chromsizes[i];j++){
          soln.push_back(UOMS.at(iUOMS)(j,ind));
        }
        iUOMS++;
      }
    }

    IntegerMatrix solnmat(soln.length(),inds.length());
    solnmat.column(0)=soln;
    for (int indi=1;indi<inds.length();indi++){
      int ind=inds[indi];
      IntegerVector soln;
      int iBOOL=0;

      int iOS=0;
      int iUOS=0;
      int iOMS=0;
      int iUOMS=0;
      for (int i=0;i<chromsizes.length();i++){
        if (chromtypes[i]=="BOOL"){
          for (int j=0;j<chromsizes[i];j++){
            soln.push_back(BOOL.at(iBOOL)(j,ind));
          }
          iBOOL++;
        }

        if (chromtypes[i]=="OS"){
          for (int j=0;j<chromsizes[i];j++){
            soln.push_back(OS.at(iOS)(j,ind));
          }
          iOS++;
        }
        if (chromtypes[i]=="UOS"){
          for (int j=0;j<chromsizes[i];j++){

            soln.push_back(UOS.at(iUOS)(j,ind));
          }
          iUOS++;
        }
        if (chromtypes[i]=="OMS"){
          for (int j=0;j<chromsizes[i];j++){
            soln.push_back(OMS.at(iOMS)(j,ind));
          }
          iOMS++;
        }
        if (chromtypes[i]=="UOMS"){
          for (int j=0;j<chromsizes[i];j++){
            soln.push_back(UOMS.at(iUOMS)(j,ind));
          }
          iUOMS++;
        }
      }
      solnmat.column(indi)=soln;
    }
    return solnmat;
  }






  void putSolnInt(int ind, IntegerVector soln){
    int iBOOL=0;

    int iOS=0;
    int iUOS=0;
    int iOMS=0;
    int iUOMS=0;
    int jj=0;
    for (int i=0;i<chromsizes.length();i++){
      if (chromtypes[i]=="BOOL"){
        for (int j=0;j<chromsizes[i];j++){
          BOOL.at(iBOOL)(j,ind)=soln(jj);
          jj++;
        }
        iBOOL++;
      }

      if (chromtypes[i]=="OS"){
        for (int j=0;j<chromsizes[i];j++){
          OS.at(iOS)(j,ind)=soln(jj);
          jj++;
        }
        iOS++;
      }
      if (chromtypes[i]=="UOS"){
        for (int j=0;j<chromsizes[i];j++){

          UOS.at(iUOS)(j,ind)=soln[jj];
          jj++;
        }
        IntegerVector TempVec=UOS.at(iUOS)(_,ind);
        TempVec.sort();
        UOS.at(iUOS)(_,ind)=TempVec;
        iUOS++;
      }
      if (chromtypes[i]=="OMS"){
        for (int j=0;j<chromsizes[i];j++){
          OMS.at(iOMS)(j,ind)=soln[jj];
          jj++;
        }
        iOMS++;
      }
      if (chromtypes[i]=="UOMS"){
        for (int j=0;j<chromsizes[i];j++){
          UOMS.at(iUOMS)(j,ind)=soln[jj];
          jj++;
        }
        IntegerVector TempVec=UOMS.at(iUOMS)(_,ind);
        TempVec.sort();
        UOS.at(iUOMS)(_,ind)=TempVec;
        iUOMS++;
      }
    }
  }


  NumericVector getSolnDbl(int ind){
    NumericVector soln;
    int iDBL=0;
    for (int i=0;i<chromsizes.length();i++){
      if (chromtypes[i]=="DBL"){
        for (int j=0;j<chromsizes[i];j++){
          soln.push_back(DBL.at(iDBL)(j,ind));
        }
        iDBL++;
      }
    }
    return soln;
  }



  NumericMatrix getSolnDbl(IntegerVector inds){

    int ind=inds[0];
    NumericVector soln;
    int iDBL=0;
    for (int i=0;i<chromsizes.length();i++){
      if (chromtypes[i]=="DBL"){
        for (int j=0;j<chromsizes[i];j++){
          soln.push_back(DBL.at(iDBL)(j,ind));
        }
        iDBL++;
      }
    }
    NumericMatrix solnmat(soln.length(),inds.length());
    solnmat.column(0)=soln;
    for (int indi=1;indi<inds.length();indi++){
      int ind=inds[indi];
      NumericVector soln;
      int iDBL=0;
      for (int i=0;i<chromsizes.length();i++){
        if (chromtypes[i]=="DBL"){
          for (int j=0;j<chromsizes[i];j++){
            soln.push_back(DBL.at(iDBL)(j,ind));
          }
          iDBL++;
        }
      }
      solnmat.column(indi)=soln;

    }
    return solnmat;
  }






  void putSolnDbl(int ind, NumericVector soln){
    int iDBL=0;
    int jj=0;
    for (int i=0;i<chromsizes.length();i++){
      if (chromtypes[i]=="DBL"){
        for (int j=0;j<chromsizes[i];j++){
          DBL.at(iDBL)(j,ind)=soln[jj];
          jj++;
        }
        iDBL++;
      }
    }
  }

  ///////////


  void set_STATCLASS(STATCLASSMOO STATC_){
    StatClass=STATC_;
  }


  void init_Fitness(){
    FitnessVals =NumericMatrix(StatClass.numstat,npop);
  }

  void set_Fitness(int ind, Function Stat){
    if (nDBL>0){
      FitnessVals(_, ind)=StatClass.GetStat(getSolnInt(ind),getSolnDbl(ind), Stat);
    } else {
      FitnessVals(_, ind)=StatClass.GetStat(getSolnInt(ind), Stat);
    }
  }
  void set_Fitness(int ind, NumericVector val){
    FitnessVals(_, ind)=val;
  }


  NumericVector get_Fitness(int ind, Function Stat){
    NumericVector out;
    if (nDBL>0){
      out=StatClass.GetStat(getSolnInt(ind),getSolnDbl(ind), Stat);
    } else {
      out=StatClass.GetStat(getSolnInt(ind), Stat);
    }
    return out;
  }


  NumericVector get_Fitness(IntegerVector soln_int, NumericVector soln_dbl, Function Stat){
    NumericVector out;
    if (nDBL>0){
      out=StatClass.GetStat(soln_int,soln_dbl, Stat);
    } else {
      out=StatClass.GetStat(soln_int, Stat);
    }
    return StatClass.GetStat(soln_int,soln_dbl, Stat);
  }

  NumericMatrix get_Fitness(){
    return FitnessVals;
  }

  NumericVector get_Fitness(int i){
    return FitnessVals(_, i);
  }

  NumericMatrix get_Fitness(IntegerVector inds){
    return subcolNM(FitnessVals, inds);
  }

};











/////////////////////////////////////////////////////////////////////////////////////



/////////////////////////////////////////////////////////////////////////////////////////////////////////
class OutTrainSelMOO {
public:
  IntegerMatrix Best_Sols_Int;
  NumericMatrix Best_Sols_DBL;
  NumericMatrix Best_Vals;



  OutTrainSelMOO(List Data,
                 List CANDIDATES,
                 Rcpp::IntegerVector setsizes,
                 Rcpp::CharacterVector settypes,
                 Rcpp::Function Stat,
                 int nstat,
                 List control) {


    int NPOP = as<int>(control["npop"]);
    int NITERGA = as<int>(control["niterations"]);
    double MUTPROB = as<double>(control["mutprob"]);
    bool PROGRESS = as<bool>(control["progress"]);


    bool maxiter = false;

    bool CheckData = false;
    if (Data.size() > 0) {
      CheckData = true;
    }



    /////errors


    STATCLASSMOO STATc;

    ///
    if (!CheckData) {
      STATc = STATCLASSMOO();

    }

    if (CheckData) {

      STATc = STATCLASSMOO(Data);

    }

    STATc.numstat=nstat;


    ////////////////

    PopulationMOO  pop;
    pop.set_npop(NPOP);


    pop.set_nchrom(setsizes.length());
    pop.set_chromsizes(setsizes);
    pop.set_chromtypes(settypes);
    for (int i=0;i<pop.get_nchrom();i++){
      if (settypes[i]=="OS"){
        pop.push_back_CandOS(as<IntegerVector>(CANDIDATES[i]));
      }
      if (settypes[i]=="UOS"){
        pop.push_back_CandUOS(as<IntegerVector>(CANDIDATES[i]));

      }
      if (settypes[i]=="OMS"){
        pop.push_back_CandOMS(as<IntegerVector>(CANDIDATES[i]));
      }
      if (settypes[i]=="UOMS"){
        pop.push_back_CandUOMS(as<IntegerVector>(CANDIDATES[i]));
      }
      if (settypes[i]=="DBL"){
        pop.push_back_CandDBL(as<NumericVector>(CANDIDATES[i]));
      }
    }



    pop.set_STATCLASS(STATc);

    pop.init_Fitness();

    pop.InitRand();

    for (int i=0;i<pop.get_npop();i++){

      pop.set_Fitness(i, Stat);

    }

    int Generation = 0;
    Progress p(NITERGA, PROGRESS);
    LogicalVector dominatedbool=do_is_dominated(pop.get_Fitness());
    LogicalVector duplicated = duplicatedRcpp(transpose(pop.get_Fitness()));

    IntegerVector bestsols =  whichRcpp((!dominatedbool & ! duplicated));
    IntegerVector worstsols = whichRcpp((dominatedbool | duplicated));




    while (!maxiter) {

//thinning
      if (worstsols.length()<5){
        /**NumericVector prob(pop.get_npop());

        for (int i=0;i<pop.get_npop();i++){
          std::fill(prob.begin(),prob.end(),1);
          for (int i=0;i<pop.get_Fitness().nrow();i++){
            double mean_E =sum(pop.get_Fitness()(i,_))/pop.get_npop();
            double sd_E=sd(pop.get_Fitness()(i,_));
            for (int j=0;j<pop.get_npop();j++){
              double x=pop.get_Fitness()(i,j);
              NumericVector xvec={x};
              prob[j]=prob[j]*(10+Rcpp::dnorm(xvec,mean_E,sd_E+1e-6,false)[0]);
            }
          }
        }
        worstsols=sample(bestsols,5, false, prob);
        **/
        worstsols=sample(bestsols,5, false);

        bestsols=setdiff(bestsols,worstsols);
      }
      if (bestsols.length()<5){
        /** NumericVector prob(pop.get_npop());
        for (int i=0;i<pop.get_npop();i++){
          std::fill(prob.begin(),prob.end(),1);
          for (int i=0;i<pop.get_Fitness().nrow();i++){
            double mean_E =sum(pop.get_Fitness()(i,_))/pop.get_npop();
            double sd_E=sd(pop.get_Fitness()(i,_));
            for (int j=0;j<pop.get_npop();j++){
              double x=pop.get_Fitness()(i,j);
              NumericVector xvec={x};
              prob[j]=prob[j]*1/(10+Rcpp::dnorm(xvec,mean_E,sd_E+1e-6,false)[0]);
            }
          }
        }
        bestsols=sample(worstsols,5, false, prob);
         **/
        bestsols=sample(worstsols,5, false);

        worstsols=setdiff(worstsols,bestsols);
      }

      p.increment();
      R_CheckUserInterrupt();
      Generation++;
      if (Generation == NITERGA) {
        Rcout << "Maximum number of iterations reached." << std::endl;
        maxiter = true;
      }

      Best_Vals=pop.get_Fitness(bestsols);
      for (int i=0;i<worstsols.length();i++){
        NumericVector prob(bestsols.length());
        std::fill(prob.begin(),prob.end(),1);
        for (int i=0;i<Best_Vals.nrow();i++){
          double mean_E =sum(Best_Vals(i,_))/bestsols.length();
          double sd_E=sd(Best_Vals(i,_));
          for (int j=0;j<bestsols.length();j++){
            double x=Best_Vals(i,j);
            NumericVector xvec={x};
            prob[j]=prob[j]*1/(10+Rcpp::dnorm(xvec,mean_E,sd_E+1e-6,false)[0]);
          }
        }

        int p1=sample(bestsols,1, false, prob)(0);
        int p2=sample(bestsols,1, false, prob)(0);
        pop.MakeCross(p1,p2,worstsols[i]);
        pop.Mutate(worstsols[i],MUTPROB);
        pop.set_Fitness(worstsols[i], Stat);
      }

      dominatedbool=do_is_dominated(pop.get_Fitness());
      duplicated = duplicatedRcpp(transpose(pop.get_Fitness()));

      bestsols =  whichRcpp((!dominatedbool & ! duplicated));
      worstsols = whichRcpp((dominatedbool | duplicated));


    }

    Best_Sols_Int=pop.getSolnInt(bestsols);
    Best_Sols_DBL=pop.getSolnDbl(bestsols);
    Best_Vals=pop.get_Fitness(bestsols);




  }

  List getSol() {
    return Rcpp::List::create(Rcpp::Named("BestSol_int") = Best_Sols_Int,
                              Rcpp::Named("BestSol_DBL") =  Best_Sols_DBL,
                              Rcpp::Named("BestVal") =Best_Vals
    );
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////




// [[Rcpp::export]]
List TrainSelCMOO(List Data,
                  List CANDIDATES,
                  Rcpp::IntegerVector setsizes,
                  Rcpp::CharacterVector settypes,
                  Rcpp::Function Stat,
                  int  nstat,
                  List control) {

  OutTrainSelMOO out(Data,
                     CANDIDATES,
                     setsizes,
                     settypes,
                     Stat,
                     nstat,
                     control);
  return out.getSol();

}









