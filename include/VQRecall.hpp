#ifndef NEURALVQ_VQRECALL_HPP
#define NEURALVQ_VQRECALL_HPP


#include "annoy/annoylib.h"
#include "annoy/kissrandom.h"
#include <omp.h>
#include <map>

#include "../include/pybind11/pybind11.h"
#include <pybind11/numpy.h>
namespace py = pybind11;

typedef double ANNOYTYPE;
typedef Annoy::AnnoyIndex <int, ANNOYTYPE, Annoy::Euclidean, Annoy::Kiss64Random, Annoy::AnnoyIndexSingleThreadedBuildPolicy> MyAnnoyIndex;
const std::vector<int> XL_empty = {};
typedef std::pair<int,int> Tpairint;
typedef std::map<Tpairint, int> TCADJMap;
typedef std::map<int,int> TLabelMap;

#pragma once

inline void RFL_Winner(int& WinLabel, double& Purity, const TLabelMap& x) {

  TLabelMap::const_iterator bestit;
  int bestval = 0;
  int sumval = 0;
  for(TLabelMap::const_iterator it = x.begin(); it != x.end(); ++it) {
    if(it->second > bestval) {
      bestit = it;
      bestval = it->second;
    }
    sumval += it->second;
  }

  WinLabel = bestit->first;

  Purity = 1.0 - std::sqrt(1.0 - std::sqrt(double(bestval) / double(sumval))); // 1 - Hellinger Distance(ideal, observed)

  return;
}


// *** Versions that write BMU & QE as a vector, with indices 0->(N-1) holding BMU1, elements N->(2N-1) holding BMu2, etc.
inline void cpp_AnnoyBMU(std::vector<int>& BMU, std::vector<double>& QE,
                         const double* X,
                         const double* W,
                         unsigned int N, unsigned int M, unsigned int d,
                         unsigned int nBMU = 2,
                         unsigned int nAnnoyTrees = 50) {
  // Build Annoy indexing object
  MyAnnoyIndex AnnoyObj(d);

  for(unsigned int i=0; i<M; ++i) {
    AnnoyObj.add_item(i, &W[i*d]);
  }

  AnnoyObj.build(nAnnoyTrees);


  BMU.resize(N * nBMU);
  QE.resize(N * nBMU);
 
  // Find BMU of each x
#pragma omp parallel for
  for(unsigned int i=0; i<N; ++i) {

    std::vector<int> tmp_BMU;
    std::vector<double> tmp_QE;

    AnnoyObj.get_nns_by_vector(&X[i*d], nBMU, -1, &tmp_BMU, &tmp_QE);
    for(unsigned int j=0; j<nBMU; ++j) {
      BMU[i + j*N] = tmp_BMU[j];
      QE[i + j*N] = tmp_QE[j];
    }

  }

  return;
}

inline void cpp_AnnoyRecall(std::vector<std::vector<int>>& RF,
                            std::vector<int>& RF_Size,
                            std::vector<int>& CADJi, std::vector<int>& CADJj, std::vector<int>& CADJx,
                            std::vector<TLabelMap>& RFL_Dist,
                            std::vector<int>& RFL,
                            std::vector<double>& RFL_Purity, double& RFL_Purity_UOA, double& RFL_Purity_WOA,
                            const std::vector<int>& BMU,
                            const std::vector<double>& QE,
                            unsigned int N,
                            unsigned int M,
                            const std::vector<int>& XL = XL_empty
) {
  // BMU & QE should be arrange such that elements 0->N-1 represent BMU1 of each X, elements N->2N-1 represent BMU2 of each X, and so on
  // BMU must have length >= 2N to calculate CADJ

  RF.resize(M);
  RF_Size.resize(M);
  std::fill(RF_Size.begin(), RF_Size.end(), 0);

  std::vector<int> XL_unq;
  unsigned int XL_N_unq = 0;
  TLabelMap XLMap;
  if(XL.size() > 0) {
    XL_unq = XL;
    std::vector<int>::iterator last = std::unique(XL_unq.begin(), XL_unq.end());
    XL_unq.erase(last, XL_unq.end());
    std::sort(XL_unq.begin(), XL_unq.end());
    XL_N_unq = XL_unq.size();

    for(unsigned int i=0; i<XL_N_unq; ++i) {
      XLMap[XL_unq[i]] = 0;
    }

    RFL_Dist.resize(M);
    std::fill(RFL_Dist.begin(), RFL_Dist.end(), XLMap);
    RFL.resize(M);
    std::fill(RFL.begin(), RFL.end(), 0);
    RFL_Purity.resize(M);
    std::fill(RFL_Purity.begin(), RFL_Purity.end(), 0.0);
  }

  double purWOA_num = 0.0, purWOA_denom = 0.0, purUOA_num = 0.0, purUOA_denom = 0.0;
#pragma omp declare reduction (mergevec : std::vector<int> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for reduction(mergevec: CADJi) reduction(mergevec: CADJj) reduction(mergevec:CADJx) reduction(+:purWOA_num) reduction(+:purWOA_denom) reduction(+:purUOA_num) reduction(+:purUOA_denom)
  for(unsigned int i=0; i<M; ++i) {

    TCADJMap mapCADJ;

    for(unsigned int j=0; j<N; ++j) {
      if(BMU[j]==i) {
        RF[i].push_back(j);
        RF_Size[i]++;
        mapCADJ[std::make_pair(i, BMU[j + N])]++;
        if(XL_N_unq > 0) RFL_Dist[i][XL[j]]++; // Add labels, if they exist
      }
    }

    for(TCADJMap::iterator it = mapCADJ.begin(); it != mapCADJ.end(); ++it) {
      CADJi.push_back(it->first.first);
      CADJj.push_back(it->first.second);
      CADJx.push_back(it->second);
    }

    if(XL_N_unq > 0 && RF_Size[i] > 0) {
      RFL_Winner(RFL[i], RFL_Purity[i], RFL_Dist[i]);

      // Update overall average purities
      purWOA_num += double(RF_Size[i]) * RFL_Purity[i];
      purWOA_denom += double(RF_Size[i]);
      purUOA_num += RFL_Purity[i];
      purUOA_denom += 1.0;
    }

  }

  RFL_Purity_UOA = (purUOA_denom > 0.0) ? purUOA_num / purUOA_denom : 0.0;
  RFL_Purity_WOA = (purWOA_denom > 0.0) ? purWOA_num / purWOA_denom : 0.0;

  return;
}




// *** Versions that write BMU & QE as a list, with BMU[0] holding c(BMU1,BMU2,...) of X[0], BMU[1] holding c(BMU1,BMU2,...) of X[1], etc.
inline void cpp_AnnoyBMU(std::vector<std::vector<int>>& BMU,
                         std::vector<std::vector<double>>& QE,
                         const double* X,
                         const double* W,
                         unsigned int N, unsigned int M, unsigned int d,
                         unsigned int nBMU = 2,
                         unsigned int nAnnoyTrees = 50) {


  // Build Annoy indexing object
  MyAnnoyIndex AnnoyObj(d);

  for(unsigned int i=0; i<M; ++i) {
    AnnoyObj.add_item(i, &W[i*d]);
  }

  AnnoyObj.build(nAnnoyTrees);

  BMU.resize(N);
  QE.resize(N);

  // Find BMU of each x
#pragma omp parallel for
  for(unsigned int i=0; i<N; ++i) {
    AnnoyObj.get_nns_by_vector(&X[i*d], nBMU, -1, &BMU[i], &QE[i]);
  }

  return;
}


// class RecallContainer {
// public:
//   int nBMU;
//   int nAnnoyTrees;

//   std::vector<int> BMU;
//   std::vector<double> QE;

//   std::vector<std::vector<int>> RF;
//   std::vector<int> RF_Size;
//   std::vector<int> CADJi, CADJj, CADJ;

//   std::vector<TLabelMap> RFL_Dist;
//   std::vector<int> RFL;
//   std::vector<double> RFL_Purity;
//   double RFL_Purity_UOA;
//   double RFL_Purity_WOA;

//   // Constructor
//   RecallContainer() {};
// };

// inline RecallContainer cpp_Recall(const double* X, const double* W, 
//                               unsigned int N, unsigned int M, unsigned int d, 
//                               const std::vector<int>& XL = XL_empty, 
//                               unsigned int nBMU = 2, unsigned int nAnnoyTrees = 50) {

//   if(nBMU < 2) {
//     throw std::runtime_error("nBMU must be >= 2");
//   }

//   // Initialize output container 
//   RecallContainer rec;

//   rec.nBMU = nBMU; 
//   rec.nAnnoyTrees = nAnnoyTrees; 
  
//   // Find BMUs
//   cpp_AnnoyBMU(rec.BMU, rec.QE, 
//   X, W, N, M, d, nBMU, nAnnoyTrees);

//   // Recall 
//   cpp_AnnoyRecall(rec.RF, rec.RF_Size,
//                   rec.CADJi, rec.CADJj, rec.CADJ,
//                   rec.RFL_Dist, rec.RFL, rec.RFL_Purity, rec.RFL_Purity_UOA, rec.RFL_Purity_WOA,
//                   rec.BMU, rec.QE, N, M, XL);

//   return rec; 
// }


class VQRecall {
public:
  int nBMU;
  int nAnnoyTrees;

  unsigned int N, M, d; 
  std::vector<int> BMU;
  std::vector<double> QE;

  std::vector<std::vector<int>> RF;
  std::vector<int> RF_Size;
  std::vector<int> CADJi, CADJj, CADJ;

  std::vector<TLabelMap> RFL_Dist;
  std::vector<int> RFL;
  std::vector<double> RFL_Purity;
  double RFL_Purity_UOA;
  double RFL_Purity_WOA;

  // Constructor
  VQRecall(int nBMU = 2, int nAnnoyTrees = 50) : nBMU(nBMU), nAnnoyTrees(nAnnoyTrees) {};

  // Recall functions
  // The first version takes C++ constructs (pointers), while the second takes python types. 
  // They can't be overloaded (have to have different names) because pybind11 doesn't handle function overloading appropriately (as of June 9 2023)
  void cpp_Recall(const double* X, const double* W, 
              unsigned int N, unsigned int M, unsigned int d, 
              const std::vector<int>& XL = XL_empty,
              bool BMU_only = false);

  void Recall(const py::array_t<double,  py::array::c_style>& X, 
              const py::array_t<double,  py::array::c_style>& W,
              const std::vector<int>& XL = XL_empty,
              bool BMU_only = false);

};


inline void VQRecall::cpp_Recall(const double* X, const double* W, 
              unsigned int N, unsigned int M, unsigned int d, 
              const std::vector<int>& XL, bool BMU_only) {

  // Store dimensions 
  this->N = N; 
  this->M = M; 
  this->d = d; 

  // Find BMUs
  cpp_AnnoyBMU(this->BMU, this->QE, 
                X, W, N, M, d, nBMU, nAnnoyTrees);

  if(BMU_only) return; 

  // Full recall 
  cpp_AnnoyRecall(this->RF, this->RF_Size,
                  this->CADJi, this->CADJj, this->CADJ,
                  this->RFL_Dist, this->RFL, this->RFL_Purity, this->RFL_Purity_UOA, this->RFL_Purity_WOA,
                  this->BMU, this->QE, this->N, this->M, XL);

  return; 
}

inline void VQRecall::Recall(const py::array_t<double,  py::array::c_style>& X, 
              const py::array_t<double,  py::array::c_style>& W,
              const std::vector<int>& XL, bool BMU_only) {

  // Strip out dimensions, check 
    py::buffer_info bufX = X.request(), bufW = W.request();
    unsigned int N = bufX.shape[0]; 
    unsigned int d = bufX.shape[1]; 
    unsigned int M = bufW.shape[0]; 
    if(bufW.shape[1] != d) throw std::runtime_error("ncol(X) != ncol(W)");

    // Strip out pointers 
    double* ptrX = static_cast<double *>(bufX.ptr);
    double* ptrW = static_cast<double *>(bufW.ptr);

    // Call other version of recall 
    this->cpp_Recall(ptrX, ptrW, N, M, d, XL, BMU_only);

    return;

}
#endif

