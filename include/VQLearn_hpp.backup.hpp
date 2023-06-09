#ifndef NEURALVQ_VQLEARN_HPP
#define NEURALVQ_VQLEARN_HPP

#include "VQRecall.hpp"

#include <omp.h>
#include <algorithm>
#include <numeric>


class AnnoyBatchLearnWorker {
public:
  int d; // data dimension
  std::vector<double> X; // data as vector, C-style arrangement, e.g., (x1-d1, x1-d2, ... x1-dd, x2-d1, ...)
  std::vector<double> W; // initial prototypes as vector, C-style arrangement
  int n_epochs; // total number of training epochs
  
  double rho0; // starting value of neighborhood radius
  double rho_anneal; // annealing factor of neighbor radius, typically 0.95 or 0.9 (must be < 1)
  double rho_min; // minimum value of rho
  
  double min_h; // minimum value of neighborhood activation.

  // Containers
  int N; // num of data vectors, set internally = length(X) / d
  int M; // num of prototype vector, set internally = length(W) /d

  std::vector<int> XBMU; // BMUs of X. indices 0->(N-1) contain BMU1, indices N->(2N-1) contain BMU2, etc.
  std::vector<double> XQE; // QEs of X, arranged as XBMU
  std::vector<int> prevXBMU;
  std::vector<double> prevXQE;

  double XMQE, delXBMU;
  double prevXMQE, delXMQE;

  double rho; // current rho value at each epoch = rho0 * rho_anneal^(epoch - 1)
  std::vector<std::vector<int>> hidx;
  std::vector<std::vector<double>> h;
  double h_sum_strength;

  std::vector<double> sum_hX;
  std::vector<double> sum_h;

  AnnoyBatchLearnWorker(int d, std::vector<double> X, std::vector<double> W,
                        int n_epochs, 
                        double rho0, double rho_anneal, double rho_min, double min_h) :
    d(d), X(X), W(W), n_epochs(n_epochs), rho0(rho0), rho_anneal(rho_anneal), rho_min(rho_min), min_h(min_h) {
    N = X.size() / d;
    M = W.size() / d;
    sum_hX.resize(M*d);
    sum_h.resize(M);
  }

  void clear_containers() {
    std::fill(sum_hX.begin(), sum_hX.end(), 0.0);
    std::fill(sum_h.begin(), sum_h.end(), 0.0);
    hidx.clear();
    h.clear();
  }

  void set_h_NG() {
    int n_update;
    if(min_h > 0.0) {
      n_update = std::ceil(-std::log(min_h) * rho) + 1; // +1 to include self
      n_update = std::min(M, n_update);
    } else {
      n_update = M;
    }


    cpp_AnnoyBMU(hidx, h, W, W, d, n_update); // compute BMUs of W using W as codebook

    std::vector<double> tmph(n_update);

    h_sum_strength = 0.0;
    for(unsigned int k=0; k<n_update; ++k) {
      tmph[k] = std::exp(-double(k) / rho);
      h_sum_strength += tmph[k];
    }

    for(unsigned int i=0; i<M; ++i) {
      h[i] = tmph;
    }

    return;
  }


  void prepare_updateW() {

    //std::fill(sum_hX.begin(), sum_hX.end(), 0.0);
    //std::fill(sum_h.begin(), sum_h.end(), 0.0);

    // ** Prepare Update
#pragma omp declare reduction(vecadd : std::vector<double> : std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<double>())) \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))
#pragma omp parallel for reduction(vecadd: sum_hX) reduction(vecadd: sum_h)
      for(unsigned int i=0; i<N; ++i) {
        // numerator

        for(unsigned int k=0; k<hidx[XBMU[i]].size(); ++k) {

          for(unsigned int dd=0; dd<d; ++dd) {

            sum_hX[hidx[XBMU[i]][k]*d + dd] += h[XBMU[i]][k] * X[i*d + dd];

          } // close dimension loop
          sum_h[hidx[XBMU[i]][k]] += h[XBMU[i]][k];
        } // close neighbor loop
      } // close data loop


      // Update
#pragma omp parallel for
    for(unsigned int j=0; j<M; ++j) {
      if(!(sum_h[j] > 0.0)) continue;

      for(unsigned int dd=0; dd<d; ++dd) {
        W[j*d + dd] = sum_hX[j*d + dd] / sum_h[j];
      }
    }




  }


 


  void train() {

   
    // Find BMU of data
    printf("Xsize = %d, Wsize = %d\n", X.size(), W.size()); 
    cpp_AnnoyBMU(XBMU, XQE, X, W, d, 2, 50); // nBMU=2, nTrees=50
    prevXBMU = XBMU;
    prevXMQE = std::accumulate(XQE.begin(), XQE.begin()+N, 0.0) / N;
    delXBMU = 1.0;

    printf("%5s  %5s  %5s  %5s  %5s  %5s\n","Epoch", "rho", "NbEff", "MQE", "\u0394MQE", "\u0394BMU");
    printf("%5d  %5.3f  %5.3f  %5.3f  %5.3f  %5.3f\n",  0, rho0, 0.0, prevXMQE, 0.0, delXBMU);



    for(unsigned int epoch = 1; epoch <= n_epochs; ++epoch) {
      // Anneal rho
      rho = rho0 * std::pow(rho_anneal, double(epoch)-1.0);
      rho = std::max(rho_min, rho);

      // Clear containers from previous iteration
      this->clear_containers();

      // Update neighbor activations
      this->set_h_NG();

      // Prepare prototype update
      this->prepare_updateW();

      // Update prototypes
      //this->updateW();

      // Compute new BMUs of data
      cpp_AnnoyBMU(XBMU, XQE, X, W, d, 2, 50); // nBMU=2, nTrees=50
      XMQE = std::accumulate(XQE.begin(), XQE.begin()+N, 0.0) / N;
      delXMQE = XMQE - prevXMQE;
      delXBMU = 1.0 - double( std::inner_product(XBMU.begin(), XBMU.begin()+N, prevXBMU.begin(), 0, std::plus<int>(), std::equal_to<int>()) ) / N;

      prevXBMU = XBMU;
      prevXQE = XQE;
      prevXMQE = XMQE;


      // Update progress
      //bar.update();

      // if(epoch % 10 == 0) Rprintf("%5s  %5s  %5s  %5s  %5s  %5s\n",  "Epoch", "rho", "NbEff", "MQE", "\u0394MQE", "\u0394BMU");
      if(epoch % 10 == 0) printf("%5s  %5s  %5s  %5s  %5s  %5s\n",  "Epoch", "rho", "NbEff", "MQE", "\u0394MQE", "\u0394BMU");
      // Rprintf("%5d  %5.3f  %5.3f  %5.3f  %5.3f  %5.3f\n",  epoch, rho, h_sum_strength, prevXMQE, delXMQE, delXBMU);
      printf("%5d  %5.3f  %5.3f  %5.3f  %5.3f  %5.3f\n",  epoch, rho, h_sum_strength, prevXMQE, delXMQE, delXBMU);



    } // close training loop

  }


};


#endif


