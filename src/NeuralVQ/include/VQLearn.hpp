#ifndef NEURALVQ_VQLEARN_HPP
#define NEURALVQ_VQLEARN_HPP

#include "VQRecall.hpp"
#include "pybind_custom_types.hpp"
#include "progressbar.hpp"

#include <omp.h>
#include <algorithm>
#include <numeric>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;


// ** Learn History Container 
struct LearnHistoryContainer {

  std::vector<int> Epoch;
  std::vector<double> rho;
  std::vector<double> NbEff;
  std::vector<double> MQE;
  //std::vector<double> AD_MQE;
  std::vector<double> delMQE;
  std::vector<double> delBMU;

  std::vector<double> QE_prev;
  std::vector<int> BMU_prev;

  // Constructors
  //LearnHistoryContainer();
  //LearnHistoryContainer(int N) : N(N) {};

  // Updater
  void Update(int Epoch_cur, double rho_cur, double NbEff_cur,
              const std::vector<double>& QE_cur, const std::vector<int>& BMU_cur, int N) {

    Epoch.push_back(Epoch_cur);
    rho.push_back(rho_cur);
    NbEff.push_back(NbEff_cur);

    MQE.push_back(std::accumulate(QE_cur.begin(), QE_cur.begin()+N, 0.0) / N);

    double tmp_delMQE;
    double tmp_delBMU;
    if(Epoch_cur > 0) {
      tmp_delMQE = std::abs( MQE[MQE.size()-1] - MQE[MQE.size()-2] ) / MQE[MQE.size()-2];
      tmp_delBMU = 1.0 - double( std::inner_product(BMU_cur.begin(), BMU_cur.begin()+N, BMU_prev.begin(), 0, std::plus<int>(), std::equal_to<int>()) ) / N;
    } else {
      tmp_delMQE = std::numeric_limits<double>::quiet_NaN();
      tmp_delBMU = std::numeric_limits<double>::quiet_NaN();
    }
    delMQE.push_back(tmp_delMQE);
    delBMU.push_back(tmp_delBMU);

    // Store current QE & BMU as previous
    QE_prev = QE_cur;
    BMU_prev = BMU_cur;
  }

  // Printer
  void Print() {
    if(Epoch[Epoch.size()-1] % 10 == 0) printf("%5s  %5s  %5s  %5s  %5s  %5s\n",  "Epoch", "rho", "NbEff", "MQE", "\u0394MQE", "\u0394BMU");
    printf("%5d  %5.3f  %5.3f  %5.3f  %5.3f  %5.3f\n",  Epoch[Epoch.size()-1], rho[rho.size()-1], NbEff[NbEff.size()-1], MQE[MQE.size()-1], delMQE[delMQE.size()-1], delBMU[delBMU.size()-1]);
  }

  // Get current value of delBMU 
  double get_cur_delBMU() {
    return delBMU[delBMU.size()-1];
  }

  double get_cur_delMQE() {
    return delMQE[delMQE.size()-1];
  }


};


// ** VQ Learning 
class VQLearnWorker {
private:
  double* X;
  std::vector<double> W; 
  const std::vector<int> XL; 

public:
  unsigned int N, M, d; // # data vectors, # prototypes, data dimension 
  
  unsigned int age;   // running total of learning epochs performed 
  
  double rho0; // starting value of neighborhood radius
  double rho_anneal; // annealing factor of neighbor radius, typically 0.95 or 0.9 (must be < 1)
  double rho_min; // minimum value of rho
  
  double eta_min; // minimum value of neighborhood activation.

  int verbosity; // amount of verbosity during learning 

  double rho; // current rho value at each epoch = rho0 * rho_anneal^(epoch - 1)
  std::vector<std::vector<int>> eta_idx; // list of vecs storing the W update indices
  std::vector<std::vector<double>> eta; // list of vecs storing the W update strenghts
  double eta_avg; 

  std::vector<double> sum_etaX; // sum(contribution of eta * X) to each prototype update 
  std::vector<double> sum_eta;  // sum(contribution of eta) to each prototype 

  LearnHistoryContainer LearnHist; 
  VQRecall Recall; 

  // Constructor 
  VQLearnWorker(const numpyCarr& X_, const numpyCarr& W_, 
                //const std::vector<int>& XL = XL_empty,
                const std::vector<int>& XL,
                double rho0 = -1.0, double rho_anneal = 0.95, double rho_min = 0.75, 
                double eta_min = 0.01, int verbosity = 2);

  // Clear out the containers used to update prototypes at each epoch. 
  // This prepares them for the next learning epoch. 
  void reset_Update_Containers();

  // Calculate the neighborhood update strengths for each prototype at each epoch. 
  void calc_eta_NG(); 

  // Perform the prototype updates 
  void update_W(); 

  // Main learning method 
  void learn(unsigned int n_epochs, double conv_delBMU = std::numeric_limits<double>::max(), double conv_delMQE = std::numeric_limits<double>::max());

  // Return the prototpyes to python as a numpy array 
  numpyCarr get_Wnumpy();
};


// Constructor 
VQLearnWorker::VQLearnWorker(const numpyCarr& X_, const numpyCarr& W_,
                              const std::vector<int>& XL,
                              double rho0, double rho_anneal, double rho_min, 
                              double eta_min, int verbosity) : 
                              XL(XL), rho0(rho0), rho_anneal(rho_anneal), rho_min(rho_min), 
                              eta_min(eta_min), verbosity(verbosity)  {

    // Store pointer to data, get its numpy array dimensions 
    unwrap_numpyCarr(this->X, this->N, this->d, X_); 
    // Extract initial prototypes into a std::vector 
    unsigned int dW; 
    unwrap_numpyCarr(W, this->M, dW, W_); 
   
    if(this->d != dW) throw std::runtime_error("ncol(X) != ncol(W)");
   
    // Set age to 0 
    this->age = 0; 

    // Default rho0 = -1 means rho0 = sqrt(N)
    if(rho0 < 0.0) rho0 = std::sqrt(double(N));

    // Initialize update containers
    sum_etaX.resize(M*d);
    sum_eta.resize(M);

  }



// Reset Update Containers 
void VQLearnWorker::reset_Update_Containers() {
    std::fill(sum_etaX.begin(), sum_etaX.end(), 0.0);
    std::fill(sum_eta.begin(), sum_eta.end(), 0.0);
    eta_idx.clear();
    eta.clear();
  }


// Calculate the neighborhood update strengths for each prototype at each epoch. 
void VQLearnWorker::calc_eta_NG() {

    // First determine how many neighboring prototypes will influence each prototype's update. 
    // This is controlled via min_eta. E.g., count(exp(-rank / rho) > min_eta)
    unsigned int n_influence;
    if(eta_min > 0.0) {
      n_influence = std::ceil(-std::log(eta_min) * rho) + 1; // +1 to include self
      n_influence = std::min(M, n_influence);
    } else {
      n_influence = M;
    }

    // Find the n_update nearest neighbors of each prototype, 
    // i.e., compute BMUs of W using W as codebook
    cpp_AnnoyBMU(eta_idx, eta, this->W.data(), this->W.data(), M, M, d, n_influence); 

    // Create a temporary vector containing eta(k) for the influence contribution of each of 
    // of the k-nearest neighbors to each prototype 
    std::vector<double> tmp_eta(n_influence);
    eta_avg = 0.0;
    for(unsigned int k=0; k<n_influence; ++k) {
      tmp_eta[k] = std::exp(-double(k) / rho);
      eta_avg += tmp_eta[k];
    }

    // Assign the neighborhood contributions of each prototype, in parallel 
    #pragma omp parallel for
    for(unsigned int i=0; i<M; ++i) {
      eta[i] = tmp_eta;
    }

    return;
  }


// Perform the prototype updates 
void VQLearnWorker::update_W() {

   
    // Prepare Update: 
    // This involves looping over each data, and compute its contribution to each prototype's update 
#pragma omp declare reduction(vecadd : std::vector<double> : std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<double>())) \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))
#pragma omp parallel for reduction(vecadd: sum_etaX) reduction(vecadd: sum_eta)
      for(unsigned int i=0; i<N; ++i) {
        // numerator

        for(unsigned int k=0; k<eta_idx[Recall.BMU[i]].size(); ++k) {

          for(unsigned int dd=0; dd<d; ++dd) {

            sum_etaX[eta_idx[Recall.BMU[i]][k]*d + dd] += eta[Recall.BMU[i]][k] * X[i*d + dd];

          } // close dimension loop
          sum_eta[eta_idx[Recall.BMU[i]][k]] += eta[Recall.BMU[i]][k];
        } // close neighbor loop
      } // close data loop


      // ** Update in place 
#pragma omp parallel for
    for(unsigned int j=0; j<M; ++j) {
      if(!(sum_eta[j] > 0.0)) continue;

      for(unsigned int dd=0; dd<d; ++dd) {
        W[j*d + dd] = sum_etaX[j*d + dd] / sum_eta[j];
      }
    }

  }


// Main learning method 
void VQLearnWorker::learn(unsigned int n_epochs, double conv_delBMU, double conv_delMQE) {

    // Find BMU of data using initial prototypes. 
    // The resulting BMUs & QEs are stored in the recall object 
    Recall.find_BMU(X, W.data(), N, M, d);
  
    // Initialize the learn history container, if it isn't already 
    if(age == 0) {
      LearnHist.Update(0, rho0, 0.0, Recall.QE, Recall.BMU, N);
      if(verbosity==2) LearnHist.Print();
    } 

    // Setup progress bar 
    progressbar pbar(n_epochs);
    pbar.set_todo_char(" ");
    pbar.set_done_char("=");
    pbar.set_opening_bracket_char("VQ Learn [");
    pbar.set_closing_bracket_char("]");


    // ** Learning loop over specified number of epochs 
    int delBMU_flag = 0, delMQE_flag = 0; 
    bool exit_flag = false; 
    unsigned int epoch = 1; 

    //for(unsigned int epoch = 1; epoch <= n_epochs; ++epoch) {
    while(!exit_flag) {

      // Anneal rho, keeping in mind its given minimum value 
      rho = rho0 * std::pow(rho_anneal, double(age));
      rho = std::max(rho_min, rho);

      // Clear containers from previous iteration
      this->reset_Update_Containers(); 

      // Update neighbor activations
      this->calc_eta_NG();

      // Update prototypes 
      this->update_W();

      // Compute new BMUs of data
      Recall.find_BMU(X, W.data(), N, M, d);

      // Update learn history
      LearnHist.Update(epoch, rho, eta_avg, Recall.QE, Recall.BMU, N);
      if(verbosity==1) pbar.update();
      if(verbosity==2) LearnHist.Print();

      // Increment age (total number of learning epochs performed)
      epoch++; 
      this->age++; 

      // Monitor consecutive convergence 
      delBMU_flag = (LearnHist.get_cur_delBMU() < conv_delBMU) ? delBMU_flag + 1 : std::max(delBMU_flag-1, 0); 
      delMQE_flag = (LearnHist.get_cur_delMQE() < conv_delMQE) ? delMQE_flag + 1 : std::max(delMQE_flag-1, 0); 

      // Check exit conditions 
      if(epoch >= n_epochs || delBMU_flag >= 3 || delMQE_flag >= 3) exit_flag = true; 

    } // close learning loop

    // Populate all recall containers from last BMU & QE 
    //Recall.cpp_RecallOnly(); 
    Recall.set_RecallContainers(this->XL); 

  }


// Return the prototpyes to python as a numpy array 
numpyCarr VQLearnWorker::get_Wnumpy() {

  return wrap_numpyCarr(W.data(), this->M, this->d); 
    
}


#endif


