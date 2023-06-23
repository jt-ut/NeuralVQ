#include "include/pybind_custom_types.hpp"
#include "include/VQRecall.hpp"
#include "include/VQLearn.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_MODULE(_cpp_NVQ, m)
{

  m.attr("__name__") = "NeuralVQ._cpp_NVQ";

    // ** Recall 
  py::class_<VQRecall>(m, "VQRecall")
    // Constructor 
    .def(py::init<int, int>(), py::arg("nBMU")=int(2), py::arg("nAnnoyTrees")=int(50)) 

    // Methods 
    .def("Recall", &VQRecall::Recall, py::arg("X"), py::arg("W"), py::arg("XL")=XL_empty, py::arg("BMU_only")=false)
    // Attributes, set during construction 
    .def_readonly("nBMU", &VQRecall::nBMU)
    .def_readonly("nAnnoyTrees", &VQRecall::nAnnoyTrees)
    // Attributes, calc'd during method calls 
    .def_readonly("BMU", &VQRecall::BMU)
    .def_readonly("QE", &VQRecall::QE)
    .def_readonly("RF", &VQRecall::RF)
    .def_readonly("RF_Size", &VQRecall::RF_Size)
    .def_readonly("CADJi", &VQRecall::CADJi)
    .def_readonly("CADJj", &VQRecall::CADJj)
    .def_readonly("CADJ", &VQRecall::CADJ)
    .def_readonly("RFL_Dist", &VQRecall::RFL_Dist)
    .def_readonly("RFL", &VQRecall::RFL)
    .def_readonly("RFL_Purity", &VQRecall::RFL_Purity)
    .def_readonly("RFL_Purity_UOA", &VQRecall::RFL_Purity_UOA)
    .def_readonly("RFL_Purity_WOA", &VQRecall::RFL_Purity_WOA);


    // ** Learn History 
    py::class_<LearnHistoryContainer>(m, "LearnHistory")
    .def(py::init<>())
    .def_readonly("Epoch", &LearnHistoryContainer::Epoch)
    .def_readonly("rho", &LearnHistoryContainer::rho)
    .def_readonly("NbEff", &LearnHistoryContainer::NbEff)
    .def_readonly("MQE", &LearnHistoryContainer::MQE)
    .def_readonly("delMQE", &LearnHistoryContainer::delMQE)
    .def_readonly("delBMU", &LearnHistoryContainer::delBMU);
  
  
    // ** Learn 
  py::class_<VQLearnWorker>(m, "VQLearn")
    // Constructor 
    .def(py::init<const numpyCarr&, const numpyCarr&, const std::vector<int>&, double, double, double, double, int>(), 
    py::arg("X"), py::arg("W"), py::arg("XL") = XL_empty, 
    py::arg("rho0")=double(-1.0), py::arg("rho_anneal")=0.95, py::arg("rho_min")=double(0.75), py::arg("eta_min") = 0.01, py::arg("verbosity") = 2) 

    // Methods 
    .def("learn", &VQLearnWorker::learn, py::arg("n_epochs"), 
      py::arg("conv_delBMU")=std::numeric_limits<double>::max(), 
      py::arg("conv_delMQE")=std::numeric_limits<double>::max())
    // Regurgitated attributes, set during construction 
    .def_readonly("N", &VQLearnWorker::N)
    .def_readonly("M", &VQLearnWorker::M)
    .def_readonly("d", &VQLearnWorker::d)
    .def_readonly("rho0", &VQLearnWorker::rho0)
    .def_readonly("rho_anneal", &VQLearnWorker::rho_anneal)
    .def_readonly("rho_min", &VQLearnWorker::rho_min)
    .def_readonly("eta_min", &VQLearnWorker::eta_min)

    // Attributes set during learning 
    .def_readonly("age", &VQLearnWorker::age)    
    .def_readonly("eta", &VQLearnWorker::eta)
    .def_readonly("eta_idx", &VQLearnWorker::eta_idx)
  
    .def("W", &VQLearnWorker::get_Wnumpy)
    .def_readonly("LearnHist", &VQLearnWorker::LearnHist)
    .def_readonly("Recall", &VQLearnWorker::Recall);
}