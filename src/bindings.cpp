#include "../include/VQRecall.hpp"
#include "../include/VQLearn.hpp"
#include "../include/pybind11/pybind11.h"
#include "../include/pybind11/stl.h"
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;
typedef py::array_t<double,  py::array::c_style> numpyarr; 

PYBIND11_MODULE(NeuralVQ, m)
{

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


  
    // ** Learn 
  py::class_<AnnoyBatchLearnWorker>(m, "VQLearn")
    // Constructor 
    .def(py::init<const numpyarr&, numpyarr, double, double, double, double>(), 
    py::arg("X"), py::arg("W"), 
    py::arg("rho0")=double(-1.0), py::arg("rho_anneal")=0.95, py::arg("rho_min")=double(0.75), py::arg("min_h") = 0.01) 

    // Methods 
    .def("train", &AnnoyBatchLearnWorker::train, py::arg("n_epochs"))
    // Attributes, set during construction 
    .def_readonly("N", &AnnoyBatchLearnWorker::N)
    .def_readonly("M", &AnnoyBatchLearnWorker::M)
    .def_readonly("d", &AnnoyBatchLearnWorker::d)

    .def_readonly("age", &AnnoyBatchLearnWorker::age)
//     //.def_property_readonly("X", &AnnoyBatchLearnWorker::X)
//     //.def_property_readonly("X", &AnnoyBatchLearnWorker::X, py::return_value_policy::reference_internal)
//     .def_readonly("W", &AnnoyBatchLearnWorker::W)
//     .def_readonly("n_epochs", &AnnoyBatchLearnWorker::n_epochs)
    
    .def_readonly("rho0", &AnnoyBatchLearnWorker::rho0)
    .def_readonly("rho_anneal", &AnnoyBatchLearnWorker::rho_anneal)
    .def_readonly("rho_min", &AnnoyBatchLearnWorker::rho_min)
    .def_readonly("min_h", &AnnoyBatchLearnWorker::min_h)

    .def_readonly("h", &AnnoyBatchLearnWorker::h)
    .def_readonly("hidx", &AnnoyBatchLearnWorker::hidx)
    .def_readonly("XBMU", &AnnoyBatchLearnWorker::XBMU)
    .def_readonly("XQE", &AnnoyBatchLearnWorker::XQE)

    .def("W", &AnnoyBatchLearnWorker::get_W);
}