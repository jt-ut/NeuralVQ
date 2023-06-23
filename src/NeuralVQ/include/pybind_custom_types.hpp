#ifndef NEURALVQ_PYTYPES_HPP
#define NEURALVQ_PYTYPES_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

typedef py::array_t<double,  py::array::c_style> numpyCarr; 

// Return pointer to a numpy array's data, + the array size 
inline void unwrap_numpyCarr(double*& data_ptr, unsigned int& N, unsigned int& d, const numpyCarr& arr) {

    // Open buffer 
    py::buffer_info buffer = arr.request();

    // Set array dimensions 
    N = buffer.shape[0]; 
    d = buffer.shape[1]; 
    
    // Strip out pointers 
    data_ptr = static_cast<double *>(buffer.ptr);
    
    return; 
}


// Overload: Extract the data in a numpy array into a std::vector 
inline void unwrap_numpyCarr(std::vector<double>& data_vec, unsigned int& N, unsigned int& d, const numpyCarr& arr) {

    // Open buffer 
    py::buffer_info buffer = arr.request();

    // Set array dimensions 
    N = buffer.shape[0]; 
    d = buffer.shape[1]; 
    
    // Strip out pointers 
    double* data_ptr = static_cast<double *>(buffer.ptr);

    data_vec = std::vector<double>(data_ptr, data_ptr+(N*d));
    
    return; 
}


inline numpyCarr wrap_numpyCarr(double* data_ptr, unsigned int N, unsigned int d) {
    
    numpyCarr out = numpyCarr(
        {N, d}, // set array dimensions 
        {int(d * 8), 8}, // set strides, 8 corresponds to double precision 
        data_ptr
    ); 

    return out;     
}


#endif 