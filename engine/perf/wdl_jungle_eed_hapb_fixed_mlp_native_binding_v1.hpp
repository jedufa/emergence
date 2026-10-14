#ifndef EIEL_WDL_JUNGLE_EED_HAPB_FIXED_MLP_NATIVE_BINDING_V1_HPP_
#define EIEL_WDL_JUNGLE_EED_HAPB_FIXED_MLP_NATIVE_BINDING_V1_HPP_

#include <pybind11/pybind11.h>

namespace eiel::cuda_architecture_b::hapb_fixed_mlp_native_binding_v1 {

void bind_hapb_fixed_mlp_native_v1(pybind11::module_& module);

}  // namespace eiel::cuda_architecture_b::hapb_fixed_mlp_native_binding_v1

#endif
