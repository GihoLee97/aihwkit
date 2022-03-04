/**
 * (C) Copyright 2020, 2021 IBM. All Rights Reserved.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#include "pwu_kernel_parameter.h"
#include "rpu_pulsed_meta_parameter.h"
#include "rpucuda_custom_device.h"

namespace RPU {
namespace {
#define UPDATE_ONCE                                                                                \
  {                                                                                                \
    int sector = (int)((w - wmin)/(wmax - wmin)*num_sectors);                        \
    T stoch_value = curand_normal(&local_state);                                                   \
    stoch_value *= noise_std_dw;                                                                   \
    if (!(negative > 0)) {                                                                         \
        w -= (global_pars[sector*6+3+2]*w*w + global_pars[sector*6+4+2]*w + global_pars[sector*6+5+2])   \
                                        * ((T)1.0 + stoch_value);                                  \
    } else {                                                                                       \
        w += (global_pars[sector*6+2]*w*w + global_pars[sector*6+1+2]*w + global_pars[sector*6+2+2])     \
                                        * ((T)1.0 + stoch_value);                                  \
    }                                                                                              \
    w = (w > wmax) ? wmax - (T)0.000001 : w;                                                                     \
    w = (w < wmin) ? wmin + (T)0.000001: w;                                                                     \
  }
    

template <typename T> struct UpdateFunctorCustom {

  __device__ __forceinline__ void operator()(
      T &apparent_weight,
      uint32_t n,
      uint32_t negative,
      const float4 par_4,
      const float2 par_2,
      T &persistent_weight,
      const T *global_pars,
      T noise_std_dw,
      curandState &local_state)

  {
    // par_4 order (min_bound, scale_down, max_bound, scale_up )
    // global_pars see below
    // global_pars[0~]
    int num_sectors = (int)global_pars[0];
    T uw_std = global_pars[1];
    T wmax = par_4.z; //[2];
    T wmin = par_4.x; //[0];
    T b_diff = (wmax - wmin);

    T &w = uw_std > 0.0f ? persistent_weight : apparent_weight;

    if (b_diff > 0.0f) { // only do something when bounds make sense

    //   T A = negative ? global_pars[1] : global_pars[0];        // 1: up, 0: down
    //   T gamma = negative ? global_pars[3] : (-global_pars[2]); // 3: up, 2 down
    //   T a = global_pars[4];
    //   T b = global_pars[5];
    //   T dw = (negative > 0) ? (par_4.w) : (-par_4.y); // [3], [1]

      // n is larger 0 in any case
      if (n == 1) {
        UPDATE_ONCE;
      } else {
        for (int i_updates = 0; i_updates < n; i_updates++) {
          UPDATE_ONCE;
        }
      }
      // add update write noise onto apparent weight
      if (uw_std > 0) {
        T stoch_value = curand_normal(&local_state);
        apparent_weight = persistent_weight + uw_std * stoch_value;
      }
    }
  }
};
#undef UPDATE_ONCE

} // namespace

template <typename T>
pwukpvec_t<T> CustomRPUDeviceCuda<T>::getUpdateKernels(
    int m_batch, int nK32, int use_bo64, bool out_trans, const PulsedUpdateMetaParameter<T> &up) {

  pwukpvec_t<T> v;
  const auto &pars = getPar();
  //TODO 32?

  v.push_back(RPU::make_unique<
              PWUKernelParameterSingleFunctor<T, UpdateFunctorCustom<T>, 32>>(
      this->context_, this->x_size_, this->d_size_, m_batch, nK32, use_bo64, out_trans, up,
      pars.getName()));

  v.push_back(
      RPU::make_unique<PWUKernelParameterBatchFunctor<T, UpdateFunctorCustom<T>, 32>>(
          this->context_, this->x_size_, this->d_size_, m_batch, nK32, use_bo64, out_trans, up,
          pars.getName()));

  v.push_back(RPU::make_unique<
              PWUKernelParameterBatchSharedFunctor<T, UpdateFunctorCustom<T>, 32>>(
      this->context_, this->x_size_, this->d_size_, m_batch, nK32, use_bo64, out_trans, up,
      pars.getName()));

  return v;
}

template class CustomRPUDeviceCuda<float>;
#ifdef RPU_USE_DOUBLE
template class CustomRPUDeviceCuda<double>;
#endif
} // namespace RPU


// /**
//  * (C) Copyright 2020, 2021 IBM. All Rights Reserved.
//  *
//  * This code is licensed under the Apache License, Version 2.0. You may
//  * obtain a copy of this license in the LICENSE.txt file in the root directory
//  * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//  *
//  * Any modifications or derivative works of this code must retain this
//  * copyright notice, and modified files need to carry a notice indicating
//  * that they have been altered from the originals.
//  */

// #include "pwu_kernel_parameter.h"
// #include "rpu_pulsed_meta_parameter.h"
// #include "rpucuda_custom_device.h"

// namespace RPU {
// namespace {
// #define UPDATE_ONCE                                                                                \
//   {                                                                                                \
//     int sector = (int)((w - wmin - (T)0.000001)/(wmax - wmin)*num_sectors);                        \
//     T stoch_value = curand_normal(&local_state);                                                   \
//     stoch_value *= noise_std_dw;                                                                   \
//     if (!(negative > 0)) {                                                                         \
//         w -= (global_pars[sector*6+3]*w*w + global_pars[sector*6+4]*w + global_pars[sector*6+5])   \
//                                         * ((T)1.0 + stoch_value);                                  \
//     } else {                                                                                       \
//         w += (global_pars[sector*6]*w*w + global_pars[sector*6+1]*w + global_pars[sector*6+2])     \
//                                         * ((T)1.0 + stoch_value);                                  \
//     }                                                                                              \
//     w = (w > wmax) ? wmax : w;                                                                     \
//     w = (w < wmin) ? wmin : w;                                                                     \
//   }
    

// template <typename T> struct UpdateFunctorCustom {

//   __device__ __forceinline__ void operator()(
//       T &apparent_weight,
//       uint32_t n,
//       uint32_t negative,
//       const float4 par_4,
//       const float2 num_sectors_,
//       T &persistent_weight,
//       const T *global_pars,
//       T noise_std_dw,
//       curandState &local_state)

//   {
//     // par_4 order (min_bound, scale_down, max_bound, scale_up )
//     // global_pars see below
//     // global_pars[0~]
//     int num_sectors = (int)num_sectors_.x;
//     T uw_std = num_sectors_.y;
//     T wmax = par_4.z; //[2];
//     T wmin = par_4.x; //[0];
//     T b_diff = (wmax - wmin);

//     T &w = uw_std > 0.0f ? persistent_weight : apparent_weight;

//     if (b_diff > 0.0f) { // only do something when bounds make sense

//     //   T A = negative ? global_pars[1] : global_pars[0];        // 1: up, 0: down
//     //   T gamma = negative ? global_pars[3] : (-global_pars[2]); // 3: up, 2 down
//     //   T a = global_pars[4];
//     //   T b = global_pars[5];
//     //   T dw = (negative > 0) ? (par_4.w) : (-par_4.y); // [3], [1]

//       // n is larger 0 in any case
//       if (n == 1) {
//         UPDATE_ONCE;
//       } else {
//         for (int i_updates = 0; i_updates < n; i_updates++) {
//           UPDATE_ONCE;
//         }
//       }
//       // add update write noise onto apparent weight
//       if (uw_std > 0) {
//         T stoch_value = curand_normal(&local_state);
//         apparent_weight = persistent_weight + uw_std * stoch_value;
//       }
//     }
//   }
// };
// #undef UPDATE_ONCE

// // #define UPDATE_ONCE_COMPLEX_NOISE                                                                  \
// //   {                                                                                                \
// //     T z = 2.0 * w / b_diff * a + b;                                                                \
// //     T y = 1.0 - A * __expf(gamma * z);                                                             \
// //     if (y > 0.0f) {                                                                                \
// //       T dw_act = y * dw;                                                                           \
// //       T stoch_value = curand_normal(&local_state);                                                 \
// //       stoch_value *= noise_std_dw * (fabs(dw_act) + dw_std_add + dw_std_slope * fabs(w));          \
// //       w += dw_act + stoch_value;                                                                   \
// //       w = (w > wmax) ? wmax : w;                                                                   \
// //       w = (w < wmin) ? wmin : w;                                                                   \
// //     }                                                                                              \
// //   }

// // template <typename T> struct UpdateFunctorCustomComplexNoise {

// //   __device__ __forceinline__ void operator()(
// //       T &apparent_weight,
// //       uint32_t n,
// //       uint32_t negative,
// //       const float4 par_4,
// //       const float2 par_2,
// //       T &persistent_weight,
// //       const T *global_pars,
// //       T noise_std_dw,
// //       curandState &local_state)

// //   {
// //     // par_4 order (min_bound, scale_down, max_bound, scale_up )
// //     // global_pars see below
// //     T uw_std = global_pars[6];
// //     T dw_std_add = global_pars[7];
// //     T dw_std_slope = global_pars[8];
// //     T wmax = par_4.z; //[2];
// //     T wmin = par_4.x; //[0];
// //     T b_diff = (wmax - wmin);

// //     T &w = uw_std > 0.0f ? persistent_weight : apparent_weight;

// //     if (b_diff > 0.0f) { // only do something when bounds make sense

// //       T A = negative ? global_pars[1] : global_pars[0];        // 1: up, 0: down
// //       T gamma = negative ? global_pars[3] : (-global_pars[2]); // 3: up, 2 down
// //       T a = global_pars[4];
// //       T b = global_pars[5];
// //       T dw = (negative > 0) ? (par_4.w) : (-par_4.y); // [3], [1]

// //       // n is larger 0 in any case
// //       if (n == 1) {
// //         UPDATE_ONCE_COMPLEX_NOISE;
// //       } else {
// //         for (int i_updates = 0; i_updates < n; i_updates++) {
// //           UPDATE_ONCE_COMPLEX_NOISE;
// //         }
// //       }
// //       // add update write noise onto apparent weight
// //       if (uw_std > 0) {
// //         T stoch_value = curand_normal(&local_state);
// //         apparent_weight = persistent_weight + uw_std * stoch_value;
// //       }
// //     }
// //   }
// // };
// // #undef UPDATE_ONCE_COMPLEX_NOISE

// } // namespace

// template <typename T>
// pwukpvec_t<T> CustomRPUDeviceCuda<T>::getUpdateKernels(
//     int m_batch, int nK32, int use_bo64, bool out_trans, const PulsedUpdateMetaParameter<T> &up) {

//   pwukpvec_t<T> v;
//   const auto &pars = getPar();

//   v.push_back(RPU::make_unique<
//               PWUKernelParameterSingleFunctor<T, UpdateFunctorCustom<T>, 30>>(
//       this->context_, this->x_size_, this->d_size_, m_batch, nK32, use_bo64, out_trans, up,
//       pars.getName()));

//   v.push_back(
//       RPU::make_unique<PWUKernelParameterBatchFunctor<T, UpdateFunctorCustom<T>, 30>>(
//           this->context_, this->x_size_, this->d_size_, m_batch, nK32, use_bo64, out_trans, up,
//           pars.getName()));

//   v.push_back(RPU::make_unique<
//               PWUKernelParameterBatchSharedFunctor<T, UpdateFunctorCustom<T>, 30>>(
//       this->context_, this->x_size_, this->d_size_, m_batch, nK32, use_bo64, out_trans, up,
//       pars.getName()));

//   return v;
// }

// template class CustomRPUDeviceCuda<float>;
// #ifdef RPU_USE_DOUBLE
// template class CustomRPUDeviceCuda<double>;
// #endif
// } // namespace RPU
