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

#pragma once

#include "pwu_kernel_parameter_base.h"
#include "rpu_custom_device.h"
#include "rpucuda_pulsed_device.h"
#include <memory>

namespace RPU {

template <typename T> class CustomRPUDeviceCuda : public PulsedRPUDeviceCuda<T> {

public:
  BUILD_PULSED_DEVICE_CONSTRUCTORS_CUDA(
      CustomRPUDeviceCuda,
      CustomRPUDevice,
      /*ctor body*/
      dev_cu_par_ = RPU::make_unique<CudaArray<float>>(this->context_, this->num_sectors_*6+2);
      //dev_slope_ = RPU::make_unique<CudaArray<float>>(this->context_, 2 * this->size_);
      //dev_write_noise_std_ = RPU::make_unique<CudaArray<T>>(this->context_, 1);
      ,
      /*dtor body*/
      ,
      /*copy body*/
      dev_cu_par_->assign(*other.dev_cu_par_);
      //dev_slope_->assign(*other.dev_slope_);
      //dev_write_noise_std_->assign(*other.dev_write_noise_std_);
      ,
      /*move assigment body*/
      dev_cu_par_ = std::move(other.dev_cu_par_);
      //dev_slope_ = std::move(other.dev_slope_);
      //dev_write_noise_std_ = std::move(other.dev_write_noise_std_);
      ,
      /*swap body*/
      swap(a.dev_cu_par_, b.dev_cu_par_);
      //swap(a.dev_slope_, b.dev_slope_);
      //swap(a.dev_write_noise_std_, b.dev_write_noise_std_);
      ,
      /*host copy from cpu (rpu_device). Parent device params are copyied automatically*/
      num_sectors_ = rpu_device.getNumSectors();

      T ***w_coeff_up_a_ = rpu_device.getCoeffUpA();
      T ***w_coeff_up_b_ = rpu_device.getCoeffUpB();
      T ***w_coeff_up_c_ = rpu_device.getCoeffUpC();
      T ***w_coeff_down_a_ = rpu_device.getCoeffDownA();
      T ***w_coeff_down_b_ = rpu_device.getCoeffDownB();
      T ***w_coeff_down_c_ = rpu_device.getCoeffDownC();
      float *tmp_cu_par = new float[32];
      dev_cu_par_ = RPU::make_unique<CudaArray<float>>(this->context_, 32);

      tmp_cu_par[0] = (float)rpu_device.getNumSectors();
      tmp_cu_par[1] = getPar().getScaledWriteNoise();
      for (int k = 0; k < num_sectors_; ++k){
        tmp_cu_par[k*6+2] = w_coeff_up_a_[0][0][k];
        tmp_cu_par[k*6+1+2] = w_coeff_up_b_[0][0][k];
        tmp_cu_par[k*6+2+2] = w_coeff_up_c_[0][0][k];
        tmp_cu_par[k*6+3+2] = w_coeff_down_a_[0][0][k];
        tmp_cu_par[k*6+4+2] = w_coeff_down_b_[0][0][k];
        tmp_cu_par[k*6+5+2] = w_coeff_down_c_[0][0][k];
      }
      dev_cu_par_->assign(tmp_cu_par);
      
      this->context_->synchronize();
      delete[] tmp_cu_par;
      
      /*
      int d_size = this->d_size_;
      int x_size = this->x_size_;
      T **w_slope_up = rpu_device.getSlopeUp();
      T **w_slope_down = rpu_device.getSlopeDown();
      float *tmp_slope = new float[2 * this->size_];

      for (int i = 0; i < d_size; ++i) {
        for (int j = 0; j < x_size; ++j) {
          int kk = j * (d_size * 2) + 2 * i;
          tmp_slope[kk] = w_slope_down[i][j];
          tmp_slope[kk + 1] = w_slope_up[i][j];
        }
      } dev_slope_->assign(tmp_slope);
      dev_write_noise_std_->setConst(getPar().getScaledWriteNoise());

      this->context_->synchronize();
      delete[] tmp_slope;
      */
      );

  pwukpvec_t<T> getUpdateKernels(
      int m_batch,
      int nK32,
      int use_bo64,
      bool out_trans,
      const PulsedUpdateMetaParameter<T> &up) override;
  T *getGlobalParamsData() override { return dev_cu_par_->getData(); };
  T *get1ParamsData() override {
    return getPar().usesPersistentWeight() ? this->dev_persistent_weights_->getData() : nullptr;
  };

private:
  std::unique_ptr<CudaArray<float>> dev_cu_par_ = nullptr;
//   std::unique_ptr<CudaArray<T>> dev_write_noise_std_ = nullptr;
  int num_sectors_ = 5;
};

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

// #pragma once

// #include "pwu_kernel_parameter_base.h"
// #include "rpu_custom_device.h"
// #include "rpucuda_pulsed_device.h"
// #include <memory>

// namespace RPU {

// template <typename T> class CustomRPUDeviceCuda : public PulsedRPUDeviceCuda<T> {

// public:
//   BUILD_PULSED_DEVICE_CONSTRUCTORS_CUDA(
//       CustomRPUDeviceCuda,
//       CustomRPUDevice,
//       /*ctor body*/
//       dev_cu_par_ = RPU::make_unique<CudaArray<float>>(this->context_, 30);
//       dev_cu_num_sectors_ = RPU::make_unique<CudaArray<float>>(this->context_, 2);
//     //   dev_slope_ = RPU::make_unique<CudaArray<float>>(this->context_, 2 * this->size_);
//     //   dev_write_noise_std_ = RPU::make_unique<CudaArray<T>>(this->context_, 1);
//       ,
//       /*dtor body*/
//       ,
//       /*copy body*/
//       dev_cu_par_->assign(*other.dev_cu_par_);
//       dev_cu_num_sectors_->assign(*other.dev_cu_num_sectors_);
//       //dev_slope_->assign(*other.dev_slope_);
//       //dev_write_noise_std_->assign(*other.dev_write_noise_std_);
//       ,
//       /*move assigment body*/
//       dev_cu_par_ = std::move(other.dev_cu_par_);
//       dev_cu_num_sectors_ = std::move(other.dev_cu_num_sectors_);      
//       //dev_slope_ = std::move(other.dev_slope_);
//       //dev_write_noise_std_ = std::move(other.dev_write_noise_std_);
//       ,
//       /*swap body*/
//       swap(a.dev_cu_par_, b.dev_cu_par_);
//       swap(a.dev_cu_num_sectors_, b.dev_cu_num_sectors_);
//       //swap(a.dev_slope_, b.dev_slope_);
//       //swap(a.dev_write_noise_std_, b.dev_write_noise_std_);
//       ,
//       /*host copy from cpu (rpu_device). Parent device params are copyied automatically*/
//       int d_size = this->d_size_;
//       int x_size = this->x_size_;
//       int num_sectors = rpu_device.getNumSectors();
//       // RPU_FATAL("giho-for using cuda version custom device you need to configure the number of sectors at most 5");
//       float *tmp_cu_num_sectors_ = new float[2];
//       tmp_cu_num_sectors_[0] = (float)rpu_device.getNumSectors();
//       tmp_cu_num_sectors_[1] = getPar().getScaledWriteNoise();
//       dev_cu_num_sectors_->assign(tmp_cu_num_sectors_);
//       delete[] tmp_cu_num_sectors_;

//       T ***w_coeff_up_a_ = rpu_device.getCoeffUpA();
//       T ***w_coeff_up_b_ = rpu_device.getCoeffUpB();
//       T ***w_coeff_up_c_ = rpu_device.getCoeffUpC();
//       T ***w_coeff_down_a_ = rpu_device.getCoeffDownA();
//       T ***w_coeff_down_b_ = rpu_device.getCoeffDownB();
//       T ***w_coeff_down_c_ = rpu_device.getCoeffDownC();
//       float *tmp_cu_par = new float[6 * num_sectors];
//       dev_cu_par_ = RPU::make_unique<CudaArray<float>>(this->context_, 6 * num_sectors);

//       for (int i = 0; i < d_size; ++i) {
//         for (int j = 0; j < x_size; ++j) {
//             for (int k = 0; k < num_sectors; ++k){
//                 // int zz = k * (d_size * x_size * 6) 
//                 //         + j * (d_size * 6) 
//                 //         + i * 6;
//                 tmp_cu_par[k*6] += w_coeff_up_a_[i][j][k];
//                 tmp_cu_par[k*6+1] += w_coeff_up_b_[i][j][k];
//                 tmp_cu_par[k*6+2] += w_coeff_up_c_[i][j][k];
//                 tmp_cu_par[k*6+3] += w_coeff_down_a_[i][j][k];
//                 tmp_cu_par[k*6+4] += w_coeff_down_b_[i][j][k];
//                 tmp_cu_par[k*6+5] += w_coeff_down_c_[i][j][k];
//             }
//         }
//       }
//       for (int i = 0; i < num_sectors*6; ++i){
//           tmp_cu_par[i] /= d_size * x_size;
//       }
//       dev_cu_par_->assign(tmp_cu_par);
      
//       this->context_->synchronize();
//       delete[] tmp_cu_par;
      
//       /*
//       int d_size = this->d_size_;
//       int x_size = this->x_size_;
//       T **w_slope_up = rpu_device.getSlopeUp();
//       T **w_slope_down = rpu_device.getSlopeDown();
//       float *tmp_slope = new float[2 * this->size_];

//       for (int i = 0; i < d_size; ++i) {
//         for (int j = 0; j < x_size; ++j) {
//           int kk = j * (d_size * 2) + 2 * i;
//           tmp_slope[kk] = w_slope_down[i][j];
//           tmp_slope[kk + 1] = w_slope_up[i][j];
//         }
//       } dev_slope_->assign(tmp_slope);
//       dev_write_noise_std_->setConst(getPar().getScaledWriteNoise());

//       this->context_->synchronize();
//       delete[] tmp_slope;
//       */
//       );

//   pwukpvec_t<T> getUpdateKernels(
//       int m_batch,
//       int nK32,
//       int use_bo64,
//       bool out_trans,
//       const PulsedUpdateMetaParameter<T> &up) override;
//   T *getGlobalParamsData() override { return dev_cu_par_->getData(); };
//   float *get2ParamsData() override { return dev_cu_num_sectors_->getData(); };
//   T *get1ParamsData() override {
//     return getPar().usesPersistentWeight() ? this->dev_persistent_weights_->getData() : nullptr;
//   };

// private:
//   std::unique_ptr<CudaArray<float>> dev_cu_par_ = nullptr;
// //   std::unique_ptr<CudaArray<T>> dev_write_noise_std_ = nullptr;
//   std::unique_ptr<CudaArray<float>> dev_cu_num_sectors_ = nullptr;
// };

// } // namespace RPU


      // for (int i = 0; i < d_size; ++i) {
      //   for (int j = 0; j < x_size; ++j) {
      //       for (int k = 0; k < num_sectors_; ++k){
      //           // int zz = k * (d_size * x_size * 6) 
      //           //         + j * (d_size * 6) 
      //           //         + i * 6;
      //           tmp_cu_par[k*6+2] = w_coeff_up_a_[0][0][k];
      //           tmp_cu_par[k*6+1+2] = w_coeff_up_b_[0][0][k];
      //           tmp_cu_par[k*6+2+2] = w_coeff_up_c_[0][0][k];
      //           tmp_cu_par[k*6+3+2] = w_coeff_down_a_[0][0][k];
      //           tmp_cu_par[k*6+4+2] = w_coeff_down_b_[0][0][k];
      //           tmp_cu_par[k*6+5+2] = w_coeff_down_c_[0][0][k];
      //       }
      //   }
      // }
      
      
// for (int i = 2; i < num_sectors_*6 + 2; ++i){
//       //     // tmp_cu_par[i] /= d_size * x_size;
//           std::cout << "i : " << i << " value : " << tmp_cu_par[i] << std::endl;
//       }
