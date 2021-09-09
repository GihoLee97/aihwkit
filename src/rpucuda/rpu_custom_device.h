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

#include "rng.h"
#include "rpu_pulsed_device.h"
#include "utility_functions.h"

namespace RPU {

template <typename T> class CustomRPUDevice;

BUILD_PULSED_DEVICE_META_PARAMETER(
    Custom,
    /*implements*/
    DeviceUpdateType::Custom,
    /*parameter def*/
    '''
    T ls_decrease_up = (T)0.0;
    T ls_decrease_down = (T)0.0;
    T ls_decrease_up_dtod = (T)0.05;
    T ls_decrease_down_dtod = (T)0.05;

    bool ls_allow_increasing_slope = false;
    bool ls_mean_bound_reference = true;
    bool ls_mult_noise = true;
    '''
    ,
    /*print body*/

    ss << "There is no further parameters for custom device. - giho"  << std::endl;
    ,
    /* calc weight granularity body */
    return this->dw_min;
    ,
    /*Add*/
    bool implementsWriteNoise() const override { return true; };);


template <typename T> class CustomRPUDevice : public PulsedRPUDevice<T> {

  BUILD_PULSED_DEVICE_CONSTRUCTORS(
      CustomRPUDevice,
      /* ctor*/
      '''int x_sz = this->x_size_;
      int d_sz = this->d_size_;

      w_slope_down_ = Array_2D_Get<T>(d_sz, x_sz);
      w_slope_up_ = Array_2D_Get<T>(d_sz, x_sz);

      for (int j = 0; j < x_sz; ++j) {
        for (int i = 0; i < d_sz; ++i) {
          w_slope_up_[i][j] = (T)0.0;
          w_slope_down_[i][j] = (T)0.0;
        }
      }'''
    ,
      /* dtor*/
      '''Array_2D_Free<T>(w_slope_down_);
      Array_2D_Free<T>(w_slope_up_);'''
      ,
      /* copy */
      '''for (int j = 0; j < other.x_size_; ++j) {
        for (int i = 0; i < other.d_size_; ++i) {
          w_slope_down_[i][j] = other.w_slope_down_[i][j];
          w_slope_up_[i][j] = other.w_slope_up_[i][j];
        }
      }'''
    ,
      /* move assignment */
      '''w_slope_down_ = other.w_slope_down_;
      w_slope_up_ = other.w_slope_up_;

      other.w_slope_down_ = nullptr;
      other.w_slope_up_ = nullptr;'''
      ,
      /* swap*/
      '''swap(a.w_slope_up_, b.w_slope_up_);
      swap(a.w_slope_down_, b.w_slope_down_);'''
      ,
      /* dp names*/
      '''names.push_back(std::string("slope_up"));
      names.push_back(std::string("slope_down"));'''
      ,
      /* dp2vec body*/
      '''int n_prev = (int)names.size();
      int size = this->x_size_ * this->d_size_;

      for (int i = 0; i < size; ++i) {
        data_ptrs[n_prev][i] = w_slope_up_[0][i];
        data_ptrs[n_prev + 1][i] = w_slope_down_[0][i];
      }'''
    ,
      /* vec2dp body*/
      '''int n_prev = (int)names.size();
      int size = this->x_size_ * this->d_size_;

      for (int i = 0; i < size; ++i) {
        w_slope_up_[0][i] = data_ptrs[n_prev][i];
        w_slope_down_[0][i] = data_ptrs[n_prev + 1][i];
      }'''
    ,
      /*invert copy DP */
      '''T **slope_down = rpu->getSlopeDown();
      T **slope_up = rpu->getSlopeUp();
      for (int j = 0; j < this->x_size_; ++j) {
        for (int i = 0; i < this->d_size_; ++i) {
          w_slope_down_[i][j] = -slope_up[i][j];
          w_slope_up_[i][j] = -slope_down[i][j];
        }
      }'''
  );

  void printDP(int x_count, int d_count) const override;

//   inline T **getSlopeUp() const { return w_slope_up_; };
//   inline T **getSlopeDown() const { return w_slope_down_; };

  void doSparseUpdate(
      T **weights, int i, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng)
      override;
  void doDenseUpdate(T **weights, int *coincidences, RNG<T> *rng) override;

private:
//   T **w_slope_up_ = nullptr;
//   T **w_slope_down_ = nullptr;
};
} // namespace RPU