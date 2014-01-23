/*
   Copyright (c) by respective owners including Yahoo!, Microsoft, and
   individual contributors. All rights reserved.  Released under a BSD (revised)
   license as described in the file LICENSE.
   */
#include <fstream>
#include <float.h>
#ifndef _WIN32
#include <netdb.h>
#endif
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <sys/timeb.h>
#include "parse_example.h"
#include "constant.h"
#include "sparse_dense.h"
#include "owlqn.h"
#include "cache.h"
#include "simple_label.h"
#include "accumulate.h"
#include <exception>

using namespace std;

#define MEM_GT 0 // last m history of loss gradient(GT)
#define MEM_XT 1 // last m history of parameter(XT)
#define MEM_YT 0 // last m history of loss gradient delta(YT) // GT(k+1) - GT(k)
#define MEM_ST 1 // last m history of parameter delta(ST) // XT(k+1) - XT(k)

#define W_XT 0   // current parameter w(XT)
#define W_GT 1   // current gradient  g(GT)
#define W_DIR 2  // current direction g*H-1(DIR)
#define W_PGT 3  // current pesudo gradient pg(PGT)
#define W_OT 4   // current orthant to explore ot(OT)
#define W_XP 5   // previous parameter wp(XP)
#define W_COND 6 // current preconditioner(COND)

#define LEARN_OK 0
#define LEARN_CURV 1
#define LEARN_CONV 2

class curv_exception: public exception {} curv_ex_owlqn;

/********************************************************************/
/* mem & w definition ***********************************************/
/********************************************************************/ 
// mem[2*i] = y_t
// mem[2*i+1] = s_t
//
// w[0] = weight
// w[1] = accumulated first derivative
// w[2] = step direction
// w[3] = pesudo gradient
// w[4] = orthant to explore
// w[5] = previous weight
// w[6] = pre-conditioner

namespace OWLQN {

  //nonrentrant
  struct bfgs {
    double wolfe1_bound;

    struct timeb t_start, t_end;
    double net_comm_time;

    struct timeb t_start_global, t_end_global;
    double net_time;

    size_t example_number;
    size_t current_pass;

    // set by initializer
    int mem_stride;
    bool output_regularizer;
    float* mem;
    double* rho;
    double* alpha;

    weight* regularizers;

    // the below needs to be included when resetting, in addition to derivative
    int lastj, origin;
    double loss_sum, previous_loss_sum;
    float step_size;
    double importance_weight_sum;

    // first pass specification
    bool first_pass;
    bool gradient_pass;
  };

  const char* curv_message = "Zero or negative curvature detected.\n"
    "To increase curvature you can increase regularization or rescale features.\n"
    "It is also possible that you have reached numerical accuracy\n"
    "and further decrease in the objective cannot be reliably detected.\n";

  void zero_derivative(vw& all) { //set derivative to 0.
    uint32_t length = 1 << all.num_bits;
    size_t stride = all.stride;
    weight* weights = all.reg.weight_vector;
    for(uint32_t i = 0; i < length; i++)
      weights[stride*i + W_GT] = 0;
  }

  void zero_state(vw& all) {
    uint32_t length = 1 << all.num_bits;
    size_t stride = all.stride;
    weight* weights = all.reg.weight_vector;
    for(uint32_t i = 0; i < length; i++) {
      weights[stride*i + W_GT] = 0;
      weights[stride*i + W_DIR] = 0;
      weights[stride*i + W_PGT] = 0;
      weights[stride*i + W_OT] = 0;
      weights[stride*i + W_XP] = 0;
      //weights[stride*i + W_COND] = 0;
    }
  }
  
  void reset_state(vw& all, bfgs& b, bool zero) {
    b.lastj = b.origin = 0;
    b.loss_sum = b.previous_loss_sum = 0.;
    b.importance_weight_sum = 0.;
    b.first_pass = true;
    b.gradient_pass = true;
    if (zero) {
      zero_state(all);
      //zero_derivative(all);
    }
  }

  // use in gradient prediction
  void quad_grad_update(weight* weights, feature& page_feature, 
      v_array<feature> &offer_features, size_t mask, float g) {
    size_t halfhash = quadratic_constant * page_feature.weight_index;
    float update = g * page_feature.x;
    for (feature* ele = offer_features.begin; ele != offer_features.end; ele++)
    {
      weight* w=&weights[(halfhash + ele->weight_index) & mask];
      w[W_GT] += update * ele->x;
    }
  }

  void cubic_grad_update(weight* weights, feature& f0, feature& f1,
      v_array<feature> &cross_features, size_t mask, float g) {
    size_t halfhash = cubic_constant2 * (cubic_constant * f0.weight_index + f1.weight_index);
    float update = g * f0.x * f1.x;
    for (feature* ele = cross_features.begin; ele != cross_features.end; ele++) {
      weight* w=&weights[(halfhash + ele->weight_index) & mask];
      w[W_GT] += update * ele->x;
    }
  }


  bool test_example(example* ec) {
    return ((label_data*)ec->ld)->label == FLT_MAX;
  }

  float bfgs_predict(vw& all, example* &ec) {
    ec->partial_prediction = GD::inline_predict<vec_add>(all,ec);
    return GD::finalize_prediction(all, ec->partial_prediction);
  }

  float predict_and_gradient(vw& all, example* &ec) {
    float fp = bfgs_predict(all, ec);

    label_data* ld = (label_data*)ec->ld;
    all.set_minmax(all.sd, ld->label);

    float loss_grad = all.loss->first_derivative(all.sd, fp,ld->label)
      * ld->weight;

    size_t mask = all.weight_mask;
    weight* weights = all.reg.weight_vector;
    for (unsigned char* i = ec->indices.begin; i != ec->indices.end; i++) {
      feature *f = ec->atomics[*i].begin;
      for (; f != ec->atomics[*i].end; f++) {
        weight* w = &weights[f->weight_index & mask];
        w[W_GT] += loss_grad * f->x;
      }
    }

    // bi-gram feature
    for (vector<string>::iterator i = all.pairs.begin(); i != all.pairs.end(); i++) {
      if (ec->atomics[(int)(*i)[0]].size() > 0) {
        v_array<feature> temp = ec->atomics[(int)(*i)[0]];
        for (; temp.begin != temp.end; temp.begin++)
          quad_grad_update(weights, *temp.begin, ec->atomics[(int)(*i)[1]], mask, loss_grad);
      } 
    }

    // tri-gram feature
    for (vector<string>::iterator i = all.triples.begin(); i != all.triples.end();i++) {
      if ((ec->atomics[(int)(*i)[0]].size() == 0) 
          || (ec->atomics[(int)(*i)[1]].size() == 0) 
          || (ec->atomics[(int)(*i)[2]].size() == 0)) { 
        continue; 
      }
      v_array<feature> temp1 = ec->atomics[(int)(*i)[0]];
      for (; temp1.begin != temp1.end; temp1.begin++) {
        v_array<feature> temp2 = ec->atomics[(int)(*i)[1]];
        for (; temp2.begin != temp2.end; temp2.begin++)
          cubic_grad_update(weights, *temp1.begin, *temp2.begin, ec->atomics[(int)(*i)[2]], mask, loss_grad);
      }
    }
    return fp;
  }

  float compute_magnitude(vw& all, int offset) {
    double ret = 0.;
    uint32_t length = 1 << all.num_bits;
    size_t stride = all.stride;
    weight* weights = all.reg.weight_vector;
    for(uint32_t i = 0; i < length; i++) {
      ret += weights[stride * i + offset] * weights[stride * i + offset];
    }
    return (float)ret;
  }

  void bfgs_iter_start(vw& all, bfgs& b, float* mem, int& lastj, double importance_weight_sum, int&origin) {
    uint32_t length = 1 << all.num_bits;
    size_t stride = all.stride;
    weight* w = all.reg.weight_vector;

    //double g1_Hg1 = 0.;
    //double g1_g1 = 0.;

    origin = 0;
    for(uint32_t i = 0; i < length; i++, mem += b.mem_stride, w += stride) {
      mem[(MEM_XT + origin) % b.mem_stride] = w[W_XT];
      mem[(MEM_GT + origin) % b.mem_stride] = w[W_GT];
      // initial Hg and g, seems no place to use
      //g1_Hg1 += w[W_GT] * w[W_GT]; 
      //g1_g1 += w[W_GT] * w[W_GT];
      // initial direction
      w[W_DIR] = - w[W_GT]; 
      w[W_GT] = 0;
    }
    lastj = 0;
  }

  //origin = (origin + mem_stride - 2) % mem_stride; 类似1个环形队列
  //同时将梯度归零，与zero_derivative做的事一样
  //lastj = (lastj < all.m - 1) ? lastj + 1 : all.m - 1;
  //最开始时：lastj为零
  //更新w[W_DIR]，同时w[W_GT] -> zero
  void bfgs_iter_middle(vw& all, bfgs& b, float* mem, double* rho, double* alpha, 
      int& lastj, int &origin) {  
    uint32_t length = 1 << all.num_bits;
    size_t stride = all.stride;
    weight* w = all.reg.weight_vector;

    float* mem0 = mem;
    float* w0 = w;

    if (all.m <= 0) {
      fprintf(stderr, "bfgs need history length > 0\n");
    }

    double y_s = 0.;  // YT .* ST
    double y_Hy = 0.; // YT .* YT
    double s_q = 0.;  // st .* DIR  //NOTE: q = DIR, not GT

    for(uint32_t i = 0; i < length; i++, mem+=b.mem_stride, w+=stride) {
      mem[(MEM_YT+origin)%b.mem_stride] = w[W_GT] - mem[(MEM_GT+origin)%b.mem_stride];
      mem[(MEM_ST+origin)%b.mem_stride] = w[W_XT] - mem[(MEM_XT+origin)%b.mem_stride];

      // direction is opposite to gradient
      if (all.l1_lambda > 0.0) {
        w[W_DIR] = -w[W_PGT];   
      } else {
        w[W_DIR] = -w[W_GT]; 
      }

      y_s += mem[(MEM_YT + origin) % b.mem_stride] 
        * mem[(MEM_ST + origin) % b.mem_stride];
      y_Hy += mem[(MEM_YT + origin) % b.mem_stride]
        * mem[(MEM_YT + origin) % b.mem_stride]; 
      s_q += mem[(MEM_ST + origin) % b.mem_stride] * w[W_DIR];  
    }

    if (y_s <= 0. || y_Hy <= 0.)
      throw curv_ex_owlqn;
    rho[0] = 1 / y_s;

    // initial hessian matrix = gamma * I
    double gamma = y_s / y_Hy;  

    // two-loops: alpha = rho * ST * q ; (q = wDIR)
    for (int j = 0; j < lastj; j++) {
      alpha[j] = rho[j] * s_q;
      s_q = 0.;
      mem = mem0;
      w = w0;
      for(uint32_t i = 0; i < length; i++, mem += b.mem_stride, w += stride) {
        w[W_DIR] -= (float)alpha[j] * mem[(2 * j + MEM_YT + origin) % b.mem_stride];
        s_q += mem[(2 * j + 2 + MEM_ST + origin) % b.mem_stride] * w[W_DIR];
      }
    }
    alpha[lastj] = rho[lastj] * s_q;

    // two-loops: q = q - alpha * YT
    double y_r = 0.;  // YT .* dir
    mem = mem0;
    w = w0;
    for(uint32_t i = 0; i < length; i++, mem += b.mem_stride, w += stride) {
      w[W_DIR] -= (float)alpha[lastj] * mem[(2 * lastj + MEM_YT + origin) % b.mem_stride];
      w[W_DIR] *= (float)gamma;
      y_r += mem[(2 * lastj + MEM_YT + origin) % b.mem_stride] * w[W_DIR];
    }

    // two-loops: coef = alpha - beta; (beta = rho * y_r)
    double coef_j = 0.;  
    for (int j=lastj; j>0; j--) {
      coef_j = alpha[j] - rho[j] * y_r;
      y_r = 0.;
      mem = mem0;
      w = w0;
      for(uint32_t i = 0; i < length; i++, mem+=b.mem_stride, w+=stride) {
        w[W_DIR] += (float)coef_j*mem[(2*j+MEM_ST+origin)%b.mem_stride];
        y_r += mem[(2*j-2+MEM_YT+origin)%b.mem_stride]*w[W_DIR];
      }
    }
    coef_j = alpha[0] - rho[0] * y_r;

    // two-loops: dir = dir + ST * coef
    mem = mem0;
    w = w0;
    for(uint32_t i = 0; i < length; i++, mem += b.mem_stride, w += stride) {
      w[W_DIR] += (float) coef_j * mem[(MEM_ST + origin) % b.mem_stride];
    }

    /************************
     ** shift memory history 
     ************************/
    mem = mem0;
    w = w0;
    lastj = (lastj < all.m - 1) ? lastj + 1 : all.m - 1;
    origin = (origin + b.mem_stride - 2) % b.mem_stride;
    for(uint32_t i = 0; i < length; i++, mem += b.mem_stride, w += stride) {
      mem[(MEM_GT + origin) % b.mem_stride] = w[W_GT];
      mem[(MEM_XT + origin) % b.mem_stride] = w[W_XT];
      w[W_GT] = 0;  //NOTE: gradient -> zero
    }
    for (int j = lastj; j > 0; j--)
      rho[j] = rho[j-1];
  }

  void direction_project(vw& all) {
    uint32_t length = 1 << all.num_bits;
    size_t stride = all.stride;
    weight* w = all.reg.weight_vector;

    for (uint32_t i = 0; i < length; i++) {
      if (w[stride*i + W_DIR] * w[stride*i + W_PGT] >= 0) {
        w[stride*i + W_DIR] = 0;
      }
    }
  }

  void backup_state(vw& all) {
    uint32_t length = 1 << all.num_bits;
    size_t stride = all.stride;
    weight* weights = all.reg.weight_vector;
    for(uint32_t i = 0; i < length; i++, weights += stride) {
      weights[W_XP] = weights[W_XT];
    }
  } 

  void select_orthant(vw& all) {
    uint32_t length = 1 << all.num_bits;
    size_t stride = all.stride;
    weight* weights = all.reg.weight_vector;
    for(uint32_t i = 0; i < length; i++, weights += stride) {
      // if zero, use pesudo gradient orthant, or use current orthant
      weights[W_OT] = (weights[W_XT] == 0.0) ? -weights[W_PGT] : weights[W_XT];
    } 
  }

  double wolfe_eval(vw& all, bfgs& b, float* mem, 
      double loss_sum, double previous_loss_sum, 
      double step_size, double importance_weight_sum, 
      int &origin, double& wolfe1) { 
    uint32_t length = 1 << all.num_bits;
    size_t stride = all.stride;
    weight* w = all.reg.weight_vector;

    double g0_d = 0.;
    double g1_d = 0.;
    
    for (uint32_t i = 0; i < length; i++, mem += b.mem_stride, w += stride) {
      g0_d += mem[(MEM_GT + origin) % b.mem_stride] * w[W_DIR];
      g1_d += w[W_GT] * w[W_DIR];
    }

    wolfe1 = (loss_sum - previous_loss_sum) / (step_size * g0_d);
    if (!all.quiet) {
      double wolfe2 = g1_d / g0_d;
      fprintf(stderr, "wolfe eval: %-10f (bound:%-10f), %-10f\n", wolfe1, b.wolfe1_bound, wolfe2);
    }

    return 0.5 * step_size;
  }

  // L1-norm
  double add_regularization(vw& all, bfgs& b, float regularization) {
    double ret = 0.;
    uint32_t length = 1 << all.num_bits;
    size_t stride = all.stride;
    weight* weights = all.reg.weight_vector;
    if (b.regularizers == NULL) {
      for(uint32_t i = 0; i < length; i++) {
        // loss
        ret += regularization * fabs(weights[stride*i]);
        // gradient
        if (weights[stride*i] < 0.) {
          weights[stride*i + W_PGT] = weights[stride*i + W_GT] - regularization;
        } else if (weights[stride*i] > 0.) {
          weights[stride*i + W_PGT] = weights[stride*i + W_GT] + regularization;
        } else { // w[i] = 0
          if (weights[stride*i + W_GT] < -regularization) {
            weights[stride*i + W_PGT] = weights[stride*i + W_GT] + regularization;
          } else if (weights[stride*i + W_GT] > regularization) {
            weights[stride*i + W_PGT] = weights[stride*i + W_GT] - regularization;
          } else {
            weights[stride*i + W_PGT] = 0.;
          }
        } // end if-else
      } // end for
      fprintf(stderr, "Use consistent regularizers, regularization = %.3f\n", ret);
    } else {
      // regularizers [2*i] is lambda coef, [2*i+1] is mean
      for(uint32_t i = 0; i < length; i++) {
        // loss
        float lambda = b.regularizers[2*i];
        weight delta_weight = weights[stride*i] - b.regularizers[2*i+1];
        ret += fabs(lambda * delta_weight);
        // gradient
        if (weights[stride*i] < 0.) {
          weights[stride*i + W_PGT] = weights[stride*i + W_GT] - lambda;
        } else if (weights[stride*i] > 0.) {
          weights[stride*i + W_PGT] = weights[stride*i + W_GT] + lambda;
        } else { // w[i] = 0
          if (weights[stride*i + W_GT] < -lambda) {
            weights[stride*i + W_PGT] = weights[stride*i + W_GT] + lambda;
          } else if (weights[stride*i + W_GT] > lambda) {
            weights[stride*i + W_PGT] = weights[stride*i + W_GT] - lambda;
          } else {
            weights[stride*i + W_PGT] = 0.;
          }
        } // end if-else
      } // end for
      fprintf(stderr, "Use self-customized regularizers, regularization = %.3f\n", ret);
    }
    return ret;
  }

  void update_weight(vw& all, float step_size, size_t current_pass) {
    uint32_t length = 1 << all.num_bits;
    size_t stride = all.stride;
    weight* w = all.reg.weight_vector;

    for(uint32_t i = 0; i < length; i++, w += stride) {
      w[W_XT] = w[W_XP] + step_size * w[W_DIR];
      if (all.l1_lambda > 0.) {
        if (w[W_XT] * w[W_OT] < 0.) {
          w[W_XT] = 0.0;
        }
      } // end if
    } // end for
  }

  // add by x
  void update_solution(vw& all, bfgs& b) {
    size_t length = all.stride * (((size_t)1) << all.num_bits);
    double avg_loss = b.loss_sum / b.importance_weight_sum;
    if (avg_loss < all.reg.best_loss) {
      all.reg.backup(length);
      all.reg.best_loss = avg_loss;
    }
  }
  // add by x
  void write_solution(vw& all, bfgs& b) {
    fprintf(stdout, "write solution, optimal(maybe not best) loss:%5.6f\n", all.reg.best_loss);
    size_t length = all.stride * (((size_t)1) << all.num_bits);
    all.reg.restore(length);
  }

  int process_pass(vw& all, bfgs& b) {
    fprintf(stderr, "\n******************current_pass = %u *********************\n", 
        (unsigned int) b.current_pass);
    int status = LEARN_OK;

    // synchronize loss and gradient
    if (all.span_server != "") {
      b.loss_sum = accumulate_scalar(all, all.span_server, b.loss_sum);
      accumulate(all, all.span_server, all.reg, W_GT);
    }
    if (all.l1_lambda > 0.) {
      // pesudo gradient is also calculated inside
      b.loss_sum += add_regularization(all, b, all.l1_lambda);
    }

    if (!all.quiet) {
      fprintf(stderr, "M: W_XT=%.5f\tW_GT=%.3f\tW_DIR=%.3f\tW_PGT=%.3f\tW_XP=%.5f\n", 
          compute_magnitude(all, W_XT), compute_magnitude(all, W_GT), compute_magnitude(all, W_DIR),
          compute_magnitude(all, W_PGT), compute_magnitude(all, W_XP));
      fprintf(stderr, "pass = %2lu avg.loss = %-10.5f\n", 
          (long unsigned int)b.current_pass + 1, b.loss_sum / b.importance_weight_sum);
    }

    /********************************************************************/
    /* A) FIRST PASS FINISHED: INITIALIZE FIRST LINE SEARCH *************/
    /********************************************************************/ 
    if (b.first_pass) {
      if(all.span_server != "") {
        float temp = (float)b.importance_weight_sum;
        b.importance_weight_sum = accumulate_scalar(all, all.span_server, temp);
      }

      b.previous_loss_sum = b.loss_sum;
      b.loss_sum = 0.;
      b.example_number = 0;
      bfgs_iter_start(all, b, b.mem, b.lastj, b.importance_weight_sum, b.origin);
        
      float d_mag = compute_magnitude(all, W_DIR);
      b.step_size = 1.0 / sqrt(d_mag);

      //TODO: make sure initial variables are not a minimizer

      if (all.l1_lambda > 0.0) {
        select_orthant(all);
      } 
      backup_state(all);
      
      if (!all.quiet) {
        fprintf(stderr, "update step_size = %.5f, d_mag = %.5f\n", b.step_size, d_mag);
      }
      update_weight(all, b.step_size, b.current_pass); 
    
    } else {
      /********************************************************************/
      /* B) GRADIENT CALCULATED *******************************************/
      /********************************************************************/ 
      // We just finished computing all gradients
      double wolfe1;
      double new_step = wolfe_eval(all, b, b.mem, b.loss_sum, b.previous_loss_sum, 
          b.step_size, b.importance_weight_sum, b.origin, wolfe1);

      // add by x : record best solution 
      update_solution(all, b); // add by x

      /********************************************************************/
      /* B0) DERIVATIVE ZERO: MINIMUM FOUND *******************************/
      /********************************************************************/ 
      if (nanpattern((float)wolfe1)) {
        fprintf(stdout, "Derivative 0 detected.\n");
        b.step_size = 0.0;
        status = LEARN_CONV;
      }
      /********************************************************************/
      /* B1) LINE SEARCH FAILED *******************************************/
      /********************************************************************/ 
      else if (wolfe1 < b.wolfe1_bound || b.loss_sum > b.previous_loss_sum) {
        // curvature violated, or we stepped too far last time: step back
        float ratio = (b.step_size == 0.f) ? 0.f : (float) (new_step / b.step_size);
        if (!all.quiet) {
          fprintf(stderr, "step too far? loss %.5f < %.5f (step revise x %.1f: %.5f)\n", 
              b.loss_sum, b.previous_loss_sum, ratio, new_step);
        }

        update_weight(all, (float)(-b.step_size + new_step), b.current_pass); 
        b.step_size = (float)new_step;
        zero_derivative(all);
        b.loss_sum = 0.;
      }
      /********************************************************************/
      /* B2) LINE SEARCH SUCCESSFUL OR DISABLED          ******************/
      /*     DETERMINE NEXT SEARCH DIRECTION             ******************/
      /********************************************************************/ 
      else {
        double rel_decrease = (b.previous_loss_sum - b.loss_sum) / b.previous_loss_sum;
        if (!nanpattern((float)rel_decrease) 
            && fabs(rel_decrease) < all.rel_threshold) {
          fprintf(stdout, "\nTermination condition reached in pass %ld: "
              "decrease in loss less than %.3f%%.\n"
              "If you want to optimize further, decrease termination threshold.\n", 
              (long int)b.current_pass + 1, all.rel_threshold * 100.0);
          status = LEARN_CONV;

        } else {
          b.previous_loss_sum = b.loss_sum;
          b.loss_sum = 0.;
          b.example_number = 0;
          b.step_size = 1.0;

          try {
            bfgs_iter_middle(all, b, b.mem, b.rho, b.alpha, b.lastj, b.origin);
          } catch (curv_exception e) {
            fprintf(stdout, "Exception in bfgs_iter_middle: %s\n", curv_message);
            b.step_size=0.0;
            status = LEARN_CURV;
          }

          if (all.l1_lambda > 0.) {
            direction_project(all);
            select_orthant(all);
          }
          backup_state(all);

          if (!all.quiet) {
            fprintf(stderr, "update step_size = %.5f, d_mag = %.5f\n", 
                b.step_size, compute_magnitude(all, W_DIR));
          }
          update_weight(all, b.step_size, b.current_pass); 

        } // end if-else [loss decrease very small]
      } // end if-else [search failed or success]

    } // end if-else [first pass]

    b.current_pass++;
    b.first_pass = false;

    if (!all.quiet) {
      ftime(&b.t_end_global);
      double previous_net_time = b.net_time;
      b.net_time = (int) (1000.0 * (b.t_end_global.time - b.t_start_global.time) 
          + (b.t_end_global.millitm - b.t_start_global.millitm)); 
      fprintf(stderr, "pass net time = %.5f, this loop use = %.5f\n", 
          b.net_time, b.net_time - previous_net_time);
    }

    if (all.save_per_pass)
      save_predictor(all, all.final_regressor_name, b.current_pass);

    return status;
  }

  void process_example(vw& all, bfgs& b, example *ec) {
    label_data* ld = (label_data*)ec->ld;

    if (b.first_pass)
      b.importance_weight_sum += ld->weight;

    //if (b.gradient_pass) {
    ec->final_prediction = predict_and_gradient(all, ec);//w[0] & w[1]
    ec->loss = all.loss->getLoss(all.sd, ec->final_prediction, ld->label) * ld->weight;
    b.loss_sum += ec->loss;
  }

  void learn(void* a, void* d, example* ec) {
    vw* all = (vw*)a;
    bfgs* b = (bfgs*)d;
    assert(ec->in_use);
    if (ec->pass != b->current_pass) {
      int status = process_pass(*all, *b);
      if (status != LEARN_OK)
        reset_state(*all, *b, true);
    }
    if (test_example(ec))
      ec->final_prediction = bfgs_predict(*all,ec);//w[0]
    else
      process_example(*all, *b, ec);
  }

  void finish(void* a, void* d) {
    bfgs* b = (bfgs*)d;
    free(b->mem);
    free(b->rho);
    free(b->alpha);
    free(b);
  }

  void save_load_regularizer(vw& all, bfgs& b, io_buf& model_file, bool read, bool text) {
    char buff[512];
    int c = 0;
    uint32_t stride = all.stride;
    uint32_t length = 2*(1 << all.num_bits);
    uint32_t i = 0;
    size_t brw = 1;
    do {
      brw = 1;
      weight* v;
      if (read) {
        c++;
        brw = bin_read_fixed(model_file, (char*)&i, sizeof(i),"");
        if (brw > 0) {
          assert (i< length);		
          v = &(b.regularizers[i]);
          if (brw > 0)
            brw += bin_read_fixed(model_file, (char*)v, sizeof(*v), "");
        }
      } else { // write binary or text
        v = &(b.regularizers[i]);
        if (*v != 0.) {
          c++;
          int text_len = sprintf(buff, "%d", i);
          brw = bin_text_write_fixed(model_file,(char *)&i, sizeof (i),
              buff, text_len, text);

          text_len = sprintf(buff, ":%f\n", *v);
          brw+= bin_text_write_fixed(model_file,(char *)v, sizeof (*v),
              buff, text_len, text);
          if (read && i%2 == 1) // This is the prior mean
            all.reg.weight_vector[(i/2*stride)] = *v;
        }
      }
      if (!read)
        i++;
    } while ((!read && i < length) || (read && brw >0));
  }

  void save_load(void* in, void* d, io_buf& model_file, bool read, bool text) {
    vw* all = (vw*)in;
    bfgs* b = (bfgs*)d;

    uint32_t length = 1 << all->num_bits;

    if (read) {
      initialize_regressor(*all);

      if (all->per_feature_regularizer_input != "") {
        b->regularizers = (weight *)calloc(2*length, sizeof(weight));
        if (b->regularizers == NULL) {
          cerr << all->program_name << ": Failed to allocate regularizers array: try decreasing -b <bits>" << endl;
          throw exception();
        }
      }

      int m = all->m;
      assert(m != 0);
      b->mem_stride = 2 * m;
      b->mem = (float*) malloc(sizeof(float) * all->length() * (b->mem_stride));
      b->rho = (double*) malloc(sizeof(double) * m);
      b->alpha = (double*) malloc(sizeof(double) * m);

      if (!all->quiet) {
        fprintf(stderr, "m = %d\nAllocated %luM for weights and mem\n", m, 
            (long unsigned int)all->length()*(sizeof(float)*(b->mem_stride)+sizeof(weight)*all->stride) >> 20);
      }

      b->net_time = 0.0;
      ftime(&b->t_start_global);

      if (!all->quiet) {
        const char * header_fmt = "%2s %-10s\t%-10s\t%-10s\t %-10s\t%-10s\t%-10s\t%-10s\t%-10s\t%-10s\n";
        fprintf(stderr, header_fmt,
            "##", "avg. loss", "der. mag.", "d. m. cond.", "wolfe1", "wolfe2", "mix fraction", "curvature", "dir. magnitude", "step size");
        cerr.precision(5);
      }

      if (b->regularizers != NULL) {
        all->l1_lambda = 1; // To make sure we are adding the regularization
      }
      b->output_regularizer =  (all->per_feature_regularizer_output != "" || all->per_feature_regularizer_text != "");
      reset_state(*all, *b, false);
    }

    bool reg_vector = b->output_regularizer || all->per_feature_regularizer_input.length() > 0;
    if (model_file.files.size() > 0) {
      char buff[512];
      uint32_t text_len = sprintf(buff, ":%d\n", reg_vector);
      bin_text_read_write_fixed(model_file,(char *)&reg_vector, sizeof (reg_vector),
          "", read,
          buff, text_len, text);

      // add by x : we replace current state by best solution
      write_solution(*all, *b); // add by x
      if (reg_vector) {
        cout << "bfgs::save_load_regularizer" << endl;
        save_load_regularizer(*all, *b, model_file, read, text);
      } else {
	cout << "GD::save_load_regressor" << endl;
        GD::save_load_regressor(*all, model_file, read, text);
      }
    }
  }

  void drive(void* in, void* d) {
    vw* all = (vw*)in;
    bfgs* b = (bfgs*)d;

    example* ec = NULL;
    size_t final_pass = all->numpasses - 1;

    while ( true ) {
      if ((ec = get_example(all->p)) != NULL) { //semiblocking operation.
        assert(ec->in_use);	  

        if (ec->pass <= final_pass) {
          if (ec->pass != b->current_pass) {
            int status = process_pass(*all, *b);
            fprintf(stdout, "%2d best_loss=%10.6f\n", b->current_pass, all->reg.best_loss); // add by x
            if (status != LEARN_OK && final_pass > b->current_pass) {
              final_pass = b->current_pass;
            }
          }
          process_example(*all, *b, ec);
        }

        return_simple_example(*all, ec);

      } else if (parser_done(all->p)) {
        process_pass(*all, *b);
        fprintf(stdout, "%2d best_loss=%10.6f\n", b->current_pass, all->reg.best_loss); // add by x
        return;
      }
      else 
        ;//busywait when we have predicted on all examples but not yet trained on all.
    }
  }

  void parse_args(vw& all, std::vector<std::string>& opts, po::variables_map& vm, po::variables_map& vm_file) {
    bfgs* b = (bfgs*)calloc(1,sizeof(bfgs));
    b->wolfe1_bound = 0.01;
    b->first_pass = true;
    b->gradient_pass = true;

    learner t = {b,drive, learn, finish, save_load};
    all.l = t;

    all.bfgs = true;
    all.stride = 8; // NOTE: for more parameter storage

    if (all.m == 0) {
      fprintf(stderr, "ERROR: memory history length m = 0.\n");
      throw exception();
    }
    if (!all.quiet) {
      cerr << "enabling BFGS based optimization ";
      cerr << "**without** curvature calculation" << endl;
    }
    if (all.numpasses < 2) {
      cout << "you must make at least 2 passes to use LBFGS" << endl;
      throw exception();
    }
  }


} // end namespace
