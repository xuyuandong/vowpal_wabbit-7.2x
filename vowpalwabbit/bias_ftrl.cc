/*
 *    Copyright (c) by respective owners including Yahoo!, Microsoft, and
 *       individual contributors. All rights reserved.  Released under a BSD (revised)
 *          license as described in the file LICENSE.
 *             */
#include <fstream>
#include <math.h>
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
#include "bias_ftrl.h"
#include "cache.h"
#include "simple_label.h"
#include "accumulate.h"
#include <exception>

using namespace std;

#define W_XT 0   // current parameter w(XT)
#define W_GT 1   // current gradient  g(GT)
#define W_ZT 2   // accumulated z(t) = z(t-1) + g(t) + sigma*w(t)
#define W_G2 3   // accumulated gradient squre n(t) = n(t-1) + g(t)*g(t)

/********************************************************************/
/* mem & w definition ***********************************************/
/********************************************************************/ 
// w[0] = current weight
// w[1] = current first derivative
// w[2] = accumulated zt
// w[3] = accumulated g2

namespace BIAS_FTRL {
 
  struct ftrl {
    double alpha;
    double beta;

    float sum_part1;
    float sum_part2;

    // evaluation file pointer
    FILE* fo;
    bool progressive_validation;

    // 0-1 label loss
    float getLoss(shared_data*, float prediction, float label) {
      assert(label == 0.f || label == 1.f);
      return (label - 1) * log(1 - prediction) - label * log(prediction);
    }
  };

  void reset_state(vw& all) {
    uint32_t length = 1 << all.num_bits;
    size_t stride = all.stride;
    weight* weights = all.reg.weight_vector;
    for(uint32_t i = 0; i < length; i++, weights += stride) {
      weights[W_GT] = 0;
      weights[W_ZT] = 0;
      weights[W_G2] = 0;
    }
  }

  void update_accumulated_state(weight* w, ftrl& b) {
    double ng2 = w[W_G2] + w[W_GT]*w[W_GT];
    double sigma = (sqrt(ng2) - sqrt(w[W_G2]))/b.alpha;
    w[W_ZT] += w[W_GT] - sigma * w[W_XT];
    w[W_G2] = ng2;
  }

  float sigmoid(float a){
    float b = pow(2.71828,-1*a);
    return 1/(1+b);
  }

  float ftrl_predict(vw& all, ftrl& b, example* &ec) {
    size_t mask = all.weight_mask;
    weight* weights = all.reg.weight_vector;

    unsigned char* i;
    b.sum_part1 = b.sum_part2 = 0.0;
    
    //the first feature namespace part 
    i = ec->indices.begin;
    if (ec->atomics[*i].size() > 0) {
      feature *f = ec->atomics[*i].begin;
      for (; f != ec->atomics[*i].end; f++) {
	weight* w = &weights[f->weight_index & mask];
	b.sum_part1 += w[W_XT] * f->x;
      }
    }

    // the second feature namespace part
    i++;
    if ( i!=ec->indices.end ) {
      if (ec->atomics[*i].size() > 0){
	feature *f = ec->atomics[*i].begin;
	for (; f != ec->atomics[*i].end; f++) {
	  weight* w = &weights[f->weight_index & mask];
	  b.sum_part2 += w[W_XT] * f->x;
	}
      }
    }

    return sigmoid(b.sum_part1) * sigmoid(b.sum_part2);
  }

  float predict_and_gradient(vw& all, ftrl& b, example* &ec) {
    float fp = ftrl_predict(all, b, ec);
    ec->final_prediction = fp;

    label_data* ld = (label_data*)ec->ld;
    all.set_minmax(all.sd, ld->label);

    size_t mask = all.weight_mask;
    weight* weights = all.reg.weight_vector;

    unsigned char* i;
    
    // the first feature namespace part
    i = ec->indices.begin; 
    if (ec->atomics[*i].size() > 0) {
      feature *f = ec->atomics[*i].begin;
      for (; f != ec->atomics[*i].end; f++) {
	weight* w = &weights[f->weight_index & mask];
	w[W_GT] = ld->weight * ((fp - ld->label) / (1 - fp)) * f->x / (1 + exp(b.sum_part1)); 
	update_accumulated_state(w, b);
      }
    }

    // the second feature namespace part
    i++; 
    if (i != ec->indices.end) {
      if (ec->atomics[*i].size() > 0) {
	feature *f = ec->atomics[*i].begin;
	for (; f != ec->atomics[*i].end; f++) {
	  weight* w = &weights[f->weight_index & mask];
	  w[W_GT] = ld->weight * ((fp - ld->label) / (1 - fp)) * f->x / (1 + exp(b.sum_part2)); 
	  update_accumulated_state(w, b);
	}
      }
    }

    return fp;
  }

  void update_weight(vw& all, ftrl& b, example *ec) {
    size_t mask = all.weight_mask;
    weight* weights = all.reg.weight_vector;
    for (unsigned char* i = ec->indices.begin; i != ec->indices.end; i++) {
      feature *f = ec->atomics[*i].begin;
      for (; f != ec->atomics[*i].end; f++) {
	weight* w = &weights[f->weight_index & mask];
	float flag = sign(w[W_ZT]);
	float fabs_zt = w[W_ZT] * flag;
	if (fabs_zt <= all.l1_lambda) {
	  w[W_XT] = 0.;
	} else {
	  double step = 1/(all.l2_lambda + (b.beta + sqrt(w[W_G2]))/b.alpha);
	  w[W_XT] = step * flag * (all.l1_lambda - fabs_zt);
	}
      }
    }
  }

  void evaluate_example(vw& all, ftrl&b , example* ec) {
    label_data* ld = (label_data*)ec->ld;
    ec->loss = b.getLoss(all.sd, ec->final_prediction, ld->label) * ld->weight;
    if (b.progressive_validation) {
      float v = 1./(1 + exp(-ec->final_prediction));
      fprintf(b.fo, "%.6f\t%d\n", v, (int)(ld->label * ld->weight));
    }
  }

  void learn(void* a, void* d, example* ec) {
    vw* all = (vw*)a;
    ftrl* b = (ftrl*)d;

    assert(ec->in_use);
    // predict w*x, compute gradient, update accumulate state
    predict_and_gradient(*all, *b, ec);
    // evaluate, statistic
    evaluate_example(*all, *b, ec);
    // update weight
    update_weight(*all, *b, ec);
  }

  void save_load_online_state(vw& all, io_buf& model_file, bool read, bool text) {
    char buff[512];

    int text_len = sprintf(buff, "sum_loss %f\n", all.sd->sum_loss);
    bin_text_read_write_fixed(model_file,(char*)&all.sd->sum_loss, sizeof(all.sd->sum_loss), "", read, buff, text_len, text);

    text_len = sprintf(buff, "weighted_examples %f\n", all.sd->weighted_examples);
    bin_text_read_write_fixed(model_file,(char*)&all.sd->weighted_examples, sizeof(all.sd->weighted_examples), "", read, buff, text_len, text);

    text_len = sprintf(buff, "weighted_labels %f\n", all.sd->weighted_labels);
    bin_text_read_write_fixed(model_file,(char*)&all.sd->weighted_labels, sizeof(all.sd->weighted_labels), "", read, buff, text_len, text);

    text_len = sprintf(buff, "example_number %u\n", (uint32_t)all.sd->example_number);
    bin_text_read_write_fixed(model_file,(char*)&all.sd->example_number, sizeof(all.sd->example_number), "", read, buff, text_len, text);

    text_len = sprintf(buff, "total_features %u\n", (uint32_t)all.sd->total_features);
    bin_text_read_write_fixed(model_file,(char*)&all.sd->total_features, sizeof(all.sd->total_features),  "", read, buff, text_len, text);

    uint32_t length = 1 << all.num_bits;
    uint32_t stride = all.stride;
    uint32_t i = 0;
    size_t brw = 1;
    do 
    {
      brw = 1;
      weight* v;
      if (read) { // read binary
	brw = bin_read_fixed(model_file, (char*)&i, sizeof(i),"");
	if (brw > 0) {
	  assert (i< length);	
	  v = &(all.reg.weight_vector[stride*i]);
	  brw += bin_read_fixed(model_file, (char*)v, 4*sizeof(*v), "");  
	}
      }
      else { // write binary or text
	// save w[W_XT], w[W_ZT], w[W_G2] if any of them is not zero
	v = &(all.reg.weight_vector[stride*i]);
	if (v[W_XT] !=0. || v[W_ZT] !=0. || v[W_G2] !=0.) {
	  text_len = sprintf(buff, "%d", i);
	  brw = bin_text_write_fixed(model_file,(char *)&i, sizeof (i),
	      buff, text_len, text);

	  text_len = sprintf(buff, ":%f %f %f %f\n", *v, *(v+1), *(v+2), *(v+3));
	  brw += bin_text_write_fixed(model_file, (char *)v, 4*sizeof (*v),
	      buff, text_len, text);
	}  // end if

      } // end else

      if (!read) { i++; }
    } while ((!read && i < length) || (read && brw >0));  
  }

  void save_load(void* in, void* d, io_buf& model_file, bool read, bool text) {
    vw* all = (vw*)in;
    if (read) {
      initialize_regressor(*all);
    } 

    if (model_file.files.size() > 0) {
      bool resume = all->save_resume;
      char buff[512];
      uint32_t text_len = sprintf(buff, ":%d\n", resume);
      bin_text_read_write_fixed(model_file,(char *)&resume, sizeof (resume), "", read, buff, text_len, text);

      if (resume) {
	save_load_online_state(*all, model_file, read, text);
      } else {
	GD::save_load_regressor(*all, model_file, read, text);
      }
    }

  }

  void finish(void* a, void* d) {
    ftrl* b = (ftrl*)d;
    if (b->progressive_validation) {
      fclose(b->fo);
    }
    free(b);
  }

  void drive(void* in, void* data) {
    vw* all = (vw*)in;
    example* ec = NULL;

    while ( true ) {
      if ((ec = get_example(all->p)) != NULL) { //semiblocking operation.
	learn(all, data, ec);
	return_simple_example(*all, ec);

      } else if (parser_done(all->p)) {
	return;
      }
      else 
	;//busywait when we have predicted on all examples but not yet trained on all.
    }
  }

  void parse_args(vw& all, std::vector<std::string>& opts, po::variables_map& vm, po::variables_map& vm_file) {
    all.stride = 4; // NOTE: for more parameter storage

    ftrl* b = (ftrl*)calloc(1, sizeof(ftrl));
    b->beta = 1.0;
    b->alpha = 0.5;
    if (vm.count("ftrl_alpha")) {
      b->alpha = vm["ftrl_alpha"].as<float>(); 
    }

    b->progressive_validation = false;
    if (vm.count("progressive_validation")) {
      std::string filename = vm["progressive_validation"].as<string>();
      b->fo = fopen(filename.c_str(), "w");
      assert(b->fo != NULL);
      b->progressive_validation = true;
    }

    learner t = {b, drive, learn, finish, save_load};
    all.l = t;

    if (!all.quiet) {
      cerr << "enabling FTRL-Partitioned-Proximal based optimization" << endl;
    }
  }

}

