/*
Copyright (c) by respective owners including Yahoo!, Microsoft, and
individual contributors. All rights reserved.  Released under a BSD (revised)
license as described in the file LICENSE.
 */
#include <fstream>
#include <sstream>
#include <float.h>
#ifdef _WIN32
#include <WinSock2.h>
#else
#include <netdb.h>
#endif
#include <string.h>
#include <stdio.h>
#include <assert.h>

#if defined(__SSE2__) && !defined(VW_LDA_NO_SSE)
#include <xmmintrin.h>
#endif

#include "parse_example.h"
#include "constant.h"
#include "sparse_dense.h"
#include "gd.h"
#include "cache.h"
#include "simple_label.h"
#include "accumulate.h"
#include "learner.h"

using namespace std;

namespace GD
{

void predict(vw& all, example* ex);
void sync_weights(vw& all);

template <void (*T)(vw&, float, uint32_t, float, float)>
void generic_train(vw& all, example* &ec, float update, bool sqrt_norm)
{
  if (fabs(update) == 0.)
    return;
  
  float total_weight = 0.f;
  if(all.active)
    total_weight = (float)all.sd->weighted_unlabeled_examples;
  else
    total_weight = ec->example_t;

  uint32_t offset = ec->ft_offset;
  //TODO: cout << ec->ft_offset << endl;
  float avg_norm = all.normalized_sum_norm_x / total_weight;
  if (sqrt_norm) avg_norm = sqrt(avg_norm);

  for (unsigned char* i = ec->indices.begin; i != ec->indices.end; i++) 
    for (feature* f = ec->atomics[*i].begin; f != ec->atomics[*i].end; f++)
      T(all, f->x, f->weight_index + offset, avg_norm, update);

  for (vector<string>::iterator i = all.pairs.begin(); i != all.pairs.end();i++) 
    if ((ec->atomics[(int)(*i)[0]].size() > 0) && (ec->atomics[(int)(*i)[1]].size() > 0))
      for (feature* f0 = ec->atomics[(int)(*i)[0]].begin; f0 != ec->atomics[(int)(*i)[0]].end; f0++) {
        uint32_t halfhash = quadratic_constant * (f0->weight_index + offset);
        for (feature* f1 = ec->atomics[(int)(*i)[1]].begin; f1 != ec->atomics[(int)(*i)[1]].end; f1++)
          T(all, f1->x, f1->weight_index + halfhash + offset, avg_norm, f0->x * update);
      }

  for (vector<string>::iterator i = all.triples.begin(); i != all.triples.end();i++) 
    if ((ec->atomics[(int)(*i)[0]].size() > 0) && (ec->atomics[(int)(*i)[1]].size() > 0) && (ec->atomics[(int)(*i)[2]].size() > 0))
      for (feature* f0 = ec->atomics[(int)(*i)[0]].begin; f0 != ec->atomics[(int)(*i)[0]].end; f0++)
        for (feature* f1 = ec->atomics[(int)(*i)[1]].begin; f1 != ec->atomics[(int)(*i)[1]].end; f1++) {
          uint32_t halfhash = cubic_constant2 * (cubic_constant * (f0->weight_index + offset) + f1->weight_index + offset);
          for (feature* f2 = ec->atomics[(int)(*i)[2]].begin; f2 != ec->atomics[(int)(*i)[2]].end; f2++)
            T(all, f2->x, f2->weight_index + halfhash + offset, avg_norm, f0->x * f1->x * update);
        }
}

float InvSqrt(float x){
  float xhalf = 0.5f * x;
  int i = *(int*)&x; // store floating-point bits in integer
  i = 0x5f3759d5 - (i >> 1); // initial guess for Newton's method
  x = *(float*)&i; // convert new bits into float
  x = x*(1.5f - xhalf*x*x); // One round of Newton's method
  return x;
}

inline void general_update(vw& all, float x, uint32_t fi, float avg_norm, float update)
{
  weight* w = &all.reg.weight_vector[fi & all.weight_mask];
  float t = 1.f;
  if(all.adaptive) t = powf(w[1],-all.power_t);
  if(all.normalized_updates) {
    float norm = w[all.normalized_idx] * avg_norm;
    float power_t_norm = 1.f - (all.adaptive ? all.power_t : 0.f);
    t *= powf(norm*norm,-power_t_norm);
  }
  w[0] += update * x * t;
}

inline void specialized_update(vw& all, float x, uint32_t fi, float avg_norm, float update)
{
  weight* w = &all.reg.weight_vector[fi & all.weight_mask];
  float t = 1.f;
  float inv_norm = 1.f;
  if(all.normalized_updates) inv_norm /= (w[all.normalized_idx] * avg_norm);
  if(all.adaptive) {
#if defined(__SSE2__) && !defined(VW_LDA_NO_SSE)
    __m128 eta = _mm_load_ss(&w[1]);
    eta = _mm_rsqrt_ss(eta);
    _mm_store_ss(&t, eta);
    t *= inv_norm;
#else
    t = InvSqrt(w[1]) * inv_norm;
#endif
  } else {
    t *= inv_norm*inv_norm; //if only using normalized updates but not adaptive, need to divide by feature norm squared
  }
  w[0] += update * x * t;
}

void learn(void* a, void* d, example* ec)
{
  vw* all = (vw*)a;
  assert(ec->in_use);
  if (ec->pass != all->current_pass)
  {
    //TODO: cout << ec->pass << " " << all->current_pass << endl;
    if(all->span_server != "") {
      if(all->adaptive)
        accumulate_weighted_avg(*all, all->span_server, all->reg);
      else 
        accumulate_avg(*all, all->span_server, all->reg, 0);	      
    }

    if (all->save_per_pass)
      save_predictor(*all, all->final_regressor_name, all->current_pass);
    all->eta *= all->eta_decay_rate; // default d = 1.0

    all->current_pass = ec->pass;
  }

  if (!command_example(*all, ec))
  {
    predict(*all,ec);
    if (ec->eta_round != 0.)
    {
      if(all->power_t == 0.5)
        //inline_train(*all, ec, ec->eta_round);
        generic_train<specialized_update>(*all,ec,ec->eta_round,true);
      else
        //general_train(*all, ec, ec->eta_round, all->power_t);
        generic_train<general_update>(*all,ec,ec->eta_round,false);

      if (all->sd->contraction < 1e-10)  // updating weights now to avoid numerical instability
        sync_weights(*all);

    }
  }
}

void finish(void* a, void* d)
{
  size_t* current_pass = (size_t*)d;
  free(current_pass);
}

  void sync_weights(vw& all) {
    if (all.sd->gravity == 0. && all.sd->contraction == 1.)  // to avoid unnecessary weight synchronization
      return;
    uint32_t length = 1 << all.num_bits;
    size_t stride = all.stride;
    for(uint32_t i = 0; i < length && all.reg_mode; i++)
      all.reg.weight_vector[stride*i] = trunc_weight(all.reg.weight_vector[stride*i], (float)all.sd->gravity) * (float)all.sd->contraction;
    all.sd->gravity = 0.;
    all.sd->contraction = 1.;
  }

  bool command_example(vw& all, example* ec) {
    if (ec->indices.size() > 1)
      return false;

    if (ec->tag.size() >= 4 && !strncmp((const char*) ec->tag.begin, "save", 4))
    {//save state
      string final_regressor_name = all.final_regressor_name;

      if ((ec->tag).size() >= 6 && (ec->tag)[4] == '_')
        final_regressor_name = string(ec->tag.begin+5, (ec->tag).size()-5);

      if (!all.quiet)
        cerr << "saving regressor to " << final_regressor_name << endl;
      save_predictor(all, final_regressor_name, 0);

      return true;
    }
    return false;
  }

float finalize_prediction(vw& all, float ret) 
{
  if ( nanpattern(ret))
  {
    cout << "you have a NAN!!!!!" << endl;
    return 0.;
  }
  if ( ret > all.sd->max_label )
    return (float)all.sd->max_label;
  if (ret < all.sd->min_label)
    return (float)all.sd->min_label;
  return ret;
}

struct string_value {
  float v;
  string s;
  friend bool operator<(const string_value& first, const string_value& second);
};

bool operator<(const string_value& first, const string_value& second)
{
  return fabs(first.v) > fabs(second.v);
}

#include <algorithm>

void audit_feature(vw& all, feature* f, audit_data* a, vector<string_value>& results, string prepend, size_t offset = 0)
{
  ostringstream tempstream;
  size_t index = (f->weight_index + offset) & all.weight_mask;
  weight* weights = all.reg.weight_vector;
  size_t stride = all.stride;

  tempstream << prepend;
  if (a != NULL)
    tempstream << a->space << '^' << a->feature << ':';
  else 	if ( index == ((constant*stride)&all.weight_mask))
    tempstream << "Constant:";

  tempstream << (index/stride & all.parse_mask) << ':' << f->x;
  tempstream  << ':' << trunc_weight(weights[index], (float)all.sd->gravity) * (float)all.sd->contraction;
  if(all.adaptive)
    tempstream << '@' << weights[index+1];
  string_value sv = {weights[index]*f->x, tempstream.str()};
  results.push_back(sv);
}

void audit_features(vw& all, v_array<feature>& fs, v_array<audit_data>& as, vector<string_value>& results, string prepend, size_t offset = 0)
{
  for (size_t j = 0; j< fs.size(); j++)
    if (as.begin != as.end)
      audit_feature(all, & fs[j], & as[j], results, prepend, offset);
    else
      audit_feature(all, & fs[j], NULL, results, prepend, offset);
}

void audit_quad(vw& all, feature& left_feature, audit_data* left_audit, v_array<feature> &right_features, v_array<audit_data> &audit_right, vector<string_value>& results, uint32_t offset = 0)
{
  size_t halfhash = quadratic_constant * (left_feature.weight_index + offset);

  ostringstream tempstream;
  if (audit_right.size() != 0 && left_audit)
    tempstream << left_audit->space << '^' << left_audit->feature << '^';
  string prepend = tempstream.str();

  audit_features(all, right_features, audit_right, results, prepend, halfhash + offset);
}

void audit_triple(vw& all, feature& f0, audit_data* f0_audit, feature& f1, audit_data* f1_audit, 
    v_array<feature> &right_features, v_array<audit_data> &audit_right, vector<string_value>& results, uint32_t offset = 0)
{
  size_t halfhash = cubic_constant2 * (cubic_constant * (f0.weight_index + offset) + f1.weight_index + offset);

  ostringstream tempstream;
  if (audit_right.size() > 0 && f0_audit && f1_audit)
    tempstream << f0_audit->space << '^' << f0_audit->feature << '^' 
      << f1_audit->space << '^' << f1_audit->feature << '^';
  string prepend = tempstream.str();
  audit_features(all, right_features, audit_right, results, prepend, halfhash + offset);  
}

void print_features(vw& all, example* &ec)
{
  weight* weights = all.reg.weight_vector;

  if (all.lda > 0)
  {
    size_t count = 0;
    for (unsigned char* i = ec->indices.begin; i != ec->indices.end; i++)
      count += ec->audit_features[*i].size() + ec->atomics[*i].size();
    for (unsigned char* i = ec->indices.begin; i != ec->indices.end; i++) 
      for (audit_data *f = ec->audit_features[*i].begin; f != ec->audit_features[*i].end; f++)
      {
        cout << '\t' << f->space << '^' << f->feature << ':' << (f->weight_index/all.stride & all.parse_mask) << ':' << f->x;
        for (size_t k = 0; k < all.lda; k++)
          cout << ':' << weights[(f->weight_index+k) & all.weight_mask];
      }
    cout << " total of " << count << " features." << endl;
  }
  else
  {
    vector<string_value> features;
    string empty;

    for (unsigned char* i = ec->indices.begin; i != ec->indices.end; i++) 
      audit_features(all, ec->atomics[*i], ec->audit_features[*i], features, empty, ec->ft_offset);
    for (vector<string>::iterator i = all.pairs.begin(); i != all.pairs.end();i++) 
    {
      int fst = (*i)[0];
      int snd = (*i)[1];
      for (size_t j = 0; j < ec->atomics[fst].size(); j++)
      {
        audit_data* a = NULL;
        if (ec->audit_features[fst].size() > 0)
          a = & ec->audit_features[fst][j];
        audit_quad(all, ec->atomics[fst][j], a, ec->atomics[snd], ec->audit_features[snd], features);
      }
    }

    for (vector<string>::iterator i = all.triples.begin(); i != all.triples.end();i++) 
    {
      int fst = (*i)[0];
      int snd = (*i)[1];
      int trd = (*i)[2];
      for (size_t j = 0; j < ec->atomics[fst].size(); j++)
      {
        audit_data* a1 = NULL;
        if (ec->audit_features[fst].size() > 0)
          a1 = & ec->audit_features[fst][j];
        for (size_t k = 0; k < ec->atomics[snd].size(); k++)
        {
          audit_data* a2 = NULL;
          if (ec->audit_features[snd].size() > 0)
            a2 = & ec->audit_features[snd][k];
          audit_triple(all, ec->atomics[fst][j], a1, ec->atomics[snd][k], a2, ec->atomics[trd], ec->audit_features[trd], features);
        }
      }
    }

    sort(features.begin(),features.end());

    for (vector<string_value>::iterator sv = features.begin(); sv!= features.end(); sv++)
      cout << '\t' << (*sv).s;
    cout << endl;
  }
}

void print_audit_features(vw& all, example* ec)
{
  print_result(all.stdout_fileno,ec->final_prediction,-1,ec->tag);
  fflush(stdout);
  print_features(all, ec);
}

  template <void (*T)(vw&,float,uint32_t,float,float&,float&)>
void norm_add(vw& all, feature* begin, feature* end, float g, float& norm, float& norm_x, uint32_t offset=0)
{
  for (feature* f = begin; f!= end; f++)
    T(all, f->x, f->weight_index + offset, g, norm, norm_x);
}

  template <void (*T)(vw&,float,uint32_t,float,float&,float&)>
void norm_add_quad(vw& all, feature& f0, v_array<feature> &cross_features, float g, float& norm, float& norm_x, uint32_t offset=0)
{
  uint32_t halfhash = quadratic_constant * (f0.weight_index + offset);
  float norm_new = 0.f;
  float norm_x_new = 0.f;
  norm_add<T>(all, cross_features.begin, cross_features.end, g * f0.x * f0.x, norm_new, norm_x_new, halfhash + offset);
  norm   += norm_new   * f0.x * f0.x;
  norm_x += norm_x_new * f0.x * f0.x;
}

  template <void (*T)(vw&,float,uint32_t,float,float&,float&)>
void norm_add_cubic(vw& all, feature& f0, feature& f1, v_array<feature> &cross_features, float g, float& norm, float& norm_x, uint32_t offset=0)
{
  uint32_t halfhash = cubic_constant2 * (cubic_constant * (f0.weight_index + offset) + f1.weight_index + offset);
  float norm_new = 0.f;
  float norm_x_new = 0.f;
  norm_add<T>(all, cross_features.begin, cross_features.end, g * f0.x * f0.x * f1.x * f1.x, norm_new, norm_x_new, halfhash + offset);
  norm   += norm_new   * f0.x * f0.x * f1.x * f1.x;
  norm_x += norm_x_new * f0.x * f0.x * f1.x * f1.x;
}

inline void simple_norm_compute(vw& all, float x, uint32_t fi, float g, float& norm, float& norm_x) {
  weight* w = &all.reg.weight_vector[fi & all.weight_mask];
  float x2 = x * x;
  float t = 1.f;
  float inv_norm = 1.f;
  float inv_norm2 = 1.f;
  if(all.normalized_updates) {
    inv_norm /= w[all.normalized_idx];
    inv_norm2 = inv_norm*inv_norm;
    norm_x += x2 * inv_norm2;
  }
  if(all.adaptive){
    w[1] += g * x2;
#if defined(__SSE2__) && !defined(VW_LDA_NO_SSE)
    __m128 eta = _mm_load_ss(&w[1]);
    eta = _mm_rsqrt_ss(eta);
    _mm_store_ss(&t, eta);
    t *= inv_norm;
#else
    t = InvSqrt(w[1]) * inv_norm;
#endif
  } else {
    t *= inv_norm2; //if only using normalized but not adaptive, we're dividing update by feature norm squared
  }
  norm += x2 * t;
}

inline void powert_norm_compute(vw& all, float x, uint32_t fi, float g, float& norm, float& norm_x) {
  float power_t_norm = 1.f - (all.adaptive ? all.power_t : 0.f);

  weight* w = &all.reg.weight_vector[fi & all.weight_mask];
  float x2 = x * x;
  float t = 1.f;
  if(all.adaptive){
    w[1] += g * x2;
    t = powf(w[1], -all.power_t);
  }
  if(all.normalized_updates) {
    float range2 = w[all.normalized_idx] * w[all.normalized_idx];
    t *= powf(range2, -power_t_norm);
    norm_x += x2 / range2;
  }
  norm += x2 * t;
}

  template <void (*T)(vw&,float,uint32_t,float,float&,float&)>
float compute_norm(vw& all, example* &ec)
{//We must traverse the features in _precisely_ the same order as during training.
  label_data* ld = (label_data*)ec->ld;
  float g = all.loss->getSquareGrad(ec->final_prediction, ld->label) * ld->weight;
  if (g==0) return 1.;

  float norm = 0.;
  float norm_x = 0.;
  uint32_t offset = ec->ft_offset;

  for (unsigned char* i = ec->indices.begin; i != ec->indices.end; i++)
    norm_add<T>(all, ec->atomics[*i].begin, ec->atomics[*i].end, g, norm, norm_x, offset);

  for (vector<string>::iterator i = all.pairs.begin(); i != all.pairs.end(); i++)
    if (ec->atomics[(int)(*i)[0]].size() > 0)
      for (feature* f0 = ec->atomics[(int)(*i)[0]].begin; f0 != ec->atomics[(int)(*i)[0]].end; f0++)
        norm_add_quad<T>(all, *f0, ec->atomics[(int)(*i)[1]], g, norm, norm_x, offset);

  for (vector<string>::iterator i = all.triples.begin(); i != all.triples.end();i++) 
    if ((ec->atomics[(int)(*i)[0]].size() > 0) && (ec->atomics[(int)(*i)[1]].size() > 0) && (ec->atomics[(int)(*i)[2]].size() > 0))
      for (feature* f0 = ec->atomics[(int)(*i)[0]].begin; f0 != ec->atomics[(int)(*i)[0]].end; f0++)
        for (feature* f1 = ec->atomics[(int)(*i)[1]].begin; f1 != ec->atomics[(int)(*i)[1]].end; f1++)
          norm_add_cubic<T>(all, *f0, *f1, ec->atomics[(int)(*i)[2]], g, norm, norm_x, offset);

  if(all.normalized_updates) {
    float total_weight = 0;
    if(all.active)
      total_weight = (float)all.sd->weighted_unlabeled_examples;
    else
      total_weight = ec->example_t;

    all.normalized_sum_norm_x += ld->weight * norm_x;
    float avg_sq_norm = all.normalized_sum_norm_x / total_weight;

    if(all.power_t == 0.5) {
      if(all.adaptive) norm /= sqrt(avg_sq_norm);
      else norm /= avg_sq_norm;
    } else {
      float power_t_norm = 1.f - (all.adaptive ? all.power_t : 0.f);
      norm *= powf(avg_sq_norm,-power_t_norm);
    }
  }

  return norm;
}

void local_predict(vw& all, example* ec)
{
  label_data* ld = (label_data*)ec->ld;

  all.set_minmax(all.sd, ld->label);

  ec->final_prediction = finalize_prediction(all, ec->partial_prediction * (float)all.sd->contraction);
  //TODO: log: progressive validation
  /*FILE* fevl = fopen("gd.evl", "a");
  float v = 1./(1 + exp(-ec->final_prediction));
  fprintf(fevl, "%.6f\t%d\t%d\n", v, (int)ld->label, (int)ld->weight);
  fclose(fevl);*/

  if(all.active_simulation){
    float k = ec->example_t - ld->weight;
    ec->revert_weight = all.loss->getRevertingWeight(all.sd, ec->final_prediction, all.eta/powf(k,all.power_t));
    float importance = query_decision(all, ec, k);
    if(importance > 0){
      all.sd->queries += 1;
      ld->weight *= importance;
    }
    else //do not query => do not train
      ld->label = FLT_MAX;
  }

  float t;
  if(all.active)
    t = (float)all.sd->weighted_unlabeled_examples;
  else
    t = ec->example_t;

  ec->eta_round = 0;
  if (ld->label != FLT_MAX)
  {
    ec->loss = all.loss->getLoss(all.sd, ec->final_prediction, ld->label) * ld->weight;
    if (all.training && ec->loss > 0.)
    {
      float eta_t;
      float norm;
      if(all.adaptive || all.normalized_updates) {
        if(all.power_t == 0.5)
          norm = compute_norm<simple_norm_compute>(all,ec);
        else
          norm = compute_norm<powert_norm_compute>(all,ec);
      }
      else {
        norm = ec->total_sum_feat_sq;  
      }
      eta_t = all.eta * norm * ld->weight;
      if(!all.adaptive) eta_t *= powf(t,-all.power_t);

      float update = 0.f;
      if( all.invariant_updates )
        update = all.loss->getUpdate(ec->final_prediction, ld->label, eta_t, norm);
      else
        update = all.loss->getUnsafeUpdate(ec->final_prediction, ld->label, eta_t, norm);

      ec->eta_round = (float) (update / all.sd->contraction);

      if (all.reg_mode && fabs(ec->eta_round) > 1e-8) {
        double dev1 = all.loss->first_derivative(all.sd, ec->final_prediction, ld->label);
        double eta_bar = (fabs(dev1) > 1e-8) ? (-ec->eta_round / dev1) : 0.0;
        if (fabs(dev1) > 1e-8)
          all.sd->contraction /= (1. + all.l2_lambda * eta_bar * norm);
        all.sd->gravity += eta_bar * sqrt(norm) * all.l1_lambda;
      }
    }
  }
  else if(all.active)
    ec->revert_weight = all.loss->getRevertingWeight(all.sd, ec->final_prediction, all.eta/powf(t,all.power_t));

  if (all.audit)
    print_audit_features(all, ec);
}

void predict(vw& all, example* ex)
{
  label_data* ld = (label_data*)ex->ld;
  float prediction;
  if (all.training && all.normalized_updates && ld->label != FLT_MAX && ld->weight > 0) {
    if( all.power_t == 0.5 ) {
      if (all.reg_mode % 2)
        prediction = inline_predict<vec_add_trunc_rescale>(all, ex);
      else
        prediction = inline_predict<vec_add_rescale>(all, ex);
    }
    else {
      if (all.reg_mode % 2)
        prediction = inline_predict<vec_add_trunc_rescale_general>(all, ex);
      else
        prediction = inline_predict<vec_add_rescale_general>(all, ex);
    }
  }
  else {
    if (all.reg_mode % 2)
      prediction = inline_predict<vec_add_trunc>(all, ex);
    else
      prediction = inline_predict<vec_add>(all, ex);
  }

  //TODO: cout << ex->partial_prediction << endl;
  ex->partial_prediction += prediction;

  local_predict(all, ex);
  ex->done = true;
}

void save_load_regressor(vw& all, io_buf& model_file, bool read, bool text)
{
  uint32_t length = 1 << all.num_bits;
  uint32_t stride = all.stride;
  int c = 0;
  uint32_t i = 0;
  size_t brw = 1;

  sync_weights(all); 
  do 
  {
    brw = 1;
    weight* v;
    if (read)
    {
      c++;
      brw = bin_read_fixed(model_file, (char*)&i, sizeof(i),"");
      if (brw > 0)
      {
        assert (i< length);		
        v = &(all.reg.weight_vector[stride*i]);
        brw += bin_read_fixed(model_file, (char*)v, sizeof(*v), "");
      }
    }
    else // write binary or text
    {
      v = &(all.reg.weight_vector[stride*i]);
      if (*v != 0.)
      {
        c++;
        char buff[512];
        int text_len = sprintf(buff, "%d", i);
        brw = bin_text_write_fixed(model_file,(char *)&i, sizeof (i),
            buff, text_len, text);


        text_len = sprintf(buff, ":%f\n", *v);
        brw+= bin_text_write_fixed(model_file,(char *)v, sizeof (*v),
            buff, text_len, text);
      }
    }
    if (!read)
      i++;
  }
  while ((!read && i < length) || (read && brw >0));  
}

void save_load_online_state(vw& all, io_buf& model_file, bool read, bool text)
{
  char buff[512];

  uint32_t text_len = sprintf(buff, "initial_t %f\n", all.initial_t);
  bin_text_read_write_fixed(model_file,(char*)&all.initial_t, sizeof(all.initial_t), 
      "", read, 
      buff, text_len, text);

  text_len = sprintf(buff, "norm normalizer %f\n", all.normalized_sum_norm_x);
  bin_text_read_write_fixed(model_file,(char*)&all.normalized_sum_norm_x, sizeof(all.normalized_sum_norm_x), 
      "", read, 
      buff, text_len, text);

  text_len = sprintf(buff, "t %f\n", all.sd->t);
  bin_text_read_write_fixed(model_file,(char*)&all.sd->t, sizeof(all.sd->t), 
      "", read, 
      buff, text_len, text);

  text_len = sprintf(buff, "sum_loss %f\n", all.sd->sum_loss);
  bin_text_read_write_fixed(model_file,(char*)&all.sd->sum_loss, sizeof(all.sd->sum_loss), 
      "", read, 
      buff, text_len, text);

  text_len = sprintf(buff, "weighted_examples %f\n", all.sd->weighted_examples);
  bin_text_read_write_fixed(model_file,(char*)&all.sd->weighted_examples, sizeof(all.sd->weighted_examples), 
      "", read, 
      buff, text_len, text);

  text_len = sprintf(buff, "weighted_labels %f\n", all.sd->weighted_labels);
  bin_text_read_write_fixed(model_file,(char*)&all.sd->weighted_labels, sizeof(all.sd->weighted_labels), 
      "", read, 
      buff, text_len, text);

  text_len = sprintf(buff, "weighted_unlabeled_examples %f\n", all.sd->weighted_unlabeled_examples);
  bin_text_read_write_fixed(model_file,(char*)&all.sd->weighted_unlabeled_examples, sizeof(all.sd->weighted_unlabeled_examples), 
      "", read, 
      buff, text_len, text);

  text_len = sprintf(buff, "example_number %u\n", (uint32_t)all.sd->example_number);
  bin_text_read_write_fixed(model_file,(char*)&all.sd->example_number, sizeof(all.sd->example_number), 
      "", read, 
      buff, text_len, text);

  text_len = sprintf(buff, "total_features %u\n", (uint32_t)all.sd->total_features);
  bin_text_read_write_fixed(model_file,(char*)&all.sd->total_features, sizeof(all.sd->total_features), 
      "", read, 
      buff, text_len, text);

  uint32_t length = 1 << all.num_bits;
  uint32_t stride = all.stride;
  int c = 0;
  uint32_t i = 0;
  size_t brw = 1;
  do 
  {
    brw = 1;
    weight* v;
    if (read)
    {
      c++;
      brw = bin_read_fixed(model_file, (char*)&i, sizeof(i),"");
      if (brw > 0)
      {
        assert (i< length);		
        v = &(all.reg.weight_vector[stride*i]);
        if (stride == 2) //either adaptive or normalized
          brw += bin_read_fixed(model_file, (char*)v, sizeof(*v)*2, "");
        else //adaptive and normalized
          brw += bin_read_fixed(model_file, (char*)v, sizeof(*v)*3, "");	
      }
    }
    else // write binary or text
    {
      v = &(all.reg.weight_vector[stride*i]);
      if (*v != 0.)
      {
        c++;
        char buff[512];
        int text_len = sprintf(buff, "%d", i);
        brw = bin_text_write_fixed(model_file,(char *)&i, sizeof (i),
            buff, text_len, text);

        if (stride == 2)
        {//either adaptive or normalized
          text_len = sprintf(buff, ":%f %f\n", *v, *(v+1));
          brw+= bin_text_write_fixed(model_file,(char *)v, 2*sizeof (*v),
              buff, text_len, text);
        }
        else
        {//adaptive and normalized
          text_len = sprintf(buff, ":%f %f %f\n", *v, *(v+1), *(v+2));
          brw+= bin_text_write_fixed(model_file,(char *)v, 3*sizeof (*v),
              buff, text_len, text);
        }
      }
    }
    if (!read)
      i++;
  }
  while ((!read && i < length) || (read && brw >0));  
}

void save_load(void* in, void* data, io_buf& model_file, bool read, bool text)
{
  vw* all=(vw*)in;
  if(read)
  {
    initialize_regressor(*all);
    if(all->adaptive && all->initial_t > 0)
    {
      uint32_t length = 1 << all->num_bits;
      uint32_t stride = all->stride;
      for (size_t j = 1; j < stride*length; j+=stride)
      {
        all->reg.weight_vector[j] = all->initial_t;   //for adaptive update, we interpret initial_t as previously seeing initial_t fake datapoints, all with squared gradient=1
        //NOTE: this is not invariant to the scaling of the data (i.e. when combined with normalized). Since scaling the data scales the gradient, this should ideally be 
        //feature_range*initial_t, or something like that. We could potentially fix this by just adding this base quantity times the current range to the sum of gradients 
        //stored in memory at each update, and always start sum of gradients to 0, at the price of additional additions and multiplications during the update...
      }
    }
  }
  else
  {
    sync_weights(*all); 
    if(all->span_server != "") {
      if(all->adaptive)
        accumulate_weighted_avg(*all, all->span_server, all->reg);
      else
        accumulate_avg(*all, all->span_server, all->reg, 0);
    }
  }

  if (model_file.files.size() > 0)
  {
    bool resume = all->save_resume;
    char buff[512];
    uint32_t text_len = sprintf(buff, ":%d\n", resume);
    bin_text_read_write_fixed(model_file,(char *)&resume, sizeof (resume),
        "", read,
        buff, text_len, text);
    if (resume)
      save_load_online_state(*all, model_file, read, text);
    else
      save_load_regressor(*all, model_file, read, text);
  }
}

void driver(void* in, void* data)
{
  vw* all = (vw*)in;
  example* ec = NULL;

  while ( true )
  {
    if ((ec = get_example(all->p)) != NULL)//semiblocking operation.
    {
      learn(all, data, ec);
      return_simple_example(*all, ec);
    }
    else if (parser_done(all->p))
      return;
    else 
      ;//busywait when we have predicted on all examples but not yet trained on all.
  }
}

learner get_learner()
{
  size_t* current_pass = (size_t*)calloc(1, sizeof(size_t));
  learner ret = {current_pass,driver,learn,finish,save_load};
  return ret;
}
}
