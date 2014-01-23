/*
Copyright (c) by respective owners including Yahoo!, Microsoft, and
individual contributors. All rights reserved.  Released under a BSD
license as described in the file LICENSE.
 */
#ifndef FTRL_PROXIMAL_H
#define FTRL_PROXIMAL_H
#include "gd.h"

namespace FTRL {
  void parse_args(vw& all, std::vector<std::string>&opts, po::variables_map& vm, po::variables_map& vm_file);
}

#endif
