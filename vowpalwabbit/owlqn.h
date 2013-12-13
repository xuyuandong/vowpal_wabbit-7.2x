/*
Copyright (c) by respective owners including Yahoo!, Microsoft, and
individual contributors. All rights reserved.  Released under a BSD
license as described in the file LICENSE.
 */
#ifndef OWLQN_H
#define OWLQN_H
#include "gd.h"

namespace OWLQN {
  void parse_args(vw& all, std::vector<std::string>&opts, po::variables_map& vm, po::variables_map& vm_file);
}

#endif
