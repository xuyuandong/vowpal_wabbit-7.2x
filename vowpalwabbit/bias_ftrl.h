/*
 * Copyright (c) by respective owners including Yahoo!, Microsoft, and
 * individual contributors. All rights reserved.  Released under a BSD
 * license as described in the file LICENSE.
 *  */
#ifndef BIAS_FTRL_H
#define BIAS_FTRL_H
#include "gd.h"

namespace BIAS_FTRL {
    void parse_args(vw& all, std::vector<std::string>&opts, po::variables_map& vm, po::variables_map& vm_file);
}

#endif
