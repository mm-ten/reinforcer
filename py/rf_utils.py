# Reinforcer - machine-learning experimentation and utility functions
# Copyright (C) 2020 - Authors
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

import numpy as np

#-------------------------------------------------------------
# ONE_HOT_ENCODING
def one_hot(n_classes: int, y: int):
    # [y] - picks the row of the identity matrix, since thats
    #       the row that contains the diagonal value (one-hot-encoding)
    #       that we want.
    return np.eye(n_classes)[y]

#-------------------------------------------------------------
# ACTIVATION_FN
#-------------------------------------------------------------
def get_activation_fn(p_activ_fn_str):
    activation_funs_map = {
        "sigmoid": lambda p_z: sigmoid(p_z),
        "softmax": lambda p_z: softmax(p_z)
    }
    return activation_funs_map[p_activ_fn_str]
    
#-------------------------------------------------------------
def get_activation_deriv_fn(p_activ_fn_str):
    activation_deriv_funs_map = {
        "sigmoid": lambda p_z: sigmoid_deriv(p_z),
        "softmax": lambda p_z: softmax_deriv(p_z)
    }
    return activation_deriv_funs_map[p_activ_fn_str]

#-------------------------------------------------------------
# SIGMOID
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# SIGMOID_DERIVATIVE
def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x)) # MIGHT BE WRONG DERIVATIVE
            
#-------------------------------------------------------------
# SOFTMAX
def softmax(x):
    assert isinstance(x, np.ndarray)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=-1, keepdims=True)

# SOFTMAX_DERIVATIVE
def softmax_deriv(x):
    return softmax(x) * (1 - softmax(x))