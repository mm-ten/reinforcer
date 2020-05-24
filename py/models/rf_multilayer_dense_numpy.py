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

import os, sys

modd_str = os.path.abspath(os.path.dirname(__file__))

import random

import numpy as np
import matplotlib.pyplot as plt

sys.path.append("%s/.." % (modd_str))
import rf_utils


# -------------------------------------------------------------
# MODEL
class Model:
    def __init__(self, p_layers_lst, p_output_size_int):
        self.layers_lst = p_layers_lst  # :[:Layer]
        self.output_size_int = p_output_size_int


# LAYER
class Layer:
    def __init__(self, p_neurons_num_int, p_input_dim_int, p_activ_fn_str):
        self.neurons_num_int = p_neurons_num_int
        self.W = np.random.uniform(size=(p_neurons_num_int, p_input_dim_int), low=-0.1, high=0.1)

        # this is very important; 
        # size specifies only one dimension 
        # so numpy creates a vector; 
        # e.g. (3, 1) is still treated as a matrix and affects MUL
        # but (3, ) is a vector!
        # so the following is wrong
        # bias = np.random.uniform(low=low, high=high, size=(number_of_neurons, 1))
        # but this is correct:
        self.bias = np.random.uniform(size=(p_neurons_num_int,), low=-0.1, high=0.1)

        # ACTIVATION_FUNS
        self.activation_fn = rf_utils.get_activation_fn(p_activ_fn_str)
        self.activation_deriv_fn = rf_utils.get_activation_deriv_fn(p_activ_fn_str)


# -------------------------------------------------------------
# MODEL_CREATE
def model__create(p_layers_specs_lst,
                  p_input_dim_int=64):
    layers_lst = []
    prev_layer_output_dim_int = p_input_dim_int

    for l in p_layers_specs_lst:
        neurons_num_int, activ_fn_str = l

        # NEW_LAYER
        input_dim_int = prev_layer_output_dim_int
        layer = Layer(neurons_num_int,
                      input_dim_int,
                      activ_fn_str)

        layers_lst.append(layer)

        # the number of outputs of each layer is equal to the 
        # number of neurons in that layer.
        prev_layer_output_dim_int = neurons_num_int

    model_output_size_int = layers_lst[-1].neurons_num_int
    model = Model(layers_lst, model_output_size_int)
    return model


# -------------------------------------------------------------
def model__fit(p_x_lst,
               p_y_true_lst,
               p_model):
    # SHUFFLE_INPUT
    # random.shuffle(p_x_lst)

    all_train__nabla_JW_lst = []  # 3D numpy arr - [training_example, network_layer, 2D_W_gradient]
    all_train__nabla_Jb_lst = []  # 3D numpy arr - [training_example, network_layer, 1D_b_gradient]

    # TRAIN - over multiple training examples. 
    #         accumulate gradient update values from each example backprop pass
    for i, x_input in enumerate(p_x_lst[:60]):
        # print("example - ", i)

        y_true = p_y_true_lst[i]

        # BACKPROP
        all_layers_data_backprop_map = model__backprop(x_input, y_true, p_model)

        # accumulate gradient values for each layer in the backprop pass.
        all_train__nabla_JW_lst.append(all_layers_data_backprop_map["nabla_JW_lst"])
        all_train__nabla_Jb_lst.append(all_layers_data_backprop_map["nabla_Jb_lst"])

        # EARLY_HALT_CONDITION - if the model achieves zero loss (no gradient updates).
        #                        in practice, unless their is 
        # # if the sum of absolute values of all elements of the JW gradient is
        # # equal to 0.0, there is no more change to the model weights and the model
        # # has reached some sort of minima. 
        # # exit training
        # last_layer_nabla_JW = all_layers_data_backprop_map["nabla_JW_lst"][-1]
        # if np.sum(np.absolute(last_layer_nabla_JW)) == 0.0:
        #     break

    train_data_map = {
        "all_train__nabla_JW_lst": all_train__nabla_JW_lst,
        "all_train__nabla_Jb_lst": all_train__nabla_Jb_lst
    }
    return train_data_map


# -------------------------------------------------------------
# FORWARD

# p_x_input - is a 1D vector - (64, )
def model__forward(p_x_input,
                   p_model,
                   p_print_shapes_bool=False):
    # IMPORTANT!! - intermediate layer results are saved to be reused
    #               when doing back-propagation calculations.
    x__layers_vals_lst = []
    z__layers_out_lst = []
    # y__layers_out_lst = []

    L_prev_output = p_x_input
    for i, l in enumerate(p_model.layers_lst):

        if p_print_shapes_bool:
            print("layer %s ---------------" % (i))
            print("W   shape - %s" % (str(l.W.shape)))
            print("L-1 shape - %s" % (str(L_prev_output.shape)))
            print("b   shape - %s" % (str(l.b.shape)))

        # input for this particular layer
        x = L_prev_output

        # z = W * x + b
        # .dot() - dot product
        #          multiplies two vectors and produces a scalar
        #          = x1*y1+x2*y2+...+xn*ym
        # 
        # W*x - https://mathinsight.org/matrix_vector_multiplication
        #       its a matrix-vector product.
        #       only for the case when the number of columns in A equals the number of rows in x.
        # W - columns number is equal to the rows number in "x".
        #     rows number is equal to the number of neurons.
        #     (neurons_num, input_dim)
        # W*x - column vector of dimension - (neurons_num, 1)
        # b   - column vector of dimension - (neurons_num, 1)
        # z   - column vector of dimension - (neurons_num, 1)
        z = np.dot(l.W, x) + l.bias

        # activation - preserves dimension of "z"
        y = l.activation_fn(z)

        if p_print_shapes_bool:
            print("z   shape - %s" % (str(z.shape)))
            print("y   shape - %s" % (str(y.shape)))

        # CACHE_VALUES
        x__layers_vals_lst.append(x)
        z__layers_out_lst.append(z)
        # y__layers_out_lst.append(y)

        L_prev_output = y

    y_final = L_prev_output
    layers_data_forward_map = {
        "x__layers_vals_lst": x__layers_vals_lst,
        "z__layers_out_lst": z__layers_out_lst,
        # "y__layers_out_lst": y__layers_out_lst

    }
    return y_final, layers_data_forward_map


# -------------------------------------------------------------
# BACKPROPAGATION
def model__backprop(p_x_input,
                    p_y_true,
                    p_model,
                    p_learning_rate_f=0.01,
                    p_debug_info_bool=False):
    # FORWARD_PASS
    y_pred, layers_data_forward_map = model__forward(p_x_input, p_model, p_print_shapes_bool=False)

    x__layers_vals_lst = layers_data_forward_map["x__layers_vals_lst"]
    z__layers_out_lst = layers_data_forward_map["z__layers_out_lst"]

    lst_delta = []

    # -------------------------------------------------------------
    def last_layer_backprop():

        layer = p_model.layers_lst[-1]

        # LOSS
        loss = rf_utils.loss(y_pred, p_y_true)

        # LOSS_DERIVATIVE
        loss_deriv = rf_utils.loss_deriv(y_pred, p_y_true)

        x = x__layers_vals_lst[-1]  # input "x" vector for this layer
        z = z__layers_out_lst[-1]  # output "z" vector (pre-activation vector) for this layer

        if p_debug_info_bool:
            print("========================= L %s", 0)
            print("loss - ", loss)
            print("loss_deriv - ", loss_deriv)
            print("z - ", z)
            print(rf_utils.softmax_deriv(z))

        # -----------------------
        # DELTA
        delta = loss_deriv * layer.activation_deriv_fn(z)
        lst_delta.append(delta)

        if p_debug_info_bool:
            print("delta - %s" % (delta))

        # -----------------------

        # GRADIENTS
        nabla_JW = np.outer(delta, x)
        nabla_Jb = delta

        if p_debug_info_bool:
            print("W shape     - ", layer.W.shape)
            print("delta shape - ", delta.shape)
            print("nabla JW    - ", nabla_JW.shape)
            print("nable Jb    - ", nabla_Jb.shape)

        # UPDATE_VARS
        layer.W = layer.W - p_learning_rate_f * nabla_JW
        layer.bias = layer.bias - p_learning_rate_f * nabla_Jb

        return nabla_JW, nabla_Jb

    # -------------------------------------------------------------
    def layer_backprop(p_layer,
                       p_index_int):

        nonlocal lst_delta

        x = x__layers_vals_lst[p_index_int]  # input "x" vector for this layer
        z = z__layers_out_lst[p_index_int]  # output "z" vector (pre-activation vector) for this layer

        # LAYER_NEXT
        layer_next = p_model.layers_lst[p_index_int + 1]
        assert isinstance(layer_next, Layer)

        # -----------------------
        # DELTA_NEXT - delta of the next layer that was just backpropagated.
        #              taking the first element of the list because we prepend delta to the
        #              list lst_delta and loop-backward.
        delta_next = lst_delta[0]

        if p_debug_info_bool:
            print("========================= L %s", p_index_int)
            print("layer next - W shape: ", layer_next.W.shape)
            print("delta_next:           ", delta_next.shape)

        # DELTA
        # * - is a component-wise multiplication (regular matrix mulm not a dot-product)
        W_next = layer_next.W
        delta = p_layer.activation_deriv_fn(z) * np.dot(np.transpose(W_next), delta_next)

        # prepend delta to the list
        lst_delta = [delta] + lst_delta

        # -----------------------

        # GRADIENTS
        nabla_JW = np.outer(delta, x)
        nabla_Jb = delta

        if p_debug_info_bool:
            print("nabla JW - ", nabla_JW.shape)
            print("nable Jb - ", nabla_Jb.shape)

        # UPDATE_VARS
        layer = p_model.layers_lst[p_index_int]
        p_layer.W = p_layer.W - p_learning_rate_f * nabla_JW
        p_layer.bias = p_layer.bias - p_learning_rate_f * nabla_Jb

        return nabla_JW, nabla_Jb

    # -------------------------------------------------------------

    # one gradient (as a 2numpy array) per layer
    nabla_JW_lst = []  # 2D numpy array
    nabla_Jb_lst = []  # 1D numpy array

    # -----------------------
    # LAST_LAYER
    nabla_JW, nabla_Jb = last_layer_backprop()

    # prepend to beginning of list
    nabla_JW_lst.insert(0, nabla_JW)
    nabla_Jb_lst.insert(0, nabla_Jb)

    # -----------------------
    # ALL_LAYERS
    # "p_model.layers_lst[:-1]" - take all layers except the last one, since we process
    #                             that individuall with last_layer_backprop().
    # reversed() - reverse order so that we have a descending list of layers (from last layer first, 
    #              and so on).
    remaining_layers_lst = reversed(list(enumerate(p_model.layers_lst[:-1])))
    for i, layer in remaining_layers_lst:
        assert isinstance(layer, Layer)

        nabla_JW, nabla_Jb = layer_backprop(layer, i)

        # prepend to beginning of list
        nabla_JW_lst.insert(0, nabla_JW)
        nabla_Jb_lst.insert(0, nabla_Jb)

    # -----------------------

    layers_data_backprop_map = {
        "nabla_JW_lst": nabla_JW_lst,
        "nabla_Jb_lst": nabla_Jb_lst
    }
    return layers_data_backprop_map



if __name__ == "__main__":
    pass
