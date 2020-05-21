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

import rf_utils

#-------------------------------------------------------------
# MODEL
class Model:
    def __init__(self, p_layers_lst, p_output_size_int):
        self.layers_lst      = p_layers_lst # :[:Layer]
        self.output_size_int = p_output_size_int

# LAYER
class Layer:
    def __init__(self, p_neurons_num_int, p_input_dim_int, p_activ_fn_str):
        self.neurons_num_int = p_neurons_num_int
        self.W               = np.random.uniform(size = (p_neurons_num_int, p_input_dim_int), high=0.1, low=-0.1)
        
        # this is very important; 
        # size specifies only one dimension 
        # so numpy creates a vector; 
        # e.g. (3, 1) is still treated as a matrix and affects MUL
        # but (3, ) is a vector!
        # so the following is wrong
        # bias = np.random.uniform(low=low, high=high, size=(number_of_neurons, 1))
        # but this is correct:
        self.bias = np.random.uniform(size = (p_neurons_num_int, ), high=0.1, low=-0.1)
        
        # ACTIVATION_FUNS
        self.activation_fn       = rf_utils.get_activation_fn(p_activ_fn_str)
        self.activation_deriv_fn = rf_utils.get_activation_deriv_fn(p_activ_fn_str)
        
#-------------------------------------------------------------
def model__loss(p_y_pred, p_y_true):
    loss = np.sum((p_y_pred - p_y_true) ** 2)
    J    = loss
    return J

#-------------------------------------------------------------
def model__loss_deriv(p_y_pred, p_y_true):
    J_deriv = 2 * (p_y_pred - p_y_true)
    return J_deriv

#-------------------------------------------------------------
# MODEL_CREATE
def model__create(p_layers_specs_lst,
    p_input_dim_int = 64):
    
    layers_lst                = []
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

#-------------------------------------------------------------
# FORWARD

# p_x_input - is a 1D vector - (64, )
def model__forward(p_x_input,
    p_model,
    p_print_shapes_bool=False):
    
    # IMPORTANT!! - intermediate layer results are saved to be reused
    #               when doing back-propagation calculations.
    x__layers_in_lst  = []
    z__layers_out_lst = []
    y__layers_out_lst = []
    
    L_prev_output = p_x_input
    for i, l in enumerate(p_model.layers_lst):
        
        if p_print_shapes_bool:
            print("layer %s ---------------"%(i))
            print("W   shape - %s"%(str(l.W.shape)))
            print("L-1 shape - %s"%(str(L_prev_output.shape)))
            print("b   shape - %s"%(str(l.b.shape)))
        
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
            print("z   shape - %s"%(str(z.shape)))
            print("y   shape - %s"%(str(y.shape)))
            
        # CACHE_VALUES
        x__layers_in_lst.append(x)
        z__layers_out_lst.append(z)
        y__layers_out_lst.append(y)
        
        L_prev_output = y
        
    y_final = L_prev_output
    layers_data_map = {
        "x__layers_in_lst":  x__layers_in_lst,
        "z__layers_out_lst": z__layers_out_lst,
        "y__layers_out_lst": y__layers_out_lst

    }
    return y_final, layers_data_map

#-------------------------------------------------------------
# BACKPROPAGATION
def model__backprop(p_x_input,
    p_y_true,
    p_model,
    p_learning_rate_f=0.01,
    p_debug_info_bool=False):
    
    # FORWARD_PASS
    y_pred, layers_data_map = model__forward(p_x_input, p_model, p_print_shapes_bool=False)
    
    x__layers_in_lst  = layers_data_map["x__layers_in_lst"]
    z__layers_out_lst = layers_data_map["z__layers_out_lst"]
    
    lst_delta = []
    
    #-------------------------------------------------------------
    def last_layer_backprop():
        
        layer = p_model.layers_lst[-1]



        # LOSS
        loss = model__loss(y_pred, p_y_true)
        J    = loss

        # LOSS_DERIVATIVE
        J_deriv = model__loss_deriv(y_pred, p_y_true)
        
        
        
        x = x__layers_in_lst[-1]  # input "x" vector for this layer
        z = z__layers_out_lst[-1] # output "z" vector (pre-activation vector) for this layer
        
        
        
        if p_debug_info_bool:
            print("========================= L %s", 0)
            print("J - ", J)
            print("J_deriv - ", J_deriv)
            print("z - ", z)
            print(rf_utils.softmax_deriv(z))

        #-----------------------
        # DELTA
        delta = J_deriv * layer.activation_deriv_fn(z)
        lst_delta.append(delta)
        
        if p_debug_info_bool:
            print("delta - %s"%(delta))

        #-----------------------
    
        # GRADIENTS
        nabla_JW = np.outer(delta, x)
        nabla_Jb = delta

        if p_debug_info_bool:
            print("W shape     - ", layer.W.shape)
            print("delta shape - ", delta.shape)
            print("nabla JW    - ", nabla_JW.shape)
            print("nable Jb    - ", nabla_Jb.shape)

        # UPDATE_VARS
        layer.W    = layer.W    - p_learning_rate_f * nabla_JW
        layer.bias = layer.bias - p_learning_rate_f * nabla_Jb
        

    #-------------------------------------------------------------
    def layer_backprop(p_layer,
        p_index_int):
        
        nonlocal lst_delta

        x = x__layers_in_lst[p_index_int]  # input "x" vector for this layer
        z = z__layers_out_lst[p_index_int] # output "z" vector (pre-activation vector) for this layer

        # LAYER_NEXT
        layer_next = p_model.layers_lst[p_index_int+1]
        assert isinstance(layer_next, Layer)
        
        #-----------------------
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
        delta  = p_layer.activation_deriv_fn(z) * np.dot(np.transpose(W_next), delta_next)

        # prepend delta to the list
        lst_delta = [delta] + lst_delta

        #-----------------------

        # GRADIENTS
        nabla_JW = np.outer(delta, x)
        nabla_Jb = delta
        
        if p_debug_info_bool:
            print("nabla JW - ", nabla_JW.shape)
            print("nable Jb - ", nabla_Jb.shape)

        # UPDATE_VARS
        layer        = p_model.layers_lst[p_index_int]
        p_layer.W    = p_layer.W    - p_learning_rate_f * nabla_JW
        p_layer.bias = p_layer.bias - p_learning_rate_f * nabla_Jb
        
    #-------------------------------------------------------------
    
    # LAST_LAYER
    last_layer_backprop()
    
    # ALL_LAYERS
    remaining_layers_lst = reversed(list(enumerate(p_model.layers_lst[:-1])))
    for i, l in remaining_layers_lst:
        

        layer_backprop(l, i)