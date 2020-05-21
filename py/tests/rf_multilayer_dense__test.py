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


import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

sys.path.append("%s/.."%(modd_str))
import rf_multilayer_dense_numpy as rf_multilayer

#-------------------------------------------------------------
def main():

    

    #-----------------------
    # LOAD_DATA
    digits = load_digits()
    data   = np.asarray(digits.data,   dtype='float32')
    target = np.asarray(digits.target, dtype='int32')

    X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.15, random_state=37)

    scaler  = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    #-----------------------



    #----------------
    # LAYERS
    layers_specs_lst = [
        (20, "sigmoid"),
    ]
    for i in range(0, 3):
        layers_specs_lst.append((20, "sigmoid"))
    layers_specs_lst.append((10, "softmax"))
        
    #----------------



    # MODEL_CREATE
    model = rf_multilayer.model__create(layers_specs_lst, p_input_dim_int=64)

    # INPUT
    x_input = X_test[0]
    y_true  = Y_train[0]

    #-------------------------------------------------------------
    # MODEL_FORWARD
    def test_forward(p_x, p_y_true):
        y, layers_data_map = rf_multilayer.model__forward(p_x, model, p_print_shapes_bool=False)

        print("model forward pass:")
        print("y:", y)
        print("y sum:", sum(y))

        print("predicted y (%s) - true y (%s)"%(np.argmax(y), p_y_true))

    #-------------------------------------------------------------
    test_forward(x_input, y_true)


    # BACKPROP_SINGLE_EXAMPLE
    rf_multilayer.model__backprop(x_input, y_true, model)



    print("BACKPROP")

    for e in range(0, 10):
        
        # BACKPROP_ALL_EXAMPLES
        for i, x_input in enumerate(X_train):

            print("example - ", i)

            y_true = Y_train[i]
            rf_multilayer.model__backprop(x_input, y_true, model)


    



    x_input = X_test[0]
    y_true  = Y_test[0]

    test_forward(x_input, y_true)




#-------------------------------------------------------------




main()