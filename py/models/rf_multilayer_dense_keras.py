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
from keras.models import Sequential
from keras.layers import Dense, Dropout

class Model:
    def __init__(self, p_layers_lst, p_output_size_int):
        self.layers_lst      = p_layers_lst # :[:Layer]
        self.output_size_int = p_output_size_int

# LAYER
class Layer:
    def __init__(self, p_neurons_num_int, p_input_dim_int, p_activ_fn_str):

#-------------------------------------------------------------
# Generate dummy data
x_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size=(1000, 1))
x_test  = np.random.random((100, 20))
y_test  = np.random.randint(2, size=(100, 1))

#-------------------------------------------------------------
def model__create(p_layers_specs_lst,
    p_input_dim_int = 64
    p_dropout_f     = None):
    model = Sequential()
    

    #-----------------------
    # FIRST_LAYER
    neurons_num_int, activ_fn_str = l
    layer_keras = Dense(neurons_num_int, input_dim=p_input_dim_int, activation=activ_fn_str) # 'relu') 
    model.add(layer_keras)

    if not p_dropout_f == None:
        model.add(Dropout(p_dropout_f))

    #-----------------------

    # REMAINING_LAYERS
    for l in p_layers_specs_lst[1:]:

        neurons_num_int, activ_fn_str = l
        layer_keras = Dense(neurons_num_int, activation=activ_fn_str) # 'relu') 
        model.add(layer_keras)



    
        if not p_dropout_f == None:
            model.add(Dropout(p_dropout_f))

    #-----------------------


    model.add(Dense(64, input_dim=20, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
        optimizer = 'rmsprop',
        metrics   = ['accuracy'])

#-------------------------------------------------------------
def model__fit():
    model.fit(x_train, y_train,
            epochs=20,
            batch_size=128)


#-------------------------------------------------------------
def model__evaluate():
    score = model.evaluate(x_test, y_test, batch_size=128)