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

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

sys.path.append("%s/../models"%(modd_str))
import rf_multilayer_dense_numpy as rf_multilayer

#-------------------------------------------------------------
def main():



    #-----------------------
    # LOAD_DATA
    digits = load_digits()
    data   = np.asarray(digits.data,   dtype='float32')
    target = np.asarray(digits.target, dtype='int32')

    X_train_origin, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.15, random_state=37)

    scaler  = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train_origin)
    X_test  = scaler.transform(X_test)

    #-----------------------



    #----------------
    # LAYERS
    layers_specs_lst = [
        (20, "sigmoid"),
    ]
    for i in range(0, 3):
        layers_specs_lst.append((40, "sigmoid"))

    layers_specs_lst.append((10, "softmax"))

    #----------------



    # MODEL_CREATE
    model = rf_multilayer.model__create(layers_specs_lst, p_input_dim_int=64)

    # INPUT
    x_input = X_test[20]
    y_true  = Y_test[20]

    #-------------------------------------------------------------
    # MODEL_FORWARD
    def test_forward(p_x, p_y_true):
        y, layers_data_map = rf_multilayer.model__forward(p_x, model, p_print_shapes_bool=False)

        print("model forward pass:")
        print("y:", y)
        print("y sum:", sum(y))

        print("predicted y (%s) - true y (%s)"%(np.argmax(y), p_y_true))

        # make sure that the sum of all elements of the "y" vector (softmax output)
        # is very close to 1.0 (its not always exactly 1.0 due to lack of numerical accuracy)
        assert abs(sum(y)-1.0)<0.1

    #-------------------------------------------------------------
    test_forward(x_input, y_true)


    # BACKPROP_SINGLE_EXAMPLE
    rf_multilayer.model__backprop(x_input, y_true, model)



    print("BACKPROP")

    # for e in range(0, 10):

    # BACKPROP_ALL_EXAMPLES


    # X_train_lst = X_train.tolist()
    # random.shuffle(X_train) # X_train_lst)



    train_data_map = rf_multilayer.model__fit(X_train,
        Y_train,
        model)
    all_train__nabla_JW_lst = train_data_map["all_train__nabla_JW_lst"]
    all_train__nabla_Jb_lst = train_data_map["all_train__nabla_Jb_lst"]



    # all_train__nabla_JW_lst = [] # 3D numpy arr - [training_image, layer, 2d_W_gradient]
    # all_train__nabla_Jb_lst = [] # 3D numpy arr - [training_image, layer, 1d_b_gradient]
    #
    # # TRAIN - over multiple training examples. 
    # #         accumulate gradient update values from each example backprop pass
    # for i, x_input in enumerate(X_train[:1000]):
    #     # print("example - ", i)
    #
    #     y_true = Y_train[i]
    #
    #     # BACKPROP
    #     all_layers_data_backprop_map = rf_multilayer.model__backprop(x_input, y_true, model)
    #
    #     # accumulate gradient values for each layer in the backprop pass.
    #     all_train__nabla_JW_lst.append(all_layers_data_backprop_map["nabla_JW_lst"])
    #     all_train__nabla_Jb_lst.append(all_layers_data_backprop_map["nabla_Jb_lst"])
    #
    #     # EARLY_HALT_CONDITION - if the model achieves zero loss (no gradient updates).
    #     #                        in practice, unless their is 
    #     # # if the sum of absolute values of all elements of the JW gradient is
    #     # # equal to 0.0, there is no more change to the model weights and the model
    #     # # has reached some sort of minima. 
    #     # # exit training
    #     # last_layer_nabla_JW = all_layers_data_backprop_map["nabla_JW_lst"][-1]
    #     # if np.sum(np.absolute(last_layer_nabla_JW)) == 0.0:
    #     #     break



    #-------------------------------------------------------------
    def plot():

        fig, axs = plt.subplots(nrows=len(model.layers_lst), ncols=1, figsize=(9, 6),
            subplot_kw={'xticks': [], 'yticks': []})






        for example__nabla_JW_lst in all_train__nabla_JW_lst[-2:]:

            # print(example__nabla_JW_lst)
            # print(len(example__nabla_JW_lst))


            # one axis per layer
            for i, ax in enumerate(axs):


                layer__nabla_JW = example__nabla_JW_lst[i]


                print("---------- layer %s - %s"%(i, str(layer__nabla_JW.shape)))
                print(layer__nabla_JW)

                ax.set_ylabel(layer__nabla_JW.shape[0], fontsize=8)
                ax.set_xlabel(layer__nabla_JW.shape[1], fontsize=8)
                ax.yaxis.set_label_coords(0, 1) # align at the top of Y axis
                ax.xaxis.set_label_coords(0, 0) # align at the left of X axis
                # ax.set_title("dim: %s"%(str(layer__nabla_JW.shape)))


                ax.imshow(layer__nabla_JW, cmap=plt.cm.gray_r,
                    interpolation='nearest')

                # ax.grid(True)
                # ax.plot()


        plt.show()


    #-------------------------------------------------------------
    def plot_examples():


        fig, axs = plt.subplots(nrows=3, ncols=6, figsize=(9, 6),
                        subplot_kw={'xticks': [], 'yticks': []})



        for i, ax in enumerate(axs.flat):

            # plt.figure(figsize=(8, 5))

            print(type(i))
            print(type(ax))

            x_input = X_train_origin[i]
            y_true  = Y_train[i]


            ax.set_title("label: %d" % (y_true))

            ax.imshow(x_input.reshape(8, 8), cmap=plt.cm.gray_r,
                    interpolation='nearest')


        plt.show()




    #-------------------------------------------------------------


    # plot_examples()
    plot()



    x_input = X_test[0]
    y_true  = Y_test[0]

    test_forward(x_input, y_true)




#-------------------------------------------------------------




if __name__ == "__main__":
    main()