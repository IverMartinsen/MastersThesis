# -*- coding: utf-8 -*-
"""
Module for applying Guided Backpropagation on trained keras models.

Example of usage:
    import build_gb_model_seq as build_model
    import build_gb_model_nonseq as build_model
    
    gb_model = build_model(trained_model)

Created on Fri May 21 12:42:52 2021

@author: iverm
"""
import tensorflow as tf



@tf.custom_gradient
def guided_relu(x):
    '''
    Guided ReLU activation function.
    '''
    # nonzero gradient only if gradient and activation is positive
    def grad(dy):
        return tf.cast(dy > 0, "float32") * tf.cast(x > 0, "float32") * dy
    # use normal relu function in forward pass
    return tf.nn.relu(x), grad



class GuidedReLU(tf.keras.layers.Layer):
    '''
    Guided ReLU activation layer to be used
    if activation functions are applied as layers.
    '''
    def __init__(self):
        super().__init__()
        self.name = 'GuidedReLU'
    
    def call(self, x):
        return guided_relu(x)



def build_gb_model_seq(trained_model):
    '''
    Returns model similar to input.
    ReLU activations are replaced by guided ReLU activations.

    Parameters
    ----------
    trained_model : tf.keras.models.Model
        Sequential keras model.

    Returns
    -------
    tf.keras.models.Model
        Model similar to input.

    '''
    x = trained_model.inputs[0]

    for layer in trained_model.layers[1:]:
        # replace relu activation layers
        if isinstance(layer, tf.keras.layers.ReLU):
            x = tf.keras.layers.Activation(guided_relu)(x)
        # or replace relu activation functions
        elif hasattr(layer, 'activation') and layer.activation == 'relu':
            layer.activation = guided_relu
            x = layer(x)
        else:
            x = layer(x)
        
    return tf.keras.models.Model(inputs = trained_model.inputs, outputs = x)



def build_gb_model_nonseq(model):
    '''
    Build model similar to input model where ReLU activation layers
    are replace by Guided ReLU activation layers.

    Parameters
    ----------
    model : tf.keras.models.Model
        Non-sequential keras model.

    Returns
    -------
    tf.keras.models.Model
        Functional keras model.

    '''
    print('Warning: this function will only' +  
          'replace layers of type tf.keras.layers.ReLU.' + 
          'Other ReLU activations will go unnoticed.')
    
    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # For each layer
    for layer in model.layers:
        # For each outbound node
        for node in layer._outbound_nodes:
            # Layer output is sent to 'layer_name'
            layer_name = node.outbound_layer.name
            
            # Store 'layer_name' as key and append 'layer'
            # to list of input layers
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                    {layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(
                    layer.name)

    # Set the output tensor of the input layer
    # 'model.input' is the output of the first layer
    network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: model.input})

    # Iterate over all layers after the input
    model_outputs = []
    for layer in model.layers[1:]:

        # Determine input tensors by retrieving the output tensors
        # from every layer in list of input layers
        layer_input = [
            network_dict['new_output_tensor_of'][layer_aux] 
            for layer_aux in network_dict['input_layers_of'][layer.name]
            ]
        
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Replace layer if layer instance is ReLU
        if isinstance(layer, tf.keras.layers.ReLU):
            x = layer_input
            new_layer = GuidedReLU()
            new_layer._name = 'GuidedReLU'
            x = new_layer(x)
            print(f'Replace {layer._name} by {new_layer.name}')
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

        # Save tensor in output list if it is output in initial model
        if layer_name in model.output_names:
            model_outputs.append(x)

    return tf.keras.models.Model(inputs=model.inputs, outputs=model_outputs[-1])