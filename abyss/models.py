# Import libraries
from keras import Input, Model
from keras.layers import BatchNormalization, Dense, Activation

from termcolor import cprint

from abyss.helpers import bitify, create_blok
from keras.regularizers import l2
"""
Function creating a deep autoencoder to reduce dimensionality
"""
def create_autoencoder(n_inputs, n_bottleneck, activations, depth =4, optimizer='Adam', loss = 'mse', output_activation = 'tanh'):
    reg_lambda=0
    if depth != 0:
        # Create array of number of nodes per layer of the encoder and decoder
        neurons_per_layer_e = []

        # Bitify because multiples of 8 makes it easier to batch for your cpu
        neurons_per_layer_e.append(int(bitify(n_inputs)))
        depth = depth + 1
        for i in range(depth-1):
            neurons_per_layer_e.append(int(bitify(n_inputs*(depth-(i+1))/(depth))))
        neurons_per_layer_e = neurons_per_layer_e[1:]
        neurons_per_layer_d = neurons_per_layer_e[::-1]
        depth = depth - 1
        # Print the full model
        full_model = [n_inputs] + neurons_per_layer_e + [n_bottleneck] + neurons_per_layer_d + [n_inputs]
        full_model = ' - '.join([str(int) for int in full_model])
        cprint(f"Structure of the Autoencoder model:","blue")
        cprint(f"{full_model}","cyan")

        # Inputs of the encoder
        input_e = Input(shape=(n_inputs,), name = 'inputs encoder')

        # Create encoder
        e = create_blok(input_e, neurons_per_layer_e[0], activations)

        # Go deep
        for i in range(depth-1):
            e = create_blok(e, neurons_per_layer_e[i+1], activations)
        bottle = Dense(n_bottleneck, kernel_regularizer=l2(reg_lambda))(e)
        bottle = BatchNormalization()(bottle)
        bottle = Activation(activations)(bottle)
        encoder = Model(inputs=input_e, outputs = bottle, name = 'encoder')

        # Create decoder
        input_d = Input(shape=(n_bottleneck,))
        d = create_blok(input_d, neurons_per_layer_d[0], activations)

        # Deeper
        for i in range(depth-1):
            d = create_blok(d, neurons_per_layer_d[i+1], activations)
        output_d = Dense(n_inputs, activation= output_activation, kernel_regularizer=l2(reg_lambda))(d)
        decoder = Model(inputs = input_d, outputs = output_d, name = 'decoder')

        # Create output of the autoencoder
        output_ae = decoder(encoder(input_e))

        # Create the autoencoder
        model = Model(inputs=input_e, outputs=output_ae, name = 'autoencoder')
        model.compile(optimizer, loss, metrics = ["accuracy","mse"])

        return model
    else:
        """
        If you want just the bottleneck
        """
        # Print the full model
        full_model = [n_inputs] + [n_bottleneck] + [n_inputs]
        full_model = ' - '.join([str(int) for int in full_model])
        cprint(f"Structure of the Autoencoder model:","blue")
        cprint(f"{full_model}","cyan")

        input_e = Input(shape=(n_inputs,), name = 'inputs encoder')
        bottle = Dense(n_bottleneck, kernel_regularizer=l2(reg_lambda))(input_e)
        bottle = BatchNormalization()(bottle)
        bottle = Activation(activations)(bottle)
        encoder = Model(inputs=input_e, outputs = bottle, name = 'encoder')

        input_d = Input(shape=(n_bottleneck,))
        output_d = Dense(n_inputs, activation= output_activation, kernel_regularizer=l2(reg_lambda))(input_d)
        decoder = Model(inputs = input_d, outputs = output_d, name = 'decoder')

        # Create output of the autoencoder
        output_ae = decoder(encoder(input_e))

        # Create the autoencoder
        model = Model(inputs=input_e, outputs=output_ae, name = 'autoencoder')
        model.compile(optimizer, loss, metrics = ["accuracy","mse"])
        return model

def create_decoder(n_inputs, n_outputs, activations, depth =4, optimizer = 'Adam', loss = 'mse', output_activation = 'tanh'):
    reg_lambda=0.001

    # Create array of number of nodes per layer of the encoder and decoder
    neurons_per_layer_e = []

    # Bitify because multiples of 8 makes it easier to batch for your cpu
    neurons_per_layer_e.append(int(bitify(n_outputs)))
    depth = depth + 1
    for i in range(depth-1):
        neurons_per_layer_e.append(int(bitify(n_outputs*(depth-(i+1))/(depth))))
    neurons_per_layer_e = neurons_per_layer_e[1:]
    neurons_per_layer_d = neurons_per_layer_e[::-1]
    depth = depth - 1
    # Print the full model
    full_model = [n_inputs] + neurons_per_layer_d + [n_outputs]
    full_model = ' - '.join([str(int) for int in full_model])
    cprint(f"Structure of the decoder model:","blue")
    cprint(f"{full_model}","cyan")

    # Create decoder
    # Inputs of the encoder
    input_d = Input(shape=(n_inputs,), name = 'inputs decoder')
    d = create_blok(input_d, neurons_per_layer_d[0], activations)
    # Deeper
    for i in range(depth-1):
        d = create_blok(d, neurons_per_layer_d[i+1], activations)

    output_d = Dense(n_outputs, activation= output_activation, kernel_regularizer=l2(reg_lambda))(d)
    decoder = Model(inputs = input_d, outputs = output_d, name = 'decoder')

    # Create output of the autoencoder
    output_decoder = decoder(input_d)

    # Create the autoencoder
    model = Model(inputs=input_d, outputs=output_decoder, name = 'decoder')
    model.compile(optimizer, loss, metrics = ["accuracy","mse"])
    return model