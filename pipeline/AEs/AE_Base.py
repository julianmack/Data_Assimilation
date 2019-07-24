import torch.nn as nn
import torch

class BaseAE(nn.Module):
    """Base AE class which all should inherit from
    The following instance variables must be instantiated in __init__:
        self.layers_encode - an nn.ModuleList of all encoding layers in the network
        self.layers_decode - an nn.ModuleList of all decoding layers in the network
        self.act_fn - the activation function to use in between layers
    If known, the following instance variables *should* be instantiated in __init__:
        self.latent_sz - a tuple containing the latent size of the system
                        (NOT including the batch number).
                        e.g. if latent.shape = (M x Cout x nx x ny x nz) then
                        latent_size = (Cout, nx, ny, nz)
    """
    def forward(self, x):
        self.__check_instance_vars()
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        x = self.__maybe_convert_to_batched(x)
        layers = self.layers_encode
        for layer in layers[:-1]:
            x = self.act_fn(layer(x))
        x = layers[-1](x) #no activation function for latent space
        x = self.__flatten_encode(x)
        x = self.__maybe_convert_to_non_batched(x)
        return x

    def decode(self, x, latent_sz=None):
        x = self.__maybe_convert_to_batched(x)
        x = self.__unflatten_decode(x, latent_sz)

        layers = self.layers_decode
        for layer in layers[:-1]:
            x = self.act_fn(layer(x))

        x = layers[-1](x) #no activation function for output
        x = self.__maybe_convert_to_non_batched(x)
        return x

    def __check_instance_vars(self):
        try:
            x = self.layers_decode
            y = self.layers_encode
        except:
            raise ValueError("Must init model with instance variables layers_decode and layers_encode")

        assert isinstance(x, nn.ModuleList), "model.layers_decode must be of type nn.ModuleList"
        assert isinstance(y, nn.ModuleList), "model.layers_encode must be of type nn.ModuleList"

    def __flatten_encode(self, x):
        """Flattens input after encoding and saves latent_sz.
        NOTE: all inputs x will be batched"""

        self.latent_sz = x.shape[1:]

        x = torch.flatten(x, start_dim=1) #start at dim = 1 since batched input

        return x

    def __unflatten_decode(self, x, latent_sz=None):
        """Unflattens decoder input before decoding.
        NOTE: If the AE has not been used for an encoding, it is necessary to pass
        the desired latent_sz.
        NOTE: all inputs x will be batched"""
        if latent_sz == None:
            if hasattr(self, "latent_sz"):
                latent_sz = self.latent_sz
            else:
                latent_sz = None
        if latent_sz == None:
            raise ValueError("No latent_sz provided to decoder and encoder not run")

        self.latent_sz = latent_sz
        size = (self.batch_sz,) + tuple(self.latent_sz)

        x = x.view(size)

        return x

    def __maybe_convert_to_batched(self, x):
        """Converts system to batched input if not batched
        (since Conv3D requires batching) and sets a flag to make clear that system
        should be converted back before output"""
        # In encoder, batched input will have dimensions 2: (M x n)
        # or 5: (M x Cin x nx x ny x nz) (for batch size M).
        # In decoder, batched input will have dimensions 2: (M x L)
        dims = len(x.shape)
        if dims in [2, 5]:
            self.batch = True
        elif dims in [1, 4]:
            self.batch = False
            x = x.unsqueeze(0)
        else:
            raise ValueError("AE does not accept input with dimensions {}".format(dims))

        self.batch_sz = x.shape[0]

        return x
    def __maybe_convert_to_non_batched(self, x):
        if not self.batch:
            x = x.squeeze(0)
        return x

    def get_list_AE_layers(self, input_size, latent_dim, hidden):
        """Helper function to get a list of the number of fc nodes or conv
        channels in an autoencoder"""
        #create a list of all dimension sizes (including input/output)
        layers = [input_size]
        #encoder:
        for size in hidden:
            layers.append(size)
        #latent representation:
        layers.append(latent_dim)
        #decoder:
        for size in hidden[::-1]: #reversed list
            layers.append(size)
        layers.append(input_size)

        return layers

    def jac_explicit(self, x):
        raise NotImplemtedError("explicit Jacobian has not been implemented for this class")

