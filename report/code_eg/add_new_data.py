from VarDACAE import GetData
class NewLoaderClass(GetData):
    def get_X(self, settings):
        """
        Arguments:
            settings: (A settings.Config class)
        returns:
            np.array of dimensions B x nx x ny x nz
        """
        # ... calculate / load or download X
        # For an example see VarDACAE.data.load.GetData.get_X
        return X

from VarDACAE.settings.models.CLIC import CLIC
class ConfigNew(CLIC):
    def __init__(self, CLIC_kwargs, opt_kwargs):
        super(CLIC, self).__init__(**CLIC_kwargs)
        self.n3d = (100, 200, 300)  # Define input domain size
                                    # This is used for ConvScheduler
        self.X_FP = "SET_IF_REQ_BY_get_X"
        # ... use opt_kwargs as desired

CLIC_kwargs =  {"model_name": "Tucodec", "block_type": "NeXt",
                "Cstd": 64, "loader": NewLoaderClass}
                #NOTE: do not initialize new loader class

settings = ConfigNew(CLIC_kwargs, opt_kwargs)
