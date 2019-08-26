from VarDACAE.settings.base import Config

class Config3D(Config):
    def __init__(self, loader=None):
        super(Config3D, self).__init__(loader)
        self.THREE_DIM = True

        self.get_X_fp(True) #force init X_fp
        self.set_n( (91, 85, 32))
        self.FACTOR_INCREASE = 2.43