sys.path.append(os.getcwd()) #to import pipeline

import pipeline
from pipeline import DAPipeline
class AE_TruncSVD():
    def __init__(self, AEsettings=None):
        self.SVDsettings = pipeline.config.Config()
        self.AEsettings = pipeline.config.ToyAEConfig if AEsettings is None else AEsettings

    def compare(self):
        DA = pipeline.DAPipeline()

        
