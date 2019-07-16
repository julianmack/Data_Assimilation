"""File to run elements of pipeline module from"""
from pipeline import DataAssimilation
from pipeline.settings import config


def main():

    settings = config.ToyAEConfig()
    DA = DataAssimilation.DAPipeline(settings)
    
    #settings = config.Config()
    print(settings.COMPRESSION_METHOD)
    DA.run()


if __name__ == "__main__":
    main()
