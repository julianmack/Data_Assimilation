"""File to run elements of pipeline module from"""
from pipeline import DataAssimilation, config

def main():
    DA = DataAssimilation.DAPipeline()
    settings = config.ToyAEConfig()
    #settings = config.Config()
    print(settings.COMPRESSION_METHOD)
    DA.Var_DA_routine(settings)


if __name__ == "__main__":
    main()
