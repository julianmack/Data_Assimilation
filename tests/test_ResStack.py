from tools.check_init import check_init
from tools.check_train_load_DA import check_train_load_DA

from pipeline.settings.models_.resNeXt import Baseline1Block, ResNeXt, ResStack3
import pytest

class TestZoo():

    def test_resStackInitRelu(self):

        configs, kwargs = self.__get_configs_kwargs()

        for idx, conf in enumerate(configs):
            try:
                check_init(configs[idx], kwargs[idx], False, "relu")
            except Exception as e:
                print(e)
                pytest.fail("Unable to init model")

    def test_resStackInitGDN(self):

        configs, kwargs = self.__get_configs_kwargs()

        for idx, conf in enumerate(configs):
            try:
                check_init(configs[idx], kwargs[idx], False, "GDN")
            except Exception as e:
                print(e)
                pytest.fail("Unable to init model")

    # def test_resStackLoad(self):
    #     configs, kwargs = self.__get_configs_kwargs(1)
    #
    #     for idx, conf in enumerate(configs):
    #         try:
    #             check_train_load_DA(configs[idx], kwargs[idx], True, False, "lrelu")
    #         except Exception as e:
    #             print(e)
    #             pytest.fail("Unable to init model")



    def __get_configs_kwargs(self, max_number = None):

        """Helper function to load a selection of configs and kwargs"""
        resNext_k = {"layers": 0, "cardinality": 0}
        resNext3_k = {"layers": 3, "cardinality": 1, "block_type": "CBAM_vanilla",
                        "module_type": "ResNeXt3"}
        resNext3_k2 = {"layers": 3, "cardinality": 2, "block_type": "CBAM_NeXt",
                        "module_type": "RDB3"}

        CONFIGS = [ResNeXt, ResStack3, ResStack3]
        KWARGS = (resNext_k, resNext3_k, resNext3_k2)

        resNext3_k0 = {"layers": 0, "cardinality": 0}
        resNext3_k = {"layers": 3, "cardinality": 1, "block_type": "vanilla",
                        "module_type": "RDB3"}
        resNext3_k2 = {"layers": 3, "cardinality": 2, "block_type": "NeXt",
                        "module_type": "ResNeXt3"}

        CONFIGS.extend([ResStack3, ResStack3, ResStack3])
        KWARGS += (resNext3_k0, resNext3_k, resNext3_k2)

        if isinstance(CONFIGS, list):
            configs = CONFIGS
        else:
            configs = (CONFIGS, ) * len(KWARGS)
        assert len(configs) == len(KWARGS)

        if max_number:
            assert max_number <= len(configs)
            configs = configs[:-max_number]
            KWARGS = KWARGS[:-max_number]
        return configs, KWARGS