import numpy as np
class TestminimizeJ():
    """End-to-end tests"""
    def get_settings(self):
        if hasattr(self, "attr_1"):
            print("hi")

            return self.attr_1
        else:
            print("no")
            self.attr_1 = "here"
            return self.attr_1

if __name__ == "__main__":
    A = TestminimizeJ()
    res = A.get_settings()
    print(res)
    A.get_settings()
