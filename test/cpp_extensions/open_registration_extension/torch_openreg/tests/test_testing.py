# Owner(s): ["module: PrivateUse1"]

import unittest

import torch.testing._internal.common_device_type as common_device_type
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
    PrivateUse1TestBase,
)
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.opinfo.core import DecorateInfo, OpInfo


dummy_op1 = OpInfo(
    "dummy_op1", op=lambda x: x, dtypes=common_device_type.get_all_dtypes()
)
dummy_op2 = OpInfo(
    "dummy_op2", op=lambda x: x, dtypes=common_device_type.get_all_dtypes()
)
dummy_op3 = OpInfo(
    "dummy_op3", op=lambda x: x, dtypes=common_device_type.get_all_dtypes()
)


class TestDeviceTypeOpenReg(TestCase):
    def test_normal(self, device):
        pass

    @ops([dummy_op1, dummy_op2, dummy_op3])
    def test_op(self, device, dtype, op):
        if op.name == "dummy_op2":
            self.fail("dummy_op2 should be skipped via op_skips")
        if op.name == "dummy_op3":
            # This should fail, but since we decorated it with expectedFailure, the test will pass!
            self.fail("dummy_op3 fails but is decorated with expectedFailure")


# Modify PrivateUse1TestBase which is automatically included for OpenReg
PrivateUse1TestBase.op_skips = {
    "dummy_op2": [DecorateInfo(unittest.skip("skip dummy_op2"))],
}
PrivateUse1TestBase.op_decorators = {
    "dummy_op3": [DecorateInfo(unittest.expectedFailure)],
}

instantiate_device_type_tests(TestDeviceTypeOpenReg, globals(), only_for=("openreg",))

if __name__ == "__main__":
    run_tests()
