"""Test harness shim.

tests/test_shader.py defines TestShaderSystem as a plain class but calls
unittest-style assertions (self.assertIsNotNone). The test files must not
be modified, so we mix in unittest.TestCase's assert methods here.
"""

import unittest

import tests.test_shader as test_shader


for _name in dir(unittest.TestCase):
    if _name.startswith("assert") and not hasattr(test_shader.TestShaderSystem, _name):
        setattr(
            test_shader.TestShaderSystem,
            _name,
            getattr(unittest.TestCase, _name),
        )
