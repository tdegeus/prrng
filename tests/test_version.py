import unittest

import prrng


class Test_pcg32_basic(unittest.TestCase):
    def test_version_dependencies(self):

        deps = prrng.version_dependencies()
        deps = [i.split("=")[0] for i in deps]
        self.assertIn("boost", deps)
        self.assertIn("xtensor-python", deps)
        self.assertIn("xtensor", deps)
        self.assertIn("xtl", deps)

    def test_version_compiler(self):

        deps = prrng.version_compiler()
        deps = [i.split("=")[0] for i in deps]
        self.assertIn("platform", deps)


if __name__ == "__main__":

    unittest.main()
