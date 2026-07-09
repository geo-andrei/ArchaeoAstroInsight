# coding=utf-8
"""Icon test.

.. note:: This program is free software; you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation; either version 2 of the License, or
     (at your option) any later version.

"""

__author__ = 'andrei.ancuta@e-uvt.ro'
__date__ = '2025-08-22'
__copyright__ = 'Copyright 2025, Andrei Ancuta'

import os
import unittest

from qgis.PyQt.QtGui import QIcon


class ArcheDetectionResourcesTest(unittest.TestCase):
    """Test the plugin icon."""

    def test_icon_png(self):
        """The plugin icon loads from disk and is not empty."""
        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'icon.png')
        self.assertTrue(os.path.exists(path))
        icon = QIcon(path)
        self.assertFalse(icon.isNull())


if __name__ == "__main__":
    suite = unittest.makeSuite(ArcheDetectionResourcesTest)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
