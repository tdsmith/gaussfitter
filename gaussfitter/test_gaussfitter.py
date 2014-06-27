import unittest
import numpy as np
import gaussfitter as gf

class GaussfitCase(unittest.TestCase):
    def test_simple_fit(self):
        # fit a 2D gaussian function centered at (64, 64) with width 8
        # inpars = [height,amplitude,center_x,center_y,width_x,width_y,rota]
        inpars = [0, 1, 64, 64, 8, 8, 0]
        in_data = gf.twodgaussian(inpars, shape=(128, 128))
        fit = gf.gaussfit(in_data)
        for inval, outval in zip(inpars, fit)[:-1]:
            self.assertAlmostEqual(inval, outval)
        self.assertAlmostEqual(inpars[-1], fit[-1] % 360.0)
