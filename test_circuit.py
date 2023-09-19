import os
import unittest
import subprocess
import sys

import numpy as np
from scipy import signal

import circuit
import circuit_sol

class TestCircuit(unittest.TestCase):

    def test_usage(self):
        print("\ntesting usage")
        result = subprocess.run(["python", "circuit.py"],
                            stdout=subprocess.PIPE)
        usage = result.stdout.decode("UTF-8").strip()

        if os.name == "nt":
            string =  ("Usage:\n"
                       "    $ python circuit.py <R> <C> <input_file> " 
                       "<output_file>")

        else:
            string =  ("Usage:\n"
                       "    $ python circuit.py <R> <C> <input_file> " 
                       "<output_file>")
            

        self.assertEqual(usage,string)
        
    def test_dvdt(self):
        print("\ntesting dvdt()")

        R = np.random.rand(1)[0]*2e6 - 10e3
        C = np.random.rand(1)[0]*1e-6 + 0.5e-3
        v = np.random.rand(1)[0]*4
        f = np.random.rand(1)[0]*2

        y_submission = circuit.dvdt(f, v, R, C)
        y_solution = circuit_sol.dvdt(f, v, R, C)

        self.assertAlmostEqual(y_submission,
                               y_solution)


    def test_euler(self):
        print("\ntesting euler()")

        t = np.linspace(0,1,1000)
        f = np.random.rand(1000)

        R = np.random.rand(1)[0]*2e6 - 10e3
        C = np.random.rand(1)[0]*1e-6 + 0.5e-3
        v_0 = np.random.rand(1)[0]

        y_submission = circuit.euler(t, f, R, C, v_0)
        y_solution = circuit_sol.euler(t, f, R, C, v_0)

        np.testing.assert_array_almost_equal(y_submission,
                                             y_solution)

        y_submission = circuit.euler(t, f, R, C)
        y_solution = circuit_sol.euler(t, f, R, C)

        np.testing.assert_array_almost_equal(y_submission,
                                             y_solution)

        

    def test_rms(self):
        print("\ntesting rms()")

        f1 = np.random.rand(1000)
        f2 = np.random.rand(1000)

        y_submission = circuit.rms_diff(f1, f2)
        y_solution = circuit_sol.rms_diff(f1, f2)

        self.assertAlmostEqual(y_submission,
                               y_solution,
                               places=6)

    def test_output(self):
        print("\ntesting output")

        result = subprocess.run(["python", "circuit.py",
                                 "99.13e3","1.07e-6",
                                 "TEK0000.csv","TEK0001.csv"],
                            stdout=subprocess.PIPE)
        out_student = result.stdout.decode("UTF-8").strip()

        line = out_student.split()
        rms_val = float(line[-2])

        # print(out_student)
        # print(rms_val, 0.053762)
        
        # self.assertAlmostEqual(0.053762, rms_val, places=4)
        
        out_sol = ("the RMS error between the measured voltage and"
                   " the simulated voltage was %.3f mVrms" %
                   (rms_val))

        self.assertEqual(out_student,out_sol)


if __name__ == '__main__':
    unittest.main()




