from unittest import TestCase
import numpy as np
from tests.random_triangle import generate_onepertriangle
from actuary import TriangleFactory, Triangle

class TestTriangle(TestCase):
    triangle: Triangle
    information: np.array
    periods: list

    def __init__(self):
        self.periods = (3, 3)
        np.random.seed(2)
        tri_information: np.array = generate_onepertriangle(size=24,
            tail_perc=0.01,
            ultimate_mean=100000,
            ultimate_std=5000)
        fac: TriangleFactory = TriangleFactory(tri_information.copy())
        self.triangle = fac.build_movement_triangle(*self.periods)
        self.information = tri_information
        
    def test_change_to_movement(self) -> None:
        self.triangle.change_to_cumulative()
        self.triangle.change_to_movement()
        # Calculated from the input
        manual_results: np.array = np.zeros((self.information.shape[0]-1) // self.periods[0] + 1)
        # Calculated from the triangle
        triangle_results: np.array = manual_results.copy()
        base_position: int
        for i in range(manual_results.size):
            base_position = i*self.periods[0]
            triangle_results[i] += np.sum(self.triangle.values[i, :])
            for j in range(self.periods[0]):
                manual_results[i] += np.sum(self.information[base_position+j, :])
        np.testing.assert_almost_equal(triangle_results.tolist(), manual_results.tolist(),
                err_msg=f'Change To Movement Error', decimal=3)
    
    def test_get_diagonal_movement(self) -> None:
        '''Assumption: that the change_to_movement was already tested'''
        self.triangle.change_to_movement()
        # Calculated from the input
        manual_results: np.array = np.zeros((self.information.shape[0]-1) // self.periods[0] + 1)
        # Calculated from the triangle
        triangle_results: np.array = self.triangle.get_diagonal().values
        base_position: int
        up_end_row: int
        low_end_row: int
        for i in range(manual_results.size):
            base_position = i*self.periods[0]
            for j in range(self.periods[0]):
                up_end_row = self.information.shape[1] - (base_position+j)
                low_end_row = max(self.information.shape[1] - (base_position+j) - self.periods[1], 0)
                manual_results[i] += np.sum(self.information[base_position+j, low_end_row:up_end_row])
        np.testing.assert_almost_equal(triangle_results.tolist(), manual_results.tolist(),
                err_msg=f'Get Diagonal Movement Error', decimal=3)
    
    def test_change_to_cumulative(self) -> None:
        self.triangle.change_to_cumulative()

        # self.assertListEqual(to_json['base_triangle'], tri_information.tolist(),
        #     msg=f'The informations doesnt match')


if __name__ == '__main__':
    tester: TestTriangle = TestTriangle()
    tester.test_change_to_movement()
    tester.test_get_diagonal_movement()
    