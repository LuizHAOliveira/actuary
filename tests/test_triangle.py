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
        self.triangle.change_to_movement()
        manual_results: np.array = np.zeros((self.triangle.shape[0]-1) // self.periods[0] + 1)
        triangle_results: np.array = manual_results.copy()
        base_position: int
        for i in range(manual_results.size):
            base_position = i*self.periods[0]
            triangle_results[i] = np.sum(self.triangle.values[i, :])
            for j in range(self.periods[0]):
                manual_results[i] += np.sum(self.information[base_position+j, :])
        np.testing.assert_almost_equal(triangle_results.tolist(), manual_results.tolist(),
                err_msg=f'The informations doesnt match', decimal=3)
    
    def test_get_diagonal(self) -> None:
        '''Assumption: that the change_to_movement was already tested'''
        self.triangle.change_to_movement()
        

    def test_change_to_cumulative(self) -> None:
        self.triangle.change_to_cumulative()

        self.assertListEqual(to_json['base_triangle'], tri_information.tolist(),
            msg=f'The informations doesnt match')


if __name__ == '__main__':
    tester: TestTriangle = TestTriangle()
    tester.test_change_to_movement()
    