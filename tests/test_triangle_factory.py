from unittest import TestCase
import numpy as np
from tests.random_triangle import generate_onepertriangle
from actuary import TriangleFactory, Triangle

class TestTriangleFactory(TestCase):
    def test_triangle_creation(self) -> None:
        np.random.seed(2)
        tri_information: np.array = generate_onepertriangle(size=24,
            tail_perc=0.01,
            ultimate_mean=100000,
            ultimate_std=5000)
        fac: TriangleFactory = TriangleFactory(tri_information)
        triangle: Triangle = fac.build_movement_triangle(3, 3)
        manual_result: float = 0
        for i in range(3):
            manual_result += tri_information[0, 23-i] + tri_information[1, 22-i] + tri_information[2, 21-i]
        self.assertAlmostEqual(triangle.values[0, 7], manual_result,
            msg=f'The result should be {manual_result}', delta=1)
    
    def test_to_json(self) -> None:
        np.random.seed(2)
        tri_information: np.array = generate_onepertriangle(size=24,
            tail_perc=0.01,
            ultimate_mean=100000,
            ultimate_std=5000)
        fac: TriangleFactory = TriangleFactory(tri_information.copy())
        to_json: dict = fac.to_json()
        self.assertListEqual(to_json['base_triangle'], tri_information.tolist(),
            msg=f'The informations doesnt match')

if __name__ == '__main__':
    tester: TestTriangleFactory = TestTriangleFactory()
    tester.test_triangle_creation()
    tester.test_to_json()