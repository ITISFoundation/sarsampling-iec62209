import numpy as np
from sar_sampling.voronoi import voronoi_areas

def _test_voronoi_random(npoints: int, seed: int) -> None:
    """Test voronoi_areas with random points and validate area properties."""
    bounding_box = (0, 0, 1, 1)
    np.random.seed(seed)
    points = np.random.rand(npoints, 2) * 0.9 + 0.05
    areas = voronoi_areas(points, bounding_box)
    for i, area in enumerate(areas):
        assert area > 0, f"area not positive: {area}"
    total = sum(areas)
    assert abs(total - 1.0) < 0.01, f"areas sum to {total} != 1.0"

def test_voronoi_random() -> None:
    """Run multiple random tests with different point counts and seeds."""
    sizes = np.random.randint(4, 100, size=10).tolist()
    seeds = np.random.randint(0, 1000000, size=10).tolist()
    for size, seed in zip(sizes, seeds):
        _test_voronoi_random(size, seed)

def main() -> None:
    """Execute all voronoi validation tests and report results."""
    print("Running voronoi validation tests...")
    test_voronoi_random()
    print("All voronoi validation tests passed.")

if __name__ == "__main__":
    main()