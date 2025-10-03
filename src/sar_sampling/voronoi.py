import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box
from numpy.typing import ArrayLike

__all__ = ['voronoi_areas']

def voronoi_areas(
    points: ArrayLike, 
    bbox: tuple[float, float, float, float] = (0, 0, 1, 1)
) -> list[float]:
    """
    Calculate the areas of Voronoi cells for a set of 2D points.
    
    This function computes the Voronoi diagram for the given points and calculates
    the area of each Voronoi cell, clipped to the specified bounding box.
    Infinite Voronoi regions are handled by extending them to finite boundaries.
    
    Parameters:
        points : numpy.ndarray
            A 2D array of shape (n_points, 2) containing the x,y coordinates of points.
            Must have at least 4 points.
        bbox : tuple, optional
            Bounding box as (minx, miny, maxx, maxy) to clip the Voronoi cells.
            Default is (0, 0, 1, 1).
    
    Returns:
        list
            A list of areas corresponding to each input point. Areas are in the same
            units as the input coordinates squared.
    
    Examples:
        >>> import numpy as np
        >>> points = np.array([[0.1, 0.1], [0.9, 0.1], [0.5, 0.9], [0.2, 0.5]])
        >>> areas = voronoi_areas(points)
        >>> print(areas)
        [0.25, 0.25, 0.25, 0.25]  # Example output
    """
    points_array = np.asarray(points).copy()
    if points_array.shape[1] != 2:
        raise ValueError("points must be 2D")
    if points_array.shape[0] < 4:
        raise ValueError("at least 4 points are required")
    # rescale points to unit square
    xoffset, yoffset = bbox[0], bbox[1]
    xscale = bbox[2] - bbox[0]
    yscale = bbox[3] - bbox[1]
    points_array[:,0] = (points_array[:,0] - xoffset) / xscale
    points_array[:,1] = (points_array[:,1] - yoffset) / yscale
    vor = Voronoi(points_array)
    unit_bbox = (0, 0, 1, 1)
    regions, vertices = _voronoi_finite_polygons(vor, unit_bbox)
    bbox_poly = box(*unit_bbox)
    areas: list[float] = []
    for region in regions:
        polygon_pts = vertices[region]
        poly = Polygon(polygon_pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        clipped = poly.intersection(bbox_poly)
        if not clipped.is_empty and clipped.geom_type in ("Polygon", "MultiPolygon"):
            areas.append(clipped.area)
        else:
            areas.append(0.0)
    # scale areas back to original coordinate system
    areas = [area * xscale * yscale for area in areas]
    return areas

def _voronoi_finite_polygons(
    vor: Voronoi, 
    bbox: tuple[float, float, float, float]
) -> tuple[list[list[int]], np.ndarray]:
    """
    Convert infinite Voronoi regions to finite polygons.
    
    Parameters:
        vor : scipy.spatial.Voronoi
            Voronoi diagram object.
        bbox : tuple
            Bounding box as (minx, miny, maxx, maxy).
    
    Returns:
        tuple
            (new_regions, new_vertices) where new_regions contains finite polygon
            vertex indices and new_vertices contains the vertex coordinates.
    """
    # bbox as (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = bbox
    center = vor.points.mean(axis=0)
    radius = np.max([maxx - minx, maxy - miny]) * 10  # large, but not enormous
    new_regions: list[list[int]] = []
    new_vertices: list[list[float]] = vor.vertices.tolist()
    all_ridges: dict[int, list[tuple[int, int, int]]] = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))
    for p1, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if -1 not in region:
            new_regions.append(region)
            continue
        # infinite region
        ridges = all_ridges[p1]
        new_region: list[int] = [v for v in region if v != -1]
        for p2, v1, v2 in ridges:
            if v2 < 0 or v1 < 0:
                v = v1 if v1 >= 0 else v2
                tangent = vor.points[p2] - vor.points[p1]
                tangent /= np.linalg.norm(tangent)
                normal = np.array([-tangent[1], tangent[0]])
                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, normal)) * normal
                far_point = vor.vertices[v] + direction * radius
                new_vertices.append(far_point.tolist())
                new_region.append(len(new_vertices) - 1)
        vs = np.array([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = [new_region[i] for i in np.argsort(angles)]
        new_regions.append(new_region)
    return new_regions, np.array(new_vertices)
