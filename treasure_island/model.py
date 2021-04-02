import numpy as np
import networkx as nx
from networkx.algorithms.traversal.breadth_first_search import bfs_edges
from osgeo import gdal, osr
from shapely.geometry import Polygon, Point


def get_sea_level(year: int, slr_intensity: str = 'low', event_freq: int = None) -> np.float:

    # Get sea level return period associated with tides, surges, and tsunamis
    if event_freq is None:
        sl = 0.951
    elif event_freq == 5:
        sl = 8.2 / 3.281
    elif event_freq == 10:
        sl = 8.4 / 3.281
    elif event_freq == 50:
        sl = 8.9 / 3.281
    elif event_freq == 100:
        sl = 9.2 / 3.281
    elif event_freq == 500:
        sl = 9.7 / 3.281
    else:
        raise NotImplementedError

    # Add sea level rise.
    if year == 2000:
        sl += 0.  # Reference year.

    elif year == 2022:
        if slr_intensity == 'low':
            sl += 2 / 39.37  # 2 in
        elif slr_intensity == 'medium':
            sl += 4 / 39.37  # 4 in
        elif slr_intensity == 'high':
            sl += 7 / 39.37  # 7 in

    elif year == 2035:
        if slr_intensity == 'low':
            sl += 3 / 39.37  # 3 in
        elif slr_intensity == 'medium':
            sl += 7 / 39.37  # 7 in
        elif slr_intensity == 'high':
            sl += 13 / 39.37  # 13 in

    elif year == 2050:
        if slr_intensity == 'low':
            sl += 4.5 / 39.37  # 4.5 in
        elif slr_intensity == 'medium':
            sl += 11.0 / 39.37  # 11 in
        elif slr_intensity == 'high':
            sl += 23.8 / 39.37  # 23.8 in

    elif year == 2070:
        if slr_intensity == 'low':
            sl += 8.4 / 39.37  # 8.4 in
        elif slr_intensity == 'medium':
            sl += 18.6 / 39.37  # 18.5 in
        elif slr_intensity == 'high':
            sl += 38.5 / 39.37  # 38.5 in

    elif year == 2100:
        if slr_intensity == 'low':
            sl += 16.5 / 39.37  # 16.5 in
        elif slr_intensity == 'medium':
            sl += 36.0 / 39.37  # 36 in
        elif slr_intensity == 'high':
            sl += 66.0 / 39.37  # 66 in

    else:
        raise NotImplementedError

    return sl


class WaterBarrier:

    def __init__(self, boundary_points: list, z: float, r: float = 3.):
        self.boundary_points = boundary_points
        self.z = z
        self.r = r

        self.polygon = Polygon(boundary_points)
        self.minx, self.miny, self.maxx, self.maxy = self.polygon.bounds

    def gaussian_elev(self, d: float, d_max: float = 5) -> float:
        if abs(d) > abs(d_max):
            return 0.
        else:
            sig = self.r / 3
            return self.z * np.exp(-0.5 * (d / sig) ** 2)


class ElevatedPad(WaterBarrier):

    def calculate_elevation(self, point: tuple) -> float:
        pt = Point(point)
        if (pt.x < self.minx - 1.5 * self.r or pt.x > self.maxx + 1.5 * self.r or
                pt.y < self.miny - 1.5 * self.r or pt.y > self.maxy + 1.5 * self.r):
            elev = 0.
        elif pt.within(self.polygon):
            elev = self.z
        else:
            d = self.polygon.boundary.distance(pt)
            elev = self.gaussian_elev(d, d_max=1.5 * self.r)
        return elev


class Berm(WaterBarrier):

    def calculate_elevation(self, point: tuple) -> float:
        pt = Point(point)
        if (pt.x < self.minx - 1.5 * self.r or pt.x > self.maxx + 1.5 * self.r or
                pt.y < self.miny - 1.5 * self.r or pt.y > self.maxy + 1.5 * self.r):
            elev = 0.
        elif pt.within(self.polygon):
            d = self.polygon.exterior.distance(pt)
            elev = self.gaussian_elev(d, d_max=1.5 * self.r)
        else:
            d = self.polygon.boundary.distance(pt)
            elev = self.gaussian_elev(d, d_max=1.5 * self.r)
        return elev


class Island:

    def __init__(self, elev: np.ndarray, x: np.ndarray, y: np.ndarray, epsg: str, sea_ref_level: float, sea_ref_point: tuple):
        self.elev = elev
        self.x = x
        self.y = y
        self.epsg = epsg
        self.sea_ref_level = sea_ref_level
        self.sea_ref_point = sea_ref_point

        self.land_mask = ~self.calculate_sea_mask(sea_ref_level)
        self.elev_orig = np.copy(elev)
        self.barriers = list()

    @classmethod
    def from_file(cls, file: str, sea_ref_level: float = 0.951, sea_ref_point=(126, 86)):

        # Open file.
        raster = gdal.Open(file)

        # Get elevation, x, y
        elev = raster.GetRasterBand(1).ReadAsArray().astype('float32')
        elev[elev < -1e12] = np.nan
        x_origin, pixel_width, _, y_origin, _, pixel_height = raster.GetGeoTransform()
        x = np.asarray([x_origin + i * pixel_width for i in range(raster.RasterXSize)])
        y = np.asarray([y_origin + i * pixel_height for i in range(raster.RasterYSize)])

        # Get projection.
        proj = osr.SpatialReference(wkt=raster.GetProjection())
        epsg = proj.GetAttrValue('AUTHORITY', 1)

        return cls(elev, x, y, epsg, sea_ref_level, sea_ref_point)

    def calculate_sea_mask(self, sea_level: float) -> np.ndarray:

        # Build graph with all points below sea level.
        edges_e = [((i, j), (i, j + 1)) for i in range(0, self.elev.shape[0]) for j in range(0, self.elev.shape[1] - 1)
                   if self.elev[i, j] <= sea_level and self.elev[i, j + 1] <= sea_level]
        edges_w = [((i, j - 1), (i, j)) for i in range(0, self.elev.shape[0]) for j in range(1, self.elev.shape[1])
                   if self.elev[i, j] <= sea_level and self.elev[i, j - 1] <= sea_level]
        edges_s = [((i, j), (i + 1, j)) for i in range(0, self.elev.shape[0] - 1) for j in range(0, self.elev.shape[1])
                   if self.elev[i, j] <= sea_level and self.elev[i + 1, j] <= sea_level]
        edges_n = [((i - 1, j), (i, j)) for i in range(1, self.elev.shape[0]) for j in range(0, self.elev.shape[1])
                   if self.elev[i, j] <= sea_level and self.elev[i - 1, j] <= sea_level]
        g = nx.Graph()
        g.add_edges_from(edges_e + edges_w + edges_n + edges_s)

        # Use BFS to identify points connected to the ocean.
        edges_gen = bfs_edges(g, self.sea_ref_point)
        ocean_pts = [self.sea_ref_point] + [v for _, v in edges_gen]

        # Use that collection of points to create the water mask.
        sea_mask = np.zeros(self.elev.shape).astype(bool)
        for i, j in ocean_pts:
            sea_mask[i, j] = True

        return sea_mask

    def calculate_inundation(self, sea_level: float, ref_land_mask: np.ndarray) -> np.ndarray:

        # Get land mask for reference sea level, sea mask for elevated sea level.
        sea_mask = self.calculate_sea_mask(sea_level)

        # Inundation happens everywhere below the elevated sea level, which is on land w/ reference sea level,
        # and which is flooded with the elevated sea level.
        inundation = sea_level - self.elev
        inundation[inundation < 0] = np.nan
        inundation[~ref_land_mask] = np.nan
        inundation[~sea_mask] = np.nan

        return inundation

    def add_water_barrier(self, barrier):
        self.barriers.append(barrier)
        barrier_elev = np.zeros(self.elev.shape)
        for i, x in enumerate(self.x):
            for j, y in enumerate(self.y):
                barrier_elev[j, i] = barrier.calculate_elevation((x, y))
        self.elev = np.maximum(self.elev, barrier_elev)
        return self

    def reset_elevation(self):
        self.elev = self.elev_orig
        self.barriers = list()


def calculate_bounds_from_mask(mask: np.ndarray) -> np.ndarray:
    bounds = ((mask & ~np.roll(mask, 1, axis=0)) |
              (mask & ~np.roll(mask, -1, axis=0)) |
              (mask & ~np.roll(mask, 1, axis=1)) |
              (mask & ~np.roll(mask, -1, axis=1)))
    return bounds


shoreline_c = [
    (554660, 4186820),
    (554719, 4187013),
    (554767, 4187171),
    (554782, 4187187),
    (555002, 4187306),
    (555138, 4187382),
    (555171, 4187377),
    (555400, 4187309),
    (555553, 4187260),
    (555564, 4187245),
    (555698, 4186995),
    (555834, 4186748),
    (555972, 4186497),
    (556062, 4186324),
    (556059, 4186313),
    (555964, 4185990),
    (555942, 4185911),
    (555346, 4185583),
    (555370, 4185515),
    (555336, 4185501),
    (555200, 4185744),
    (555000, 4186108),
    (554800, 4186500),
    (554652, 4186776),
]

shoreline_broken = {
    15.6: [(554660, 4186820), (554719, 4187013)],
    15.8: [(554719, 4187013), (554767, 4187171)],
    15.1: [(554767, 4187171), (554782, 4187187), (555002, 4187306), (555138, 4187382), (555171, 4187377), (555400, 4187309)],
    14.0: [(555442, 4187297), (555553, 4187260), (555564, 4187245)],
    11.7: [(555564, 4187245), (555698, 4186995)],
    11.4: [(555698, 4186995), (555834, 4186748), (555972, 4186497), (556062, 4186324), (556059, 4186313)],
    11.2: [(556059, 4186313), (555996, 4186093)],
    10.9: [(555996, 4186093), (555964, 4185990), (555942, 4185911)],
    12.2: [(555942, 4185911), (555346, 4185583), (555370, 4185515), (555336, 4185501)],
    12.6: [(555336, 4185501), (555200, 4185744)],
    15.5: [(555200, 4185744), (555000, 4186108)],
    16.3: [(555000, 4186108), (554800, 4186500), (554652, 4186776), (554660, 4186820)],
}

shoreline = {
    15.6: [(554660, 4186820), (554719, 4187013)],
    15.8: [(554719, 4187013), (554767, 4187171)],
    15.1: [(554767, 4187171), (554782, 4187187), (555002, 4187306), (555138, 4187382), (555171, 4187377), (555400, 4187309)],
    14.0: [(555400, 4187309), (555553, 4187260), (555564, 4187245)],
    11.7: [(555564, 4187245), (555698, 4186995)],
    11.4: [(555698, 4186995), (555834, 4186748), (555972, 4186497), (556062, 4186324), (556059, 4186313)],
    11.2: [(556059, 4186313), (555996, 4186093)],
    10.9: [(555996, 4186093), (555964, 4185990), (555942, 4185911)],
    12.2: [(555942, 4185911), (555346, 4185583), (555370, 4185515), (555336, 4185501)],
    12.6: [(555336, 4185501), (555200, 4185744)],
    15.5: [(555200, 4185744), (555000, 4186108)],
    16.3: [(555000, 4186108), (554800, 4186500), (554652, 4186776), (554660, 4186820)],
}

phase1 = [
    (555000, 4186122),
    (555184, 4186296),
    (555419, 4185858),
    (555536, 4185913),
    (555503, 4185973),
    (555621, 4186033),
    (555597, 4186089),
    (555688, 4186138),
    (555581, 4186342),
    (555651, 4186416),
    (555931, 4185914),
    (555295, 4185577),
]

phase2a = [
    (554914, 4186282),
    (555086, 4186482),
    (555184, 4186292),
    (555000, 4186103),
]

phase2b = [
    (555651, 4186416),
    (555734, 4186550),
    (555954, 4186167),
    (556062, 4186318),
    (555938, 4185911),
    (555930, 4185936),
    (555984, 4186116),
    (555919, 4186128),
    (555848, 4186067),
]

phase3a = [
    (555734, 4186550),
    (555954, 4186167),
    (556062, 4186318),
    (555844, 4186716),
]

phase3b = [
    (555196, 4186510),
    (555057, 4186787),
    (555123, 4186887),
    (555189, 4186776),
    (555313, 4186848),
    (555430, 4186632),
]

phase3c = [
    (555474, 4186024),
    (555544, 4186058),
    (555489, 4186174),
    (555623, 4186244),
    (555688, 4186138),
    (555597, 4186089),
    (555621, 4186033),
    (555503, 4185973),
]

phase4 = [
    (554914, 4186282),
    (555086, 4186482),
    (555099, 4186457),
    (555196, 4186510),
    (555057, 4186787),
    (554947, 4186974),
    (554834, 4186936),
    (554729, 4186777),
    (554697, 4186821),
    (554726, 4186905),
    (554698, 4186913),
    (554655, 4186774),
    (554914, 4186282),
]
