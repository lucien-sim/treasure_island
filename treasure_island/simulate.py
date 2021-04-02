import os
import numpy as np
from PIL import Image

from treasure_island import model
from settings import ROOT_DIR


def export_binary_image(arr: np.ndarray, color0: tuple = (255, 255, 255, 255),
                        color1: tuple = (18, 146, 175, 255), colornan: tuple = (0, 0, 0, 0),
                        fname: str = 'island_outline.png'):

    img_rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    img_rgba[arr == 0] = color0
    img_rgba[arr == 1] = color1
    img_rgba[np.isnan(arr)] = colornan
    img = Image.fromarray(img_rgba, 'RGBA')
    img.save(os.path.join(ROOT_DIR, 'figs', fname))


def simulate_current_topo(inputs: list):

    # Load island DEM from file.
    file = os.path.join(ROOT_DIR, 'data', 'dem', 'San_Francisco_TopoBathy_Elevation_2m_fbJf8ZEe6dQyyTA3IBFb.tiff')
    island = model.Island.from_file(file)

    # Calculate inundation.
    print(f"Calculating baseline land mask.")
    land_mask = ~island.calculate_sea_mask(
        sea_level=model.get_sea_level(2000, slr_intensity='medium', event_freq=None))

    for input in inputs:
        print(f"Running simulation with {input}.")
        inund = island.calculate_inundation(
            model.get_sea_level(input['year'], slr_intensity=input['slr_intensity'],
                                event_freq=input['event_freq']), land_mask)
        img_arr = (inund >= 0).astype(float)
        img_arr[np.isnan(inund)] = np.nan
        fname = f"flood_mask-current_{input['year']}_{input['slr_intensity']}_{input['event_freq']}.png"
        export_binary_image(img_arr, fname=fname)


def simulate_future_topo(inputs: list, broken_shoreline: bool = False):

    # Load island DEM from file.
    file = os.path.join(ROOT_DIR, 'data', 'dem', 'San_Francisco_TopoBathy_Elevation_2m_fbJf8ZEe6dQyyTA3IBFb.tiff')
    island = model.Island.from_file(file)

    # Add adaptations.
    print(f"Adding adaptations.")
    if not broken_shoreline:
        for z_ft, coords in model.shoreline.items():
            island.add_water_barrier(model.Berm(coords + coords[::-1], z=z_ft / 3.281, r=10.))
    else:
        for z_ft, coords in model.shoreline_broken.items():
            island.add_water_barrier(model.Berm(coords + coords[::-1], z=z_ft / 3.281, r=10.))
    island.add_water_barrier(model.ElevatedPad(model.phase1, z=3.871, r=10.))
    island.add_water_barrier(model.ElevatedPad(model.phase2a, z=3.210, r=10.))
    island.add_water_barrier(model.ElevatedPad(model.phase2b, z=3.210, r=10.))
    island.add_water_barrier(model.ElevatedPad(model.phase3a, z=3.210, r=10.))
    island.add_water_barrier(model.ElevatedPad(model.phase3b, z=3.210, r=10.))
    island.add_water_barrier(model.ElevatedPad(model.phase3c, z=3.210, r=10.))
    island.add_water_barrier(model.ElevatedPad(model.phase4, z=3.210, r=10.))

    # Calculate inundation.
    print(f"Calculating baseline land mask.")
    land_mask = ~island.calculate_sea_mask(
        sea_level=model.get_sea_level(2000, slr_intensity='medium', event_freq=None))

    for input in inputs:
        print(f"Running simulation with {input}.")
        inund = island.calculate_inundation(
            model.get_sea_level(input['year'], slr_intensity=input['slr_intensity'],
                                event_freq=input['event_freq']), land_mask)
        img_arr = (inund >= 0).astype(float)
        img_arr[np.isnan(inund)] = np.nan
        broken_str = '_broken' if broken_shoreline else ''
        fname = f"flood_mask-future_{input['year']}_{input['slr_intensity']}_{input['event_freq']}{broken_str}.png"
        export_binary_image(img_arr, fname=fname)


if __name__ == '__main__':

    inputs = [
        {
            'year': 2000,
            'slr_intensity': 'high',
            'event_freq': 100
        },
        {
            'year': 2022,
            'slr_intensity': 'high',
            'event_freq': 100
        },
        {
            'year': 2035,
            'slr_intensity': 'high',
            'event_freq': 100
        },
        {
            'year': 2050,
            'slr_intensity': 'high',
            'event_freq': 100
        },
        {
            'year': 2022,
            'slr_intensity': 'medium',
            'event_freq': 100
        },
        {
            'year': 2035,
            'slr_intensity': 'medium',
            'event_freq': 100
        },
        {
            'year': 2050,
            'slr_intensity': 'medium',
            'event_freq': 100
        },
    ]

    simulate_current_topo(inputs)
    simulate_future_topo(inputs, broken_shoreline=False)

    # file = os.path.join(ROOT_DIR, 'data', 'dem', 'San_Francisco_TopoBathy_Elevation_2m_fbJf8ZEe6dQyyTA3IBFb.tiff')
    # island = model.Island.from_file(file)
    #
    # plt.figure()
    # plt.contourf(island.x, island.y, island.elev, levels=np.arange(0.5, 4, 0.1))
    # for ft, pts in model.shoreline.items():
    #     for x, y in pts:
    #         plt.scatter([x], [y], color='k', s=5)
    #         plt.text(x, y, f"({x}, {y})", color='k')
    # # for x, y in model.phase3b:
    # #     plt.scatter([x], [y], color='k', s=5)
    # #     plt.text(x, y, f"({x}, {y})", color='k')
    # # for x, y in model.phase2a:
    # #     plt.scatter([x], [y], color='r', s=5)
    # #     plt.text(x, y, f"({x}, {y})", coflor='r')
    # # for x, y in model.phase4:
    # #     plt.scatter([x], [y], color='m', s=5)
    # #     plt.text(x, y, f"({x}, {y})", color='m')
    # plt.grid()
    # plt.show()
