from helper import *

"""
scene_name: black_table_with_five_foods
description: A black table with five different foods randomly placed on the tabletop.
"""

import math
import numpy as np


def get_tabletop_surface(table_shape: Shape) -> tuple[P, P]:
    """
    Compute an approximate tabletop top surface center and usable size from the table bbox.
    Since no tabletop subpart is guaranteed, use the transformed object bbox as support area.
    """
    table_info = get_object_info(table_shape)
    center = table_info["center"]
    size = table_info["size"]
    top_z = table_info["max"][2]
    surface_center = np.array([center[0], center[1], top_z])
    return surface_center, size


def get_position_tag(y_value: float) -> str:
    """
    Scene convention: +y is left, -y is right.
    """
    return "left" if y_value >= 0 else "right"


@register()
def place_single_food(
    oid: str,
    unique_name: str,
    food_name: str,
    color_tag: str,
    shape_tag: str,
    usage_tag: str,
    x: float,
    y: float,
    z: float,
    yaw: float,
) -> Shape:
    pos_tag = get_position_tag(y)
    food_shape = library_call(
        "usd",
        oid=oid,
        keywords=[
            unique_name,
            food_name,
            color_tag,
            shape_tag,
            usage_tag,
            pos_tag,
            "on_table",
        ],
    )
    food_shape = transform_shape(food_shape, translation_matrix((x, y, z)))
    food_center = compute_shape_center(food_shape)
    food_shape = transform_shape(
        food_shape,
        rotation_matrix(yaw, direction=(0, 0, 1), point=food_center),
    )
    return food_shape


@register()
def foods_on_black_table(table_shape: Shape) -> Shape:
    surface_center, table_size = get_tabletop_surface(table_shape)

    # Conservative usable area to avoid edge falling and collisions.
    usable_x = min(table_size[0] * 0.32, 0.22)
    usable_y = min(table_size[1] * 0.32, 0.22)
    top_z = surface_center[2]

    # Fixed random-like layout for determinism and collision avoidance.
    placements = [
        {
            "oid": "benchmark_bread_000",
            "unique_name": "table_food_bread",
            "food_name": "bread",
            "color_tag": "brown",
            "shape_tag": "rectangular",
            "usage_tag": "food",
            "dx": 0.10,
            "dy": 0.10,
            "yaw": 0.35,
        },
        {
            "oid": "benchmark_peach_020",
            "unique_name": "table_food_peach",
            "food_name": "peach",
            "color_tag": "colorful",
            "shape_tag": "round",
            "usage_tag": "fruit",
            "dx": -0.11,
            "dy": 0.08,
            "yaw": -0.20,
        },
        {
            "oid": "benchmark_food_005",
            "unique_name": "table_food_burrito",
            "food_name": "burrito",
            "color_tag": "beige",
            "shape_tag": "cylindrical",
            "usage_tag": "wrap",
            "dx": 0.11,
            "dy": -0.08,
            "yaw": 0.95,
        },
        {
            "oid": "benchmark_food_010",
            "unique_name": "table_food_sandwich",
            "food_name": "sandwich",
            "color_tag": "colorful",
            "shape_tag": "complex_shaped",
            "usage_tag": "food",
            "dx": -0.10,
            "dy": -0.10,
            "yaw": -0.55,
        },
        {
            "oid": "benchmark_sliced_ham_000",
            "unique_name": "table_food_ham",
            "food_name": "sliced_ham",
            "color_tag": "pink",
            "shape_tag": "flat",
            "usage_tag": "meat",
            "dx": 0.00,
            "dy": 0.00,
            "yaw": 0.15,
        },
    ]

    food_shapes = []
    for item in placements:
        x = surface_center[0] + np.clip(item["dx"], -usable_x, usable_x)
        y = surface_center[1] + np.clip(item["dy"], -usable_y, usable_y)
        food_shapes.append(
            library_call(
                "place_single_food",
                oid=item["oid"],
                unique_name=item["unique_name"],
                food_name=item["food_name"],
                color_tag=item["color_tag"],
                shape_tag=item["shape_tag"],
                usage_tag=item["usage_tag"],
                x=x,
                y=y,
                z=top_z,
                yaw=item["yaw"],
            )
        )

    return concat_shapes(table_shape, *food_shapes)


@register()
def black_table_with_five_foods() -> Shape:
    table_shape = library_call(
        "usd",
        oid="benchmark_tea_table_001",
        keywords=[
            "black_table",
            "table",
            "black",
            "round",
            "furniture",
            "center",
        ],
    )
    return library_call("foods_on_black_table", table_shape=table_shape)


@register()
def root_scene() -> Shape:
    return library_call("black_table_with_five_foods")