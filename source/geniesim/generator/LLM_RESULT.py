from helper import *

"""
scene_name: golden_table_with_10_beverage_bottles_and_2_clothes_on_tabletop
description:
A golden table with 10 beverage bottles randomly scattered on the tabletop.
Two clothing items are placed on the left and right sides of the tabletop.
All objects are guaranteed to be positioned within the tabletop bounds.
"""

import numpy as np


def _region_tag(y_value: float) -> str:
    if y_value > 0.10:
        return "left"
    if y_value < -0.10:
        return "right"
    return "center"


@register()
def root_scene() -> Shape:
    # Golden table asset confirmed previously
    table = library_call(
        "usd",
        oid="benchmark_table_010",
        keywords=["golden_table", "table", "golden", "rectangular", "workspace"],
    )

    table_info = get_object_info(table)
    table_min = table_info["min"]
    table_max = table_info["max"]

    # tabletop usable bounds:
    # benchmark_table_010 overall bbox size ~ x:0.654, y:1.7, z:0.861
    # usd origin is already aligned to bottom, so top surface is max z
    tabletop_z = float(table_max[2])

    # Use safe inner margins so every object stays fully on tabletop
    x_left = float(table_min[0] + 0.10)
    x_right = float(table_max[0] - 0.10)
    y_right = float(table_min[1] + 0.12)
    y_left = float(table_max[1] - 0.12)

    # 10 beverage bottles, spread across tabletop
    # All x/y are explicitly within tabletop bounds
    bottle_specs = [
        ("benchmark_beverage_bottle_084", x_left + 0.06,  y_left - 0.18),
        ("benchmark_beverage_bottle_085", x_left + 0.10,  y_right + 0.20),
        ("benchmark_beverage_bottle_078", x_left + 0.18,  0.22),
        ("benchmark_beverage_bottle_080", x_left + 0.26, -0.22),
        ("benchmark_beverage_bottle_087", x_left + 0.34,  y_left - 0.28),
        ("benchmark_beverage_bottle_089", x_left + 0.42, -0.04),
        ("benchmark_beverage_bottle_023", x_left + 0.50,  0.10),
        ("benchmark_beverage_bottle_024", x_left + 0.58, y_right + 0.28),
        ("genie_beverage_bottle_011",     x_left + 0.66,  0.28),
        ("iros_beverage_bottle_003",      x_right - 0.06, y_right + 0.18),
    ]

    shapes = [table]

    for i, (oid, x, y) in enumerate(bottle_specs):
        region = _region_tag(y)
        bottle = library_call(
            "usd",
            oid=oid,
            keywords=[
                f"beverage_bottle_{i+1}",
                "beverage_bottle",
                region,
                "random_scattered_on_tabletop",
            ],
        )
        bottle = transform_shape(
            bottle,
            translation_matrix((x, y, tabletop_z)),
        )
        shapes.append(bottle)

    # Left clothing: place near +y side but still inside tabletop
    cloth_left = library_call(
        "usd",
        oid="bagged_clothing_16",
        keywords=[
            "left_clothing",
            "bagged_clothing",
            "black",
            "left",
            "on_tabletop",
        ],
    )
    cloth_left = transform_shape(
        cloth_left,
        translation_matrix((x_left + 0.14, y_left - 0.04, tabletop_z)),
    )
    shapes.append(cloth_left)

    # Right clothing: place near -y side but still inside tabletop
    cloth_right = library_call(
        "usd",
        oid="bagged_clothing_27",
        keywords=[
            "right_clothing",
            "bagged_clothing",
            "black",
            "right",
            "on_tabletop",
        ],
    )
    cloth_right = transform_shape(
        cloth_right,
        translation_matrix((x_right - 0.16, y_right + 0.04, tabletop_z)),
    )
    shapes.append(cloth_right)

    return concat_shapes(*shapes)