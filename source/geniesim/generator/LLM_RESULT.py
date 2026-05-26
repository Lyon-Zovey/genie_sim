from helper import *

"""
scene_name: golden_table_with_10_beverage_bottles
description: 一张金色桌子上随机散放10个饮料瓶，瓶子全部位于桌面上，且瓶子间距更宽
"""

import numpy as np


def _sample_pose_candidates():
    # 拉大间距后的桌面局部离散点
    # x: 沿桌长方向分布更开
    # y: 左右方向也更分散
    return [
        (-0.62,  0.34),
        (-0.62, -0.30),
        (-0.30,  0.18),
        (-0.28, -0.18),
        ( 0.02,  0.32),
        ( 0.02, -0.30),
        ( 0.36,  0.14),
        ( 0.38, -0.14),
        ( 0.70,  0.28),
        ( 0.72, -0.26),
    ]


def _region_tag(y_value: float) -> str:
    if y_value > 0.10:
        return "left"
    if y_value < -0.10:
        return "right"
    return "center"


@register()
def root_scene() -> Shape:
    table_shape = library_call(
        "usd",
        oid="benchmark_table_010",
        keywords=["golden_table", "table", "golden", "rectangular", "workspace"],
    )

    table_info = get_object_info(table_shape)
    table_min = table_info["min"]
    table_max = table_info["max"]
    tabletop_z = float(table_max[2])

    bottle_ids = [
        "benchmark_beverage_bottle_084",
        "benchmark_beverage_bottle_085",
        "benchmark_beverage_bottle_078",
        "benchmark_beverage_bottle_080",
        "benchmark_beverage_bottle_087",
        "benchmark_beverage_bottle_089",
        "benchmark_beverage_bottle_023",
        "benchmark_beverage_bottle_024",
        "genie_beverage_bottle_011",
        "iros_beverage_bottle_003",
    ]

    local_xy = _sample_pose_candidates()
    np.random.shuffle(local_xy)

    placed_bottles = []
    for i, oid in enumerate(bottle_ids):
        lx, ly = local_xy[i]

        # 更大的边缘留白，确保更宽松摆放
        x = float(
            np.clip(
                table_min[0] + 0.18 + (lx + 0.62) / (0.72 + 0.62) * (table_max[0] - table_min[0] - 0.36),
                table_min[0] + 0.12,
                table_max[0] - 0.12,
            )
        )
        y = float(np.clip(ly, table_min[1] + 0.14, table_max[1] - 0.14))

        region = _region_tag(y)

        bottle = library_call(
            "usd",
            oid=oid,
            keywords=[
                f"beverage_bottle_{i+1}",
                "beverage_bottle",
                region,
                "random_scattered_on_table",
                "wider_spacing",
            ],
        )

        bottle = transform_shape(
            bottle,
            translation_matrix((x, y, tabletop_z)),
        )

        placed_bottles.append(bottle)

    return concat_shapes(table_shape, *placed_bottles)