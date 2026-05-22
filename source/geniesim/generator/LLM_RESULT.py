from helper import *
import numpy as np

"""
scene_name: book_on_left_table_center
description: Place benchmark_book_03 at the center of the top surface of benchmark_desk_decoration_007,
with both table and book shifted to the left side of the scene and tagged with left keywords.
"""


@register()
def book_on_left_table_center() -> Shape:
    left_offset = np.array([0.0, 1.2, 0.0])

    table_shape = library_call(
        "usd",
        oid="benchmark_desk_decoration_007",
        keywords=[
            "left_center_table",
            "table",
            "desk_decoration",
            "support_surface",
            "left",
        ],
    )
    table_shape = transform_shape(table_shape, translation_matrix(left_offset))

    table_info = get_object_info(table_shape)
    table_center = np.array(table_info["center"])
    table_top_z = float(table_info["max"][2])

    book_shape = library_call(
        "usd",
        oid="benchmark_book_03",
        keywords=[
            "left_center_book",
            "book",
            "hardcover",
            "reading",
            "left",
        ],
    )

    book_info = get_object_info(book_shape)
    book_center = np.array(book_info["center"])

    translation = np.array(
        [
            table_center[0] - book_center[0],
            table_center[1] - book_center[1],
            table_top_z - book_info["min"][2],
        ]
    )
    placed_book = transform_shape(book_shape, translation_matrix(translation))

    return concat_shapes(table_shape, placed_book)


@register()
def root_scene() -> Shape:
    return library_call("book_on_left_table_center")