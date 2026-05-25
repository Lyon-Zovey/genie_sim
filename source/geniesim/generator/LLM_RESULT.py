from helper import *

"""
scene_name: brown_book_on_white_table_center
description: 一张白色桌子，桌面正中间放置一本棕色的书。
"""


@register()
def brown_book_on_white_table_center() -> Shape:
    table_shape = library_call(
        "usd",
        oid="table_000",
        keywords=["main_white_table", "table", "white", "desktop", "center"],
    )

    book_shape = library_call(
        "usd",
        oid="benchmark_book_00",
        keywords=["center_brown_book", "book", "brown", "rectangular", "center"],
    )

    table_info = get_object_info(table_shape)
    desktop_info = get_subpart_info("table_000", "desktop")

    desktop_center_world = np.array(table_info["center"]) + np.array(desktop_info["center"])
    desktop_top_z = table_info["center"][2] + desktop_info["xyz_max"][2]

    book_info = get_object_info(book_shape)
    book_bottom_offset = -book_info["min"][2]

    placed_book_shape = transform_shape(
        book_shape,
        translation_matrix((
            float(desktop_center_world[0]),
            float(desktop_center_world[1]),
            float(desktop_top_z + book_bottom_offset),
        )),
    )

    return concat_shapes(table_shape, placed_book_shape)


@register()
def root_scene() -> Shape:
    return library_call("brown_book_on_white_table_center")