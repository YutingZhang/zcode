
def canonicalize_field_name_tuple(field_names):
    if isinstance(field_names, str):
        field_names = (field_names,)
    return tuple(field_names)


def canonicalize_field_type_tuple(field_types):
    if not isinstance(field_types, tuple):
        field_types = (field_types,)
    return field_types


def canonicalize_field_shape_tuple(field_shapes):
    if field_shapes is None:
        return None
    if not isinstance(field_shapes, tuple):
        field_shapes = (field_shapes,)
    return field_shapes

