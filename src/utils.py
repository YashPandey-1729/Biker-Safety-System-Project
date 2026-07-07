def is_inside(inner_box, outer_box):
    """
    Computes whether the center point of an inner bounding box (e.g., a person)
    physically sits inside the boundaries of an outer box (a motorcycle).
    """
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box
    
    # Calculate target's inner center point coordinates
    cx = int((ix1 + ix2) / 2)
    cy = int((iy1 + iy2) / 2)
    
    # Evaluate boolean containment
    return (ox1 <= cx <= ox2) and (oy1 <= cy <= oy2)