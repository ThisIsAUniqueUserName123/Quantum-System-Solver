
def zero_boundary_condition(axis):
    def zero_boundary_condition_axis(indices_grid, num_divisions):
        mask = (indices_grid[axis] >= 0) & (indices_grid[axis] < num_divisions[axis])
        return indices_grid, mask
    return zero_boundary_condition_axis

def periodic_boundary_condition(axis):
    def periodic_boundary_condition_axis(indices_grid, num_divisions):
        indices_grid[axis] = indices_grid[axis] % num_divisions[axis]
        return indices_grid, None
    return periodic_boundary_condition_axis

def radial_boundary_condition(radial_axis, azimuth_axis, polar_axis):
    def radial_boundary_condition_axis(indices_grid, num_divisions):
        negative_indices = indices_grid[radial_axis] < 0
        indices_grid[radial_axis][negative_indices] = -indices_grid[radial_axis][negative_indices]
        indices_grid[azimuth_axis][negative_indices] = (indices_grid[azimuth_axis][negative_indices] + num_divisions[azimuth_axis] // 2) % num_divisions[azimuth_axis]
        indices_grid[polar_axis][negative_indices] = num_divisions[polar_axis]//2 - 1 - indices_grid[polar_axis][negative_indices]
        retention_mask = indices_grid[radial_axis] < num_divisions[radial_axis]
        return indices_grid, retention_mask
    return radial_boundary_condition_axis

def polar_boundary_condition_3D(azimuth_axis, polar_axis):
    def polar_boundary_condition_axis(indices_grid, num_divisions):
        below = indices_grid[polar_axis] < 0
        above = indices_grid[polar_axis] >= num_divisions[polar_axis]
        either = below | above
        indices_grid[polar_axis][below] = -indices_grid[polar_axis][below]
        indices_grid[polar_axis][above] = 2 * num_divisions[polar_axis] - 1 - indices_grid[polar_axis][above]
        indices_grid[azimuth_axis][either] = (indices_grid[azimuth_axis][either] + num_divisions[azimuth_axis] // 2) % num_divisions[azimuth_axis]
        return indices_grid, None
    return polar_boundary_condition_axis

def polar_boundary_condition_2D(radial_axis, polar_axis):
    def polar_boundary_condition_axis(indices_grid, num_divisions):
        negative_indices = indices_grid[radial_axis] < 0
        indices_grid[radial_axis][negative_indices] = -indices_grid[radial_axis][negative_indices]-1
        indices_grid[polar_axis][negative_indices] = (indices_grid[polar_axis][negative_indices] + num_divisions[polar_axis] // 2) % num_divisions[polar_axis]

        retention_mask = indices_grid[radial_axis] < num_divisions[radial_axis]
        return indices_grid, retention_mask
    return polar_boundary_condition_axis