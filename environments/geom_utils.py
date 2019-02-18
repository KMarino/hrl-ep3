import math
import numpy as np
import pdb

# Convert to/from quaternion
def quaternion_to_euler_angle(quart):
    w = quart[0]
    x = quart[1]
    y = quart[2]
    z = quart[3]

    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    roll = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw

def euler_angle_to_quaternion(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5);
    sy = math.sin(yaw * 0.5);
    cr = math.cos(roll * 0.5);
    sr = math.sin(roll * 0.5);
    cp = math.cos(pitch * 0.5);
    sp = math.sin(pitch * 0.5);

    w = cy * cr * cp + sy * sr * sp
    x = cy * sr * cp - sy * cr * sp
    y = cy * cr * sp + sy * sr * cp
    z = sy * cr * cp - cy * sr * sp

    return w, x, y, z

# Angle to unit vector
def angle_to_unit(angle):
    x = math.cos(angle)
    y = math.sin(angle)
    return np.array([x, y])

# Unit vector to angle
def unit_to_angle(v):
    x = v[0]
    y = v[1]
    angle = math.atan2(y, x)

# Average angles (do this by averaging unit vectors)
def average_angles(angles):
    # Convert to unit vectors and average
    unit_vecs = [angle_to_unit(a) for a in angles]
    avg_dir = np.mean(unit_vecs, 0)

    # Return direction of the average unit vector
    avg_angle = math.atan2(avg_dir[1], avg_dir[0])
    return avg_angle

# Convert angle to an egocentric coordinate
def convert_to_egocentric(ego_to_global_angle, global_angle):
    # ego_to_global_angle - the angle of the agent in the global coordinate system
    # global_angle - the angle (rad) in global coordinates we want to be egocentric
    ego_angle = global_angle - ego_to_global_angle
    if ego_angle > math.pi:
        ego_angle -= 2*math.pi
    elif ego_angle < -math.pi:
        ego_angle += 2*math.pi

    return ego_angle

# Convert vector to an egocentric coordinate
def convert_vector_to_egocentric(ego_to_global_angle, vector):
    #pdb.set_trace()
    # Get magnitude and direction
    xy_mag = np.linalg.norm(vector)
    xy_angle = math.atan2(vector[1], vector[0])

    # Change direction to egocentric
    xy_angle = convert_to_egocentric(ego_to_global_angle, xy_angle)

    # Reform the vector
    x = xy_mag * math.cos(xy_angle)
    y = xy_mag * math.sin(xy_angle)
    ego_vec = np.array([x, y])

    return ego_vec
