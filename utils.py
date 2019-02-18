import torch
import torch.nn as nn
import collections
import numpy as np
import math
import pdb

# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias

# Rolling average 
class RollingAverage(object):
    def __init__(self, window_sz):
        self.window_sz = window_sz
        self.data = collections.deque()
        self.sum = None

    # Append item and update sum and data struct
    def append(self, item):
        assert(type(item) is np.ndarray)
        # If full, pop left and remove remove from average
        if len(self.data) == self.window_sz:
            removed = self.data.popleft()
            self.sum -= removed

        # Update sum with new item and add to data
        if len(self.data) == 0:
            self.sum = item
        else:
            self.sum += item
        self.data.append(item)
        assert(len(self.data) <= self.window_sz)

    # Return the average value
    def average(self):
        # Exception if list is empty
        if len(self.data) == 0:
            raise Exception("Can't compute rolling average on empty list")
        
        # Return average
        return self.sum / len(self.data)
        
# Convert to/from quaternion
def quaternion_to_euler_angle(w, x, y, z):
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

# Convert angle to an egocentric coordinate (All in unit vectors)
def convert_to_egocentric(ego_to_global_angle, global_angle):
    # ego_to_global_angle - the angle of the agent in the global coordinate system
    # global_angle - the angle (rad) in global coordinates we want to be egocentric
    ego_angle = global_angle - ego_to_global_angle
    if ego_angle > math.pi:
        ego_angle -= 2*math.pi
    elif ego_angle < -math.pi:
        ego_angle += 2*math.pi

    return ego_angle

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

# https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))
