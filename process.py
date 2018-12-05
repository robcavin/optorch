
import torch

ptsA_all = torch.load("ptsA").numpy()
ptsB_all = torch.load("ptsB").numpy()

import cv2 as cv
(H, status) = cv.findHomography(ptsA_all,ptsB_all,cv.RANSAC,10)

ptsA_truth = []
ptsB_truth = []
for i in range(len(status)) :
    if (status[i]) :
        ptsA_truth.append(ptsA_all[i])
        ptsB_truth.append(ptsB_all[i])

ptsA_truth = torch.tensor(ptsA_truth)
ptsB_truth = torch.tensor(ptsB_truth)

print H

#ptsA_truth = ptsA_truth[0:100]
#ptsB_truth = ptsB_truth[0:100]

# def rotate(vector, axis, theta) :
#     a = vector.mul(torch.cos(theta))
#     b = axis.cross(vector).mul(torch.sin(theta))
#     c1 = torch.tensor(1.0).sub(torch.cos(theta))
#     c = axis.dot(vector).mul(axis).mul(c1)
#     return torch.add(a,torch.add(b,c))
#
# def rotate_axis_angle(a,b) :
#     axis = torch.cat((b[:2], torch.tensor(1)))
#     axis /= axis.norm()
#     angle = b[2]
#     return rotate(a,axis,angle)

# def quaternion_to_matrix(q) :
#     q_m = torch.zeros(4,4)
#     q_m[0,:] = torch.tensor([q[0],-q[1],-q[2],-q[3]])
#     q_m[1,:] = torch.tensor([q[1], q[0], q[3],-q[2]])
#     q_m[2,:] = torch.tensor([q[2],-q[3], q[0], q[1]])
#     q_m[3,:] = torch.tensor([q[3], q[2],-q[1], q[0]])

import quaternion

class CameraCalibEval :
    def __init__(self) :
        self.R1 = torch.tensor([0.0,0.0,0.0], requires_grad=True)
        self.T1 = torch.zeros(3, requires_grad=True)
        self.K1 = torch.zeros(4,requires_grad=True)
        self.C1 = torch.tensor([2000.0,1500.0], requires_grad=True)
        self.F1 = torch.tensor([1000.0,1000.0], requires_grad=True)

    # compute world point guesses assuming 0 distortion, rotation, or translation
    def sensorFromWorld(self, pts_world):

        X1 = pts_world

        quat = torch.cat((torch.ones(1), self.R1))
        quat = quat / quat.norm()
        quat = quat.expand(X1.shape[0],4)
        X1 = quaternion.qrot(quat,X1)

        #X1_p = torch.cat((X1,torch.zeros(X1.shape[0],1)),dim=1)
        #X2 = quaternion.qmul(quat,X1_p)
        #qp = quat * torch.tensor([1.0,-1.0,-1.0,-1.0])
        #X3 = quaternion.qmul(X2,qp)
        #print pts_world, X1, quat

        X1.add_(self.T1)

        ab1 = X1.div(X1[:,2].view(X1.shape[0],1))
        ab1 = ab1[:,:2]

        r1 = ab1.mul(ab1)
        r1 = r1.sum(dim=1,keepdim=True)
        r1.sqrt_()

        theta1 = r1.atan()
        theta1_sq = theta1.mul(theta1)
        theta1_d = torch.zeros(theta1.shape)
        theta1_d.add_(self.K1[3])
        theta1_d.mul_(theta1_sq)
        theta1_d.add_(self.K1[2])
        theta1_d.mul_(theta1_sq)
        theta1_d.add_(self.K1[1])
        theta1_d.mul_(theta1_sq)
        theta1_d.add_(self.K1[0])
        theta1_d.mul_(theta1_sq)
        theta1_d.add_(1)
        theta1_d.mul_(theta1)

        #theta1_scale = theta1_d.div(r1)
        theta1_scale = theta1_d.div(theta1)
        x1_p = ab1.mul(theta1_scale)

        uv_1 = x1_p.mul(self.F1)
        uv_1.add_(self.C1)

        return uv_1


cam0 = CameraCalibEval()
pts_world_xy = torch.tensor(ptsA_truth)
pts_world_xy.sub_(cam0.C1)
pts_world_xy.div_(cam0.F1)
pts_world_xy.mul_(1000)
pts_world_xy.detach_()
pts_world_xy.requires_grad = True

cam0 = CameraCalibEval()
cam1 = CameraCalibEval()

learning_rate = 1e-4

pts_world_fixed_Z = torch.ones(pts_world_xy.shape[0], 1)
pts_world_fixed_Z.mul_(1000)


pts_world = torch.cat((pts_world_xy, pts_world_fixed_Z), dim=1)

with torch.no_grad():
    cam1.R1[0] = 0.1
    cam1.R1[1] = 0.2
    cam1.R1[2] = 0.3
    cam1.T1[0] = 100
    ptsB_toy_truth = cam1.sensorFromWorld(pts_world)

cam1 = CameraCalibEval()

optimizer = torch.optim.SGD([
    {"params":[pts_world_xy, cam0.F1, cam1.F1, cam1.T1],"lr":1e-6},
    {"params":[cam1.R1, cam0.K1, cam1.K1],"lr":1e-12}],
    lr=1e-12, momentum=0.9)

print "Starting!"
for t in range(50000) :

    optimizer.zero_grad()

    pts_world = torch.cat((pts_world_xy, pts_world_fixed_Z), dim=1)
    pts_pred0 = cam0.sensorFromWorld(pts_world)
    pts_pred1 = cam1.sensorFromWorld(pts_world)

    loss =  (ptsA_truth - pts_pred0).pow(2).sum()
    loss += (ptsB_truth - pts_pred1).pow(2).sum()
    #loss = (ptsB_toy_truth - pts_pred1).pow(2).sum()
    loss.backward()

    optimizer.step()
    if not t % 100 :
        print(t, loss.item(), cam1.R1, cam1.T1, cam1.F1, cam0.F1)


print("Reprojection error")
print(loss.sqrt().div(len(ptsA_truth) + len(ptsB_truth) ))

print pts_pred0
print ptsA_truth
print pts_world_xy
# Backwards but irrelevant
# ptsA_t.sub_(C1)
# ptsA_t.div_(F1)
#
# rA = torch.mul(ptsA_t,ptsA_t)
# rA = rA.sum(dim=1,keepdim=True)
# rA.sqrt_()
# rA.atan_()
