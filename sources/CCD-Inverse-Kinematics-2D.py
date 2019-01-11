import numpy as np 
import matplotlib.pyplot as plt 
import math

# Draw Axis
def draw_axis(ax, scale=1.0, A=np.eye(4), style='-', draw_2d = False):
    xaxis = np.array([[0, 0, 0, 1], 
                      [scale, 0, 0, 1]]).T
    yaxis = np.array([[0, 0, 0, 1], 
                      [0, scale, 0, 1]]).T
    zaxis = np.array([[0, 0, 0, 1], 
                      [0, 0, scale, 1]]).T
    
    xc = A.dot( xaxis )
    yc = A.dot( yaxis )
    zc = A.dot( zaxis )
    
    if draw_2d:
        ax.plot(xc[0,:], xc[1,:], 'r' + style)
        ax.plot(yc[0,:], yc[1,:], 'g' + style)
    else:
        ax.plot(xc[0,:], xc[1,:], xc[2,:], 'r' + style)
        ax.plot(yc[0,:], yc[1,:], yc[2,:], 'g' + style)
        ax.plot(zc[0,:], zc[1,:], zc[2,:], 'b' + style)

def rotateZ(theta):
    rz = np.array([[math.cos(theta), - math.sin(theta), 0, 0],
                   [math.sin(theta), math.cos(theta), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    return rz

def translate(dx, dy, dz):
    t = np.array([[1, 0, 0, dx],
                  [0, 1, 0, dy],
                  [0, 0, 1, dz],
                  [0, 0, 0, 1]])
    return t

def FK(angle, link):
    P0 = np.eye(4)
    R0_1 = rotateZ(angle[0]/180*math.pi)
    T0_1 = translate(link[0], 0, 0)
    P1 = P0.dot(R0_1).dot(T0_1)
    R1_2 = rotateZ(angle[1]/180*math.pi)
    T1_2 = translate(link[1], 0, 0)
    P2 = P1.dot(R1_2).dot(T1_2)
    R2_3 = rotateZ(angle[2]/180*math.pi)
    T2_3 = translate(link[2], 0, 0)
    P3 = P2.dot(R2_3).dot(T2_3)
    return P0, P1, P2, P3

def IK(target, angle, link, max_iter = 10000, err_min = 0.1):
    solved = False
    for loop in range(max_iter):
        P0, P1, P2, P3 = FK(angle, link)
        list_joint_pos = [P2, P1, P0]
        i = len(list_joint_pos) - 1
        for joint_pos in list_joint_pos:
            cur_to_end = P3[:3, 3] - joint_pos[:3, 3]
            cur_to_end_mag = math.sqrt(cur_to_end[0] ** 2 + cur_to_end[1] ** 2)
            cur_to_target = target - joint_pos[:3, 3]
            cur_to_target_mag = math.sqrt(cur_to_target[0] ** 2 + cur_to_target[1] ** 2)
            # print("cur_to_end :", cur_to_end)
            # print("cur_to_end_mag :", cur_to_end_mag)
            # print("cur_to_tgt :", cur_to_target)
            # print("cur_to_tgt_mag :", cur_to_target_mag)

            end_target_mag = cur_to_end_mag * cur_to_target_mag

            if end_target_mag <= 0.0001:    # prevent division by small numbers
                cos_rot_ang = 1
                sin_rot_ang = 0
            else:
                cos_rot_ang = (cur_to_end[0] * cur_to_target[0] + cur_to_end[1] * cur_to_target[1]) / end_target_mag
                sin_rot_ang = (cur_to_end[0] * cur_to_target[1] - cur_to_end[1] * cur_to_target[0]) / end_target_mag

            # print("cos_rot_ang :", cos_rot_ang)
            rot_ang = math.acos(max(-1, min(1,cos_rot_ang)))
            # print("rot_ang :", rot_ang)
            if sin_rot_ang < 0.0:
                rot_ang = -rot_ang

            # Disini untuk buat kondisi link
            angle[i] = angle[i] + (rot_ang * 180 / math.pi)
            if angle[i] >= 360:
                angle[i] = angle[i] - 360
            if angle[i] < 0:
                angle[i] = 360 + angle[i]

            _, _, _, P3 = FK(angle, link)

            end_to_target = target - P3[:3, 3]
            # print("end_to_tgt :", end_to_target)
            err_end_to_target = math.sqrt(end_to_target[0] ** 2 + end_to_target[1] ** 2)
            # print("err :", err_end_to_target)
            if err_end_to_target < err_min:
                solved = True      
            i -= 1
        if solved:
            break
    return angle, err_end_to_target, solved, loop

def main():
    l1, l2, l3 = 50, 40, 35
    t0, t1 = 20, 30
    t2 = -(t0+t1)

    # Robot Parameter
    angle = [t0, t1, t2]
    link = [l1, l2, l3]
    target = [80, 80, 0] 

    # Create figure to plot
    fig = plt.figure() 
    ax = fig.add_subplot(1,1,1)
    ax.set_xlim(-50, 130)
    ax.set_ylim(-50, 130)

    # Forward Kinematics
    P0, P1, P2, P3 = FK(angle, link)
    ax.plot([P0[0,3], P1[0,3]], [P0[1,3], P1[1,3]])
    ax.plot([P1[0,3], P2[0,3]], [P1[1,3], P2[1,3]])
    ax.plot([P2[0,3], P3[0,3]], [P2[1,3], P3[1,3]])
    
    angle, err, solved, iteration = IK(target, angle, link, max_iter=100)
    P0, P1, P2, P3 = FK(angle, link)
    ax.plot([P0[0,3], P1[0,3]], [P0[1,3], P1[1,3]])
    ax.plot([P1[0,3], P2[0,3]], [P1[1,3], P2[1,3]])
    ax.plot([P2[0,3], P3[0,3]], [P2[1,3], P3[1,3]])
    if solved:
        print("Iteration :", iteration)
        print("Angle :", angle)
        print("Target :", target)
        print("EEF :", P3[:3, 3])
        print("Error :", err)
    else:
        print("IK error")
        print("Target :", target)
        print("EEF :", P3[:3, 3])
        print("Error :", err)
    plt.show()

if __name__ == "__main__":
    main()