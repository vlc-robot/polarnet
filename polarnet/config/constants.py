
def get_workspace(real_robot=False):
    if real_robot:  
        # ur5 robotics room
        TABLE_HEIGHT = 0.01 # meters

        X_BBOX = (-1, 0)        # 0 is the robot base
        Y_BBOX = (-0.175, 0.4)  # 0 is the robot base
        Z_BBOX = (0, 0.75)      # 0 is the table
    else:
        # rlbench workspace
        TABLE_HEIGHT = 0.76 # meters

        X_BBOX = (-0.5, 1.5)    # 0 is the robot base
        Y_BBOX = (-1, 1)        # 0 is the robot base 
        Z_BBOX = (0.2, 2)       # 0 is the floor

    return {
        'TABLE_HEIGHT': TABLE_HEIGHT, 
        'X_BBOX': X_BBOX, 
        'Y_BBOX': Y_BBOX, 
        'Z_BBOX': Z_BBOX
    }




