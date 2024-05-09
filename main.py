from simpleSFM import people_flow
from llmagent import people_flow1

people_num = 1
v_arg = [6,2]
repul_h = [5,5]
repul_m = [2,2]
target = [[60,240],[120,150],[90,60],[240,40],[200,120],[170,70],[150,0]]
R = 3
min_p = 0.1
p_arg = [[0.5,0.1]]
wall_x = 300
wall_y = 300
in_target_d = 3
dt = 0.1
dt2 = 5
save_format = "heat_map"
save_params = [(30,30),1]


model = people_flow(people_num, v_arg, repul_h, repul_m, target, R, min_p, p_arg, wall_x, wall_y, in_target_d, dt,
                    save_format=save_format, save_params=save_params)
model2 = people_flow1(people_num, wall_x, wall_y, dt2)

#maps = model.simulate()
model2.simulate()