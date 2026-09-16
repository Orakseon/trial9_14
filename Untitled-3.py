"""
Cityscape Lite Exact Map
with five walking pedestrians

Based on QLabs People Tutorial
"""


import math
import time
import threading


from qvl.qlabs import QuanserInteractiveLabs
from qvl.system import QLabsSystem
from qvl.free_camera import QLabsFreeCamera

from qvl.crosswalk import QLabsCrosswalk
from qvl.traffic_light import QLabsTrafficLight
from qvl.traffic_cone import QLabsTrafficCone
from qvl.qcar2 import QLabsQCar2
from qvl.person import QLabsPerson



SCALE = 10



# ===============================
# 五条斑马线路径
# ===============================


PEDESTRIAN_PATHS = [


    # 北侧
    {
        "start":[-2,14,0.005],
        "end":[5,14,0.005],
        "yaw":0
    },


    # 南侧
    {
        "start":[-2,5,0.005],
        "end":[5,5,0.005],
        "yaw":0
    },


    # 西侧
    {
        "start":[-4,6.5,0.005],
        "end":[-4,12,0.005],
        "yaw":math.pi/2
    },


    # 东侧
    {
        "start":[6,6.5,0.005],
        "end":[6,12,0.005],
        "yaw":math.pi/2
    },


    # 右上斜向
    {
        "start":[7.5,35.5,0.005],
        "end":[10.5,38.5,0.005],
        "yaw":math.radians(17)
    }

]



# ===============================
# 斑马线
# ===============================


CROSSWALKS=[


([0.1,1.4,0],[0,0,0]),

([0.1,0.5,0],[0,0,0]),

([-0.4,0.9,0],[0,0,90]),

([0.6,0.9,0],[0,0,90]),

([0.9,3.7,0],[0,0,17])

]



def spawn_crosswalks(qlabs):


    cw=QLabsCrosswalk(qlabs)


    for p,r in CROSSWALKS:


        cw.spawn_degrees(

            location=[
                p[0]*SCALE,
                p[1]*SCALE,
                p[2]
            ],

            rotation=r,

            scale=[1,1,1],

            configuration=0,

            waitForConfirmation=True

        )





# ===============================
# 红绿灯
# ===============================


# ===============================
# 红绿灯
# 根据道路方向调整
# ===============================


def spawn_lights(qlabs):


    lights=[


        # =========================
        # 西北角
        # 原方向逆时针180°
        # =========================
        {
            "pos":[-0.4,1.4,0],
            "yaw":180
        },



        # =========================
        # 东北角
        # 原方向逆时针90°
        # =========================
        {
            "pos":[0.6,1.4,0],
            "yaw":90
        },



        # =========================
        # 西南角
        # 原方向逆时针90°
        # =========================
        {
            "pos":[-0.4,0.5,0],
            "yaw":270
        },



        # =========================
        # 东南角
        # 保持不动
        # =========================
        {
            "pos":[0.6,0.5,0],
            "yaw":360
        }

    ]



    for i,data in enumerate(lights):


        light=QLabsTrafficLight(qlabs)



        light.spawn_id_degrees(


            actorNumber=i,


            location=[

                data["pos"][0]*SCALE,

                data["pos"][1]*SCALE,

                0

            ],


            rotation=[

                0,

                0,

                data["yaw"]

            ],


            configuration=0,


            waitForConfirmation=True

        )





# ===============================
# 四个锥桶
# ===============================
#
# 按参考图左侧道路上的四个橙色点布置。
# 参考图坐标约为：
#   (-1.90, 3.70)
#   (-1.90, 3.55)
#   (-1.75, 3.25)
#   (-1.75, 3.05)
#
# 本程序使用 SCALE=10，因此转换成 QLabs 世界坐标：
#   (-19.0, 37.0)
#   (-19.0, 35.5)
#   (-17.5, 32.5)
#   (-17.5, 30.5)
# ===============================

CONE_POSITIONS = [
    [-19.0, 37.0, 0.25],
    [-19.0, 35.5, 0.25],
    [-17.5, 32.5, 0.25],
    [-17.5, 30.5, 0.25],
]


def spawn_cones(qlabs):

    cone = QLabsTrafficCone(qlabs)

    for i, position in enumerate(CONE_POSITIONS):

        cone.spawn_id_degrees(
            actorNumber=200 + i,
            location=position,
            rotation=[0, 0, 0],
            scale=[1, 1, 1],
            configuration=0,
            waitForConfirmation=True
        )


# ===============================
# 行人生成
# ===============================


def spawn_people(qlabs):


    people=[]


    for i,data in enumerate(
        PEDESTRIAN_PATHS
    ):


        person=QLabsPerson(
            qlabs
        )


        person.spawn_id(

            actorNumber=100+i,


            location=data["start"],


            rotation=[

                0,

                0,

                data["yaw"]

            ],


            scale=[1,1,1],


            configuration=6,


            waitForConfirmation=True

        )


        people.append(

            {

            "person":person,

            "start":data["start"],

            "end":data["end"],

            "yaw":data["yaw"]

            }

        )


    return people





# ===============================
# 行人循环
# ===============================


def pedestrian_loop(data):


    person=data["person"]


    while True:


        # 过马路

        person.move_to(

            location=data["end"],

            speed=person.WALK,

            waitForConfirmation=True

        )


        time.sleep(1)



        # 掉头

        person.move_to(

            location=data["start"],

            speed=person.WALK,

            waitForConfirmation=True

        )


        time.sleep(1)







def start_people(people):


    for p in people:


        t=threading.Thread(

            target=pedestrian_loop,

            args=(p,)

        )


        t.daemon=True

        t.start()







# ===============================
# 主程序
# ===============================


def main():


    qlabs=QuanserInteractiveLabs()


    if not qlabs.open("localhost"):

        return



    qlabs.destroy_all_spawned_actors()



    QLabsSystem(qlabs).set_title_string(

        "Cityscape Lite Pedestrian + 4 Traffic Cones"

    )


    spawn_crosswalks(qlabs)


    spawn_lights(qlabs)


    # 根据参考图添加四个交通锥桶
    spawn_cones(qlabs)


    people=spawn_people(qlabs)


    start_people(people)



    car=QLabsQCar2(qlabs)


    car.spawn_id(

        actorNumber=0,

        location=[0,13,0],

        rotation=[0,0,-math.pi/2],

        waitForConfirmation=True

    )



    camera=QLabsFreeCamera(qlabs)


    camera.spawn(

        [-2,20,35],

        [0,0,0]

    )


    camera.possess()



    while True:

        time.sleep(1)





if __name__=="__main__":

    main()