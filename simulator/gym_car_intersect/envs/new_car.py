import numpy as np
import bisect
import math
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener, shape)

# Top-down car dynamics simulation.
#
# Some ideas are taken from this great tutorial http://www.iforce2d.net/b2dtut/top-down-car by Chris Campbell.
# This simulation is a bit more detailed, with wheels rotation.
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

SIZE = 0.02
ENGINE_POWER            = 100000000*SIZE*SIZE
WHEEL_MOMENT_OF_INERTIA = 4000*SIZE*SIZE
FRICTION_LIMIT          = 1000000*SIZE*SIZE     # friction ~= mass ~= size^2 (calculated implicitly using density)
WHEEL_R  = 27
WHEEL_W  = 14
CENTROID = 220 # that is point which follows target in car...
WHEELPOS = [
    (-45,+60-CENTROID), (+45,+60-CENTROID),
    (-45,-70-CENTROID), (+45,-70-CENTROID)
    ]
HULL_POLY4 =[
    (-45,-105-CENTROID), (+45,-105-CENTROID),
    (-45,+105-CENTROID),  (+45,+105-CENTROID)
    ]
SENSOR_SHAPE =[
    (-45,-105-CENTROID), (+45,-105-CENTROID),
    (-45,+105-CENTROID),  (+45,+105-CENTROID)
    ]
## Point sensor:
# SENSOR_BOT = [
#     (-10,350-CENTROID), (+10,350-CENTROID),
#     (-10,+360-CENTROID),  (+10,+360-CENTROID)
# ]
SENSOR_BOT = [
    (-50,+110-CENTROID), (+50,+110-CENTROID),
    (-10,+300-CENTROID),  (+10,+300-CENTROID)
]
# SENSOR_ADD = [
#     (-1,+110-CENTROID), (+1,+110-CENTROID),
#     (-50,+200-CENTROID),  (+50,+200-CENTROID)
# ]
WHEEL_COLOR = (0.0,0.0,0.0)
WHEEL_WHITE = (0.3,0.3,0.3)

class DummyCar:
    def __init__(self, world, init_coord, penalty_sec = set(), color = None, bot=False):
        """ Constructor to define Car.
        Parameters
        ----------
        world : Box2D World
        init_coord : tuple
            (angle, x, y)
        penalty_sec : set
            Numbers from 2..9 which define sections where car can't be
            so penalty can be assigned
        color : tuple
            Selfexplanatory
        """

        init_angle, init_x, init_y = init_coord
        SENSOR = SENSOR_BOT if bot else SENSOR_SHAPE
        self.world = world
        # # make two sensor dots close and far...
        # additional_fixture = [fixtureDef(shape=polygonShape(vertices=[(x*SIZE, y*SIZE) for x, y in SENSOR_ADD]),
        #                                 isSensor=True, userData='sensor')] if bot else []
        self.hull = self.world.CreateDynamicBody(
            position = (init_x, init_y),
            angle = init_angle,
            fixtures = [fixtureDef(shape = polygonShape(vertices=[ (x*SIZE,y*SIZE) for x,y in HULL_POLY4 ]),
                                density=1.0, userData='body'),
                        fixtureDef(shape = polygonShape(vertices=[(x*SIZE, y*SIZE) for x, y in SENSOR]),
                                isSensor=True, userData='sensor')])# + additional_fixture)
        self.hull.color = color or ((0.2, 0.8, 1) if bot else (0.8,0.0,0.0))
        self.hull.name = 'bot_car' if bot else 'car'
        self.hull.cross_time = float('inf')
        self.hull.stop = False
        self.hull.collision = False
        self.hull.penalty = False
        self.hull.penalty_sec = penalty_sec
        self.hull.path = ''
        self.hull.userData = self.hull
        self.wheels = []
        self.fuel_spent = 0.0
        WHEEL_POLY = [
            (-WHEEL_W,+WHEEL_R), (+WHEEL_W,+WHEEL_R),
            (+WHEEL_W,-WHEEL_R), (-WHEEL_W,-WHEEL_R)
            ]
        for wx,wy in WHEELPOS:
            front_k = 1.0 if wy > 0 else 1.0
            w = self.world.CreateDynamicBody(
                position = (init_x+wx*SIZE, init_y+wy*SIZE),
                angle = init_angle,
                fixtures = fixtureDef(
                    shape=polygonShape(vertices=[ (x*front_k*SIZE,y*front_k*SIZE) for x,y in WHEEL_POLY ]),
                    density=0.1,
                    categoryBits=0x0020,
                    maskBits=0x001,
                    restitution=0.0)
                    )
            w.wheel_rad = front_k*WHEEL_R*SIZE
            w.color = WHEEL_COLOR
            w.gas   = 0.0
            w.brake = 0.0
            w.steer = 0.0
            w.phase = 0.0  # wheel angle
            w.omega = 0.0  # angular velocity
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=w,
                localAnchorA=(wx*SIZE,wy*SIZE),
                localAnchorB=(0,0),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=180*900*SIZE*SIZE,
                motorSpeed = 0,
                lowerAngle = -0.4,
                upperAngle = +0.4,
                )
            w.joint = self.world.CreateJoint(rjd)
            w.tiles = set()
            w.name = 'wheel'
            w.collision = False
            w.penalty = False
            w.penalty_sec = penalty_sec
            w.userData = w
            self.wheels.append(w)
        self.drawlist =  self.wheels + [self.hull]
        self.target = (0, 0)

    def gas(self, gas):
        'control: rear wheel drive'
        gas = np.clip(gas, 0, 1)
        gas /= 10
        for w in self.wheels[2:4]:
            diff = gas - w.gas
            if diff > 0.01: diff = 0.01  # gradually increase, but stop immediately
            w.gas += diff

    def brake(self, b):
        'control: brake b=0..1, more than 0.9 blocks wheels to zero rotation'
        b = np.clip(b, 0, 1)
        for w in self.wheels:
            w.brake = b

    def steer(self, s):
        'control: steer s=-1..1, it takes time to rotate steering wheel from side to side, s is target position'
        s = np.clip(s, -1, 1)
        self.wheels[0].steer = s
        self.wheels[1].steer = s

    def close_to_target(self, target_path, dist=5):
        x, y = round(self.hull.position.x, 2), round(self.hull.position.y, 2)
        x_pos, y_pos = target_path
        return np.sqrt((x-x_pos)**2 + (y-y_pos)**2) < dist

    def go_to_target(self, target_path, path_tree):
        # finding closest point on target path:
        x, y = round(self.hull.position.x, 2), round(self.hull.position.y, 2)
        dist, idx = path_tree.query((x, y))
        # print(target_path)
        # print('x, y: ', x, y)
        idx = idx if idx < len(target_path)-2 else len(target_path)-1
        x_pos, y_pos = target_path[idx]
        self.target = (x_pos, y_pos)
        # as atan give angle in different direction than car angle.
        target_angle = -math.atan2(x_pos-x, y_pos-y)
        # print(f"target angle: {target_angle*180/math.pi:.2f}")
        # print(f"car angle: {self.car.hull.angle*180/math.pi:.2f}")
        # atan gives positive angle from Oy to Ox
        # car.angle give positive angle from Oy to -Ox
        # we steer car in closest direction
        # -1 clockwize +1 conterclockwize
        # -math.pi/2 to makae cos scale direction values.
        direction = -math.pi/2 + target_angle - self.hull.angle
        # the steering is cos between two vectors of current postiion and next one:
        self.steer(math.cos(direction))

        self.gas(0.2)
        self.brake(0)

    def step(self, dt):
        for w in self.wheels:
            # Steer each wheel
            dir = np.sign(w.steer - w.joint.angle)
            val = abs(w.steer - w.joint.angle)
            w.joint.motorSpeed = dir*min(50.0*val, 2.0)

            # Position => friction_limit
            grass = True
            friction_limit = FRICTION_LIMIT*0.6  # Grass friction if no tile
            for tile in w.tiles:
                friction_limit = max(friction_limit, FRICTION_LIMIT*tile.road_friction)
                grass = False

            # Force
            forw = w.GetWorldVector( (0,1) )
            side = w.GetWorldVector( (1,0) )
            v = w.linearVelocity
            vf = forw[0]*v[0] + forw[1]*v[1]  # forward speed
            vs = side[0]*v[0] + side[1]*v[1]  # side speed

            # WHEEL_MOMENT_OF_INERTIA*np.square(w.omega)/2 = E -- energy
            # WHEEL_MOMENT_OF_INERTIA*w.omega * domega/dt = dE/dt = W -- power
            # domega = dt*W/WHEEL_MOMENT_OF_INERTIA/w.omega
            w.omega += dt*ENGINE_POWER*w.gas/WHEEL_MOMENT_OF_INERTIA/(abs(w.omega)+5.0)  # small coef not to divide by zero
            self.fuel_spent += dt*ENGINE_POWER*w.gas

            if w.brake >= 0.9:
                w.omega = 0
            elif w.brake > 0:
                BRAKE_FORCE = 15    # radians per second
                dir = -np.sign(w.omega)
                val = BRAKE_FORCE*w.brake
                if abs(val) > abs(w.omega): val = abs(w.omega)  # low speed => same as = 0
                w.omega += dir*val
            w.phase += w.omega*dt

            vr = w.omega*w.wheel_rad  # rotating wheel speed
            f_force = -vf + vr        # force direction is direction of speed difference
            p_force = -vs

            # Physically correct is to always apply friction_limit until speed is equal.
            # But dt is finite, that will lead to oscillations if difference is already near zero.
            f_force *= 205000*SIZE*SIZE  # Random coefficient to cut oscillations in few steps (have no effect on friction_limit)
            p_force *= 205000*SIZE*SIZE
            force = np.sqrt(np.square(f_force) + np.square(p_force))

            if abs(force) > friction_limit:
                f_force /= force
                p_force /= force
                force = friction_limit  # Correct physics here
                f_force *= force
                p_force *= force

            w.omega -= dt*f_force*w.wheel_rad/WHEEL_MOMENT_OF_INERTIA

            w.ApplyForceToCenter( (
                p_force*side[0] + f_force*forw[0],
                p_force*side[1] + f_force*forw[1]), True )

    def draw(self, viewer):
        for obj in self.drawlist:
            for f in obj.fixtures[:1]:
                trans = f.body.transform
                path = [trans*v for v in f.shape.vertices]
                viewer.draw_polygon(path, color=obj.color)
                if "phase" not in obj.__dict__: continue
                a1 = obj.phase
                a2 = obj.phase + 1.2  # radians
                s1 = math.sin(a1)
                s2 = math.sin(a2)
                c1 = math.cos(a1)
                c2 = math.cos(a2)
                if s1>0 and s2>0: continue
                if s1>0: c1 = np.sign(c1)
                if s2>0: c2 = np.sign(c2)
                white_poly = [
                    (-WHEEL_W*SIZE, +WHEEL_R*c1*SIZE), (+WHEEL_W*SIZE, +WHEEL_R*c1*SIZE),
                    (+WHEEL_W*SIZE, +WHEEL_R*c2*SIZE), (-WHEEL_W*SIZE, +WHEEL_R*c2*SIZE)
                    ]
                viewer.draw_polygon([trans*v for v in white_poly], color=WHEEL_WHITE)

    def destroy(self):
        self.world.DestroyBody(self.hull)
        self.hull = None
        for w in self.wheels:
            self.world.DestroyBody(w)
        self.wheels = []
