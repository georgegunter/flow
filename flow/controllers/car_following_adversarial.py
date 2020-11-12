import math
import numpy as np

from flow.controllers.base_controller import BaseController


class ACC_Switched_Controller_Attacked(BaseController):

    def __init__(self,
                 veh_id,
                 car_following_params,
                 k_1=0.1,
                 k_2=0.2,
                 V_m=30,
                 h=1.2,
                 d_min=8.0,
                 SS_Threshold_min=15,
                 SS_Threshold_range=20,
                 Total_Attack_Duration = 3.0,
                 attack_decel_rate = -.8,
                 display_attack_info = False,
                 warmup_steps = 1000,
                 time_delay=0.0,
                 noise=0,
                 fail_safe=None):
        """Instantiate a Switched Adaptive Cruise controller with Cruise Control."""
        BaseController.__init__(
            self,
            veh_id,
            car_following_params,
            delay=time_delay,
            fail_safe=fail_safe,
            noise=noise)

        print('Attack Vehicle Spawned: '+veh_id)

        self.veh_id = veh_id
        self.k_1 = 1.0
        self.k_2 = 1.0
        self.k_3 = 0.5
        self.d_min = d_min
        self.V_m = V_m
        self.h = h
        self.isUnderAttack = False
        self.numSteps_Steady_State = 0

        self.SS_Threshold = SS_Threshold_min + np.random.rand()*SS_Threshold_range #number seconds at SS to initiate attack

        self.Total_Attack_Duration = Total_Attack_Duration #How long attack lasts for
        self.Curr_Attack_Duration = 0.0 
        self.attack_decel_rate = attack_decel_rate #Rate at which ACC decelerates
        self.a = 0.0
        self.display_attack_info = display_attack_info
        self.warmup_steps = warmup_steps


    def Attack_accel(self,env):
        #Declerates the car for a set period at a set rate:

        self.a = self.attack_decel_rate

        self.Curr_Attack_Duration += env.sim_step

        s = env.k.vehicle.get_headway(self.veh_id)
        L = env.k.vehicle.get_length(self.veh_id)
        s = s - L
        v = env.k.vehicle.get_speed(self.veh_id) 

        if(s < (v*(self.h-.2))):
            #If vehicle in front is getting too close, break from disturbance
            self.Reset_After_Attack(env)

        if(self.Curr_Attack_Duration >= self.Total_Attack_Duration):
            self.Reset_After_Attack(env)

    def Reset_After_Attack(self,env):
        self.isUnderAttack = False
        self.numSteps_Steady_State = 0
        self.Curr_Attack_Duration = 0.0
        pos  = env.k.vehicle.get_position(self.veh_id)
        lane = env.k.vehicle.get_lane(self.veh_id)
        if(self.display_attack_info):
            print('Attacker Finished: '+str(self.veh_id))
            print('Position of Attack: '+str(pos))
            print('Lane of Attack: '+str(lane))
            print('Time finished: '+str(env.step_counter*env.sim_step))

    def Check_For_Steady_State(self):
        self.numSteps_Steady_State += 1

        # if((self.a < .1) | (self.a > -.1)):
        #     self.numSteps_Steady_State += 1
        # else:
        #     self.numSteps_Steady_State = 0

    def normal_ACC_accel(self,env):
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        v_l = env.k.vehicle.get_speed(lead_id)
        v = env.k.vehicle.get_speed(self.veh_id)
        s = env.k.vehicle.get_headway(self.veh_id)
        L = env.k.vehicle.get_length(self.veh_id)
        s = s - L


        u = self.accel_func(v, v_l, s)

        self.a = u

    def accel_func(self,v,v_l,s):

        max_follow_dist = self.h*self.V_m

        if((s > max_follow_dist) or (v >= self.V_m)):
            # Switch to speed cotnrol if leader too far away, and max speed at V_m:
            u = self.Cruise_Control_accel(v)
        else:
            u = self.ACC_accel(v,v_l,s)

        return u

    def Cruise_Control_accel(self,v):
        return self.k_3*(self.V_m - v)

    def ACC_accel(self,v,v_l,s):
        ex = s - v*self.h - self.d_min
        ev = v_l - v
        return self.k_1*ex+self.k_2*ev

    def Check_Start_Attack(self,env):

        step_size = env.sim_step
        SS_length = step_size * self.numSteps_Steady_State
        if(SS_length >= self.SS_Threshold):
            if(not self.isUnderAttack):
                if(self.display_attack_info):
                    print('Beginning attack: '+self.veh_id+' Time: '+str(env.step_counter*env.sim_step))
            self.isUnderAttack = True
        else:
            self.isUnderAttack = False

    def get_accel(self, env):
        """See parent class."""

        is_passed_warmup = env.step_counter > self.warmup_steps

        perform_attack = self.isUnderAttack and is_passed_warmup

        if(perform_attack):
            #Attack under way:
            self.Attack_accel(env)
            if(self.display_attack_info):
                print('Attacker: '+self.veh_id+' decel: '+str(self.a)+' speed: '+str(env.k.vehicle.get_speed(self.veh_id)))
            # Specify that an attack is happening:
            env.k.vehicle.set_malicious(veh_id=self.veh_id,is_malicious=1)
        else:   
            # No attack currently happening:
            self.normal_ACC_accel(env)
            # Check to see if driving near steady-state:
            if(is_passed_warmup):
                self.Check_For_Steady_State()
            # Check to see if need to initiate attack:
            self.Check_Start_Attack(env)
            # Specificy that no attack is being executed:
            env.k.vehicle.set_malicious(veh_id=self.veh_id,is_malicious=0)


        return self.a

    def get_custom_accel(self, this_vel, lead_vel, h):
        """See parent class."""
        return self.a

