#!/usr/bin/env python

######################################################################
# OTHER IMPORTS
######################################################################

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as im 
import random
import math
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float64MultiArray
from std_msgs.msg import UInt32
import colorsys


###########################################################
# OUR STARTUP
###########################################################

# LOG IN TO COMPUTER
# user - ROB301_2
# password - Robots2000!

# ROBOT'S TERMINAL #1 and #2
# ssh ubuntu@100.69.127.124
# password: "turtlebot"

# PC'S TERMINAL #3
# roscore

# ROBOT'S TERMINAL #1
# roslaunch turtlebot3_bringup turtlebot3_robot.launch --screen
# ROBOT'S TERMINAL #2
# roslaunch camera camera.launch

# TERMINAL #4 CHECKS
# roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
# rostopic list

# TERMINAL #4 RUN
# rosrun final_project final_project.py

# /home/rob301_2/catkin_ws/src/rob301_labs/lab4


######################################################################
# HELPER FUNCTIONS
######################################################################

class BayesLoc:

    # initialisation
    def __init__(self):

        # rospy init stuff
        self.colour_sub = rospy.Subscriber("mean_img_rgb", Float64MultiArray, self.colour_callback)
        self.line_sub = rospy.Subscriber("line_idx", UInt32, self.line_callback)
        self.cmd_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
	
        # to define the actual configuration of the layout
        self.offices = {2: "yellow",
                        3: "green",
                        4: "blue",
                        5: "orange",
                        6: "orange",
                        7: "green",
                        8: "blue",
                        9: "orange",
                        10: "yellow",
                        11: "green",
                        12: "blue"}
        self.coloursss = list(self.offices.values())
        
        # defining the measurement model
        #                      b      g     y     o
        measurement_model = [[0.60, 0.20, 0.05, 0.05], # b
                             [0.20, 0.60, 0.05, 0.05], # g
                             [0.05, 0.05, 0.65, 0.20], # y
                             [0.05, 0.05, 0.15, 0.60], # o
                             [0.10, 0.10, 0.10, 0.10]] # nothing
        self.mm = np.array(measurement_model)

        # defining the state model
        #                 -1    0     +1
        state_model = [[0.85, 0.05, 0.05], # one back
                       [0.10, 0.90, 0.10], # stay
                       [0.05, 0.05, 0.85]] # one forward
        self.ss = np.array(state_model)

        # self.colour_map_scheme = [[ 50,  50, 225], # blue
        #                           [  0, 208,   0], # green
        #                           [225, 225,   0], # yellow
        #                           [225,  87,   0]] # orange

        self.colour_map_scheme = [[140,  29, 173], # blue
                                  [ 55,  50, 181], # green
                                  [ 14,  12, 173], # yellow
                                  [170, 183, 224]] # orange

        # other localisation parameters
        self.measuring = [0, 0, 0] # what the colour sensor sees
        self.temp = 0 # index of the black line
        self.line = True # if we start at an office
        self.line_check = [True, True, True]
        self.one_office = False 
        self.probs = [(1/len(self.offices))]*len(self.offices)
        self.where_we_think_were_at = -1 # -1 will be unknown
        self.colour_guess = "nothing"

        # actuation parameters
        self.desired = 320
        self.i_error = 0
        self.d_error = 0
        self.last_error = 0
        self.r = 10 # /sec = Hz

        self.kp = 0.005
        self.ki = 0.00002
        self.kd = 0.002
        self.f_speed = 0.08  # m/s
        self.t_speed = 0.6  # rad/s
       
        # this is for storing the rooms we have to go to and knowing when to stop
        self.index = 0
        self.destinations = [3, 7, 5] # given as room numbers in a order to visit
        self.init_count = [30, 20, 30] # how many steps of turning one way, wait time, turning other way
        self.count = self.init_count

        return 
    
    #########################################################################################################################
    # IMPORTANT FUNCTIONS BENEATH
    #########################################################################################################################

    # get information from the colour camera which should spit and rgb list of 3
    # callback function that receives the most recent colour measurement from the camera
    def colour_callback(self, msg):
        self.measuring = np.array(msg.data) #[r,g,b]
        self.guess_colour()
        return


    # get information from the normal camera and each time check if the line is still there
    def line_callback(self, msg):
        self.temp = msg.data
    
        # check if the line is there or not and set self.line
        black_line_index = self.temp
        if black_line_index > 540 or black_line_index < 100: 
            cur_line = False
        else:
            cur_line = True

        # appends and pops
        self.line_check.append(cur_line)
        self.line_check.pop(0)

        if sum(self.line_check) == 3:
            self.line = True
        elif sum(self.line_check) == 0:
            self.line = False

        # if self.colour_guess == 4:
        #     self.line = True
        # else:
        #     self.line = False    
        
        return


    # we execute this once we're at an office and don't get to until we reach another office
    def at_an_office(self):
        a = 5
        # so let's assume we made it to an office
        # we can take measurements until we're sure what colour we're on
        g = np.array([0, 0, 0, 0])
        while g[0] < a and g[1] < a and g[2] < a and g[3] < a:
            ind = self.guess_colour()
            if ind != 4:
                g[ind] += 1

        # exiting that loop there should be one index that pretty high
        colour = np.argmax(g)

        # then we do a step of bayesian localisation
        # but like really only the update step because we know the robot will move forwards?
        u = self.moving_assignment(1) # it's moving forwards one office
        z = colour # already an index
        self.one_pred_update(u, z)

        # now we can show our results for this round
        # print("A POSTIRI", probs_apostiri)
        self.states_printer()

        return


    # this execute one forward time step of bayesian localisation
    def one_pred_update(self, u, z):
        probs = self.probs
        coloursss = self.coloursss

        # STATE PREDICTION
        probs_apriori = [0]*len(probs)

        for i in range(len(probs)):
            split = probs[i]
            for t in range(-1, 2, 1):
                if (t+i)>=len(probs):
                    val = self.ss[t+1][u]
                    probs_apriori[t+i-len(probs)] += val*split
                else:
                    val = self.ss[t+1][u]
                    probs_apriori[t+i] += val*split

        summ = sum(probs_apriori)
        for i in range(len(probs_apriori)):
            probs_apriori[i] = probs_apriori[i]/summ
            probs_apriori[i] = float(f'{probs_apriori[i]:.4f}')

        # STATE UPDATE
        probs_apostiri = [0]*len(probs_apriori)

        for i in range(len(probs_apriori)):
            split = probs_apriori[i]
            mycol = self.colour_assignment(coloursss[i])
            multiplier = self.mm[z][mycol]
            probs_apostiri[i] = probs_apriori[i]*multiplier

        summ = sum(probs_apostiri)
        for i in range(len(probs_apostiri)):
            probs_apostiri[i] = probs_apostiri[i]/summ
            probs_apostiri[i] = float(f'{probs_apostiri[i]:.4f}')
    
        self.probs = probs_apostiri

        return

    
    # function decides when we've reached and office and after reaching office
    # when we're back on the path and gonna be at the next office
    def walking_around(self):
        rospy.loginfo("RGB " + str(self.measuring))
        # rospy.loginfo("On line?" + str(self.line))
        # rospy.loginfo("Line index " + str(self.temp))
        # rospy.loginfo("Colour guess? "+str(self.colour_guess))

        if self.line:
            rospy.loginfo("On line")
        ind = self.colour_guess
        # the rest is just for debugging and looking
        if self.line == False:
            if ind == 0:
                rospy.loginfo("guess: BLUE")
            elif ind == 1:
                rospy.loginfo("guess: GREEN")
            elif ind == 2:
                rospy.loginfo("guess: YELLOW")
            elif ind == 3:
                rospy.loginfo("guess: ORANGE")
            else:
                ind = 4
                rospy.loginfo("guess: NOTHING")
        
        # rospy.loginfo(self.line_check)


        # if self.line == False:
        #     # we could have just reached it
        #     if self.one_office == False:
        #         self.at_an_office()
        #         self.one_office = True
        #         self.decide_if_at_a_room()
        #     # but we could've already been at an office, in which case we're not updating again
        #     else: 
        #         pass
        # # we're still on the path, so not at an office
        # elif self.line == True:
        #     self.one_office = False
        
        self.good_pid_control()
        return


    # as we walk around we can also see if depending on the prob vectors we feel confident enough
    # to definitively say we're somewhere
    def decide_if_at_a_room(self):
        probs = self.probs
        best_guess_ind = np.argmax(np.array(probs))
        room_nums = list(self.offices.keys())

        if probs[best_guess_ind] >= 0.5:
            loc = room_nums[best_guess_ind]
            print("WHERE WE'RE AT : {}".format(loc))
            self.where_we_think_were_at = loc
        else:
            print("WHERE WE'RE AT : not sure!")
            self.where_we_think_were_at = -1


    # does a step of pid control forward
    def good_pid_control(self, ):

        twist = Twist()



        # we should drive straight if we don't see the line
        if self.line == False:
            twist.linear.x = self.f_speed
            twist.angular.z = 0
            self.i_error = 0
            self.d_error = 0
            self.last_error = 0

        elif sum(self.line_check) <= 2:
            actual = self.temp
            error = self.desired - actual
            self.i_error += error
            self.d_error = error - self.last_error
            twist.linear.x = self.f_speed
            twist.angular.z = 0.05 * (self.ki*self.i_error + self.kp*error + self.kd*self.d_error)*self.t_speed
            self.last_error = error

        # if we see the line, we can turn and do pid
        elif self.line == True:
            actual = self.temp
            error = self.desired - actual
            self.i_error += error
            self.d_error = error - self.last_error
            twist.linear.x = self.f_speed
            twist.angular.z = (self.ki*self.i_error + self.kp*error + self.kd*self.d_error)*self.t_speed
            self.last_error = error
        # # we also wanna check if we've reached the next destination 
        # # because if yes, we'll need to stop, turn, and pretend deliver mail
        # if self.where_we_think_were_at == self.destinations[self.index]:
            
        #     # turn one way
        #     if self.count[0] > 0:
        #         twist.linear.x = 0
        #         twist.angular.z = self.t_speed
        #         self.count[0] -= 1
            
        #     # wait for a bit
        #     elif self.count[1] > 0:
        #         twist.linear.x = 0
        #         twist.angular.z = 0
        #         self.count[1] -= 1
            
        #     # turn backwards
        #     elif self.count[2] > 0:
        #         twist.linear.x = 0
        #         twist.angular.z = -1*self.t_speed
        #         self.count[2] -= 1

        #     # you're done the delivery, set sights on the next one
        #     # reset all count things
        #     else:
        #         self.index += 1
        #         self.count = self.init_count
        
        # after deciding what the right command is
        self.cmd_pub.publish(twist)
        return



    #########################################################################################################################
    # HELPER FUNCTIONS BENEATH
    #########################################################################################################################

    # shows a pixel array in grayscale
    def printer(self, what):
        # print("showing picture...")
        plt.imshow(what, cmap=plt.cm.gray)
        plt.pause(0.5)
        plt.close()
        return

    # plots a current step to see all probability distribution
    def states_printer(self):
        room_nums = list(self.offices.keys())
        colours = list(self.offices.values())
        plt.bar(room_nums, height = self.probs, color = colours)
        plt.ylim(0, 1)
        plt.title("CURRENT STATE - APOSTIRI")
        # plt.show()
        plt.pause(3)
        plt.close()
        return

    # switches the measured colours to array indices in mm
    def colour_assignment(self, colour):
        if colour == "blue":
            return 0
        elif colour == "green":
            return 1
        elif colour == "yellow":
            return 2
        elif colour == "orange":
            return 3
        elif colour == "nothing":
            return 4
        return False

    # switches the input to array indices in ss
    def moving_assignment(self, input):
        if input == -1000:
            return 1
        return input+1

    # figures how "close" two rgb values are based on tolerances
    def close(self, x, y, tol):
        x_sum = sum(x)
        y_sum = sum(y)
        # yy = [y[0]*x_sum/y_sum, y[1]*x_sum/y_sum, y[2]*x_sum/y_sum]
        # val = [abs(x[0]-yy[0]), abs(x[1]-yy[1]), abs(x[2]-yy[2])]
        val = [abs(x[0]-y[0]), abs(x[1]-y[1]), abs(x[2]-y[2])]
        summ = sum(val)
        if val[0] < tol:
            if val[1] < tol:
                if val[2] < tol:
                    return [summ, True]
        # if summ < tol:
        #     return [summ, True]
        return [summ, False]

    # now we need a function to look at an rgb value 
    # and if it's not close enough to anything we return "nothing"
    # and ideally it's close to closing or its the floor...
    def guess_colour(self):

        rgb = self.measuring
        colours = self.colour_map_scheme

        # checks the distances to every colour using tolerance
        tol = 30
        vals = []
        ok = []
        for colour in colours:
            out = self.close(rgb, colour, tol)
            vals.append(out[0])
            ok.append(out[1])
        
        # now we go through and find the True with the smallest sum
        smallest_sum = 1000
        ind = -1
        for i in range(len(colours)):
            if ok[i] == True and vals[i] < smallest_sum:
                smallest_sum = vals[i]
                ind = i
        # nothing case
        if ind == -1:
            ind = 4
        
        # # the rest is just for debugging and looking
        # if ind == 0:
        #     print("guess: BLUE")
        # elif ind == 1:
        #     print("guess: GREEN")
        # elif ind == 2:
        #     print("guess: YELLOW")
        # elif ind == 3:
        #     print("guess: ORANGE")
        # else:
        #     ind = 4
        #     print("guess: NOTHING")

        # full = np.array([[colours[0], colours[1], colours[2], colours[3], rgb]])
        # self.printer(full)

        self.colour_guess = ind

        return ind

    



###########################################################################
# SCRIPT
###########################################################################
print("************************************************\n\n")

if __name__ == "__main__":

    # initalise our robot node
    robot = BayesLoc()
    rospy.init_node("final_project")
    rospy.sleep(0.5)
    rate = rospy.Rate(20)

    while not rospy.is_shutdown():
        robot.walking_around()
        rate.sleep()

    rospy.loginfo("finished!")


###########################################################################
# CLEAN UP + HIDDEN
###########################################################################
print("\n\n************************************************")
