#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# E. Culurciello
# August 2017

from __future__ import division
from __future__ import print_function
from vizdoom import *
import itertools as it
from random import sample, randint, random
from time import time, sleep
import numpy as np
import skimage.color, skimage.transform
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from tqdm import trange
import time as gimmetime
import os
import sys
import cv2
from skimage import color
from skimage import io
import math
import statistics

# Q-learning settings
learning_rate = float(sys.argv[2]) #0.00025
discount_factor = float(sys.argv[3]) #0.99
epochs = int(sys.argv[5]) #25
learning_steps_per_epoch = 2000 #2000
replay_memory_size = 10000

# NN learning settings
batch_size = int(sys.argv[4])#64

# Training regime
test_episodes_per_epoch = 20 #100

# Other parameters
frame_repeat = 2
resolution = (30, 45)
#resolution = (30, 45)
episodes_to_watch = 10

# load the COCO class labels our YOLO model was trained on
labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")


# derive the paths to the YOLO weights and model configuration
weightsPath = "yolov3-tiny.weights"
configPath = "yolov3-tiny.cfg"
# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


timestr = gimmetime.strftime("%Y%m%d-%H%M%S") #hacky module named time
log_savefile = "./"+timestr+"/logfile.txt"
os.makedirs(os.path.dirname(log_savefile), exist_ok=True)
f = open(log_savefile, "w")


class Unbuffered:

    def __init__(self, stream):

        self.stream = stream

    def write(self, data):

        self.stream.write(data)
        self.stream.flush()
        f.write(data)    # Write the data of stdout here to a text file as well

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

sys.stdout=Unbuffered(sys.stdout)

print(sys.argv)

save_model = True
load_model = False


model_savefile = "./"+timestr+"/model.pth"

skip_learning = False




# Configuration file path
config_file_path = "./"+str(sys.argv[1])
# config_file_path = "../../scenarios/rocket_basic.cfg"
# config_file_path = "../../scenarios/basic.cfg"

f.write("learning_pytorch\n")

settings = "# Q-learning settings\nlearning_rate = "+str(learning_rate)+"\ndiscount_factor = "+str(discount_factor)+"\nepochs = "+str(epochs)+"\nlearning_steps_per_epoch = "+str(learning_steps_per_epoch)+"\nreplay_memory_size = "+str(replay_memory_size)+"\n# NN learning settings\nbatch_size = "+str(batch_size)+"\n# Training regime\ntest_episodes_per_epoch = "+str(test_episodes_per_epoch)+"\n# Other parameters\nframe_repeat = "+str(frame_repeat)+"\nresolution = "+str(resolution)+"\nepisodes_to_watch = "+str(episodes_to_watch)+"\n"

f.write(settings)

f.write("Config:%s\n" %config_file_path)
with open(config_file_path) as conf:
    lines = conf.readlines()
    
    f.writelines(lines)




# Sleep time between actions in ms
sleep_time = 28


# Prepare some colors and drawing function
# Colors in in BGR order
doom_red_color = [0, 0, 203]
doom_blue_color = [203, 0, 0]

def draw_bounding_box(buffer, x, y, width, height, color):
    for i in range(width):
        buffer[y, x + i, :] = color
        buffer[y + height, x + i, :] = color

    for i in range(height):
        buffer[y + i, x, :] = color
        buffer[y + i, x + width, :] = color

def color_labels(labels):
    """
    Walls are blue, floor/ceiling are red (OpenCV uses BGR).
    """
    tmp = np.stack([labels] * 3, -1)
    tmp[labels == 0] = [255, 0, 0]
    tmp[labels == 1] = [0, 0, 255]

    return tmp
# Converts and down-samples the input image
def preprocess(img):
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    return img


class ReplayMemory:
    def __init__(self, capacity):
        channels = 1
        state_shape = (capacity, channels, resolution[0], resolution[1])
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward):

        self.s1[self.pos, 0, :, :, ] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, 0, :, :] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]


class Net(nn.Module):
    def __init__(self, available_actions_count):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=6, stride=3)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(192, 128)
        self.fc2 = nn.Linear(128, available_actions_count)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 192)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

criterion = nn.MSELoss()


def learn(s1, target_q):
    s1 = torch.from_numpy(s1)
    target_q = torch.from_numpy(target_q)
    s1, target_q = Variable(s1), Variable(target_q)
    output = model(s1)
    loss = criterion(output, target_q)
    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def get_q_values(state):
    state = torch.from_numpy(state)
    state = Variable(state)
    return model(state)

def get_best_action(state):
    q = get_q_values(state)
    m, index = torch.max(q, 1)
    action = index.data.numpy()[0]
    return action


def learn_from_memory():
    """ Learns from a single transition (making use of replay memory).
    s2 is ignored if s2_isterminal """

    # Get a random minibatch from the replay memory and learns from it.
    if memory.size > batch_size:
        s1, a, s2, isterminal, r = memory.get_sample(batch_size)

        q = get_q_values(s2).data.numpy()
        q2 = np.max(q, axis=1)
        target_q = get_q_values(s1).data.numpy()
        # target differs from q only for the selected action. The following means:
        # target_Q(s,a) = r + gamma * max Q(s2,_) if isterminal else r
        target_q[np.arange(target_q.shape[0]), a] = r + discount_factor * (1 - isterminal) * q2
        learn(s1, target_q)

def autoAim(lx):
    #[[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]

    #print(lx)
    #if lx < 310 and lx > 270: 
    #    a = 6
    #elif lx > 330 and lx < 370: 
    #    a = 3 

    if lx < 300:
        a = 3
    #elif lx == 276:
    #    a = 3      
    elif lx > 340:
        a = 2
    else:
        a = 1
    #print(lx,a)
    return a

def autoAim2(lx):
    #[[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0]]
    #print(lx)
    if lx < 310 and lx > 270: 
        a = 6
    elif lx > 330 and lx < 370: 
        a = 3 
    if lx > 350:
        a = 2
    elif lx == 276:
        a = 3      
    elif lx < 290:
        a = 4
    else:
        a = 1
    #print(a)
    return a

def autoAimWalk(lx):
    if lx < 305 and lx > 270: 
        a = 4
    elif lx > 335 and lx < 370: 
        a = 5 
    if lx > 350:
        a = 2
    elif lx == 276:
        a = 5      
    elif lx < 290:
        a = 1
    else:
        a = 3
    return a

global counter
last10 = []

def circleReward(state):
    angle = game.get_game_variable(GameVariable.ANGLE)
    last10.append(angle)
    flag1 = False
    flag2 = False 
    flag3 = False 
    flag4 = False  
    if len(last10) > 100:
         last10.pop(0)
    #print(last10)
    for i in last10:
        if i >= 0 and i <= 90:
            flag1 = True
        elif i > 90 and i <= 180:
            flag2 = True
        elif i > 180 and i <= 270:
            flag3 = True
        elif i > 270 and i <= 359:
            flag4 = True

    if flag1 == True and flag2 == True and flag3 == True and flag4 == True:
        return .01

    if flag1 == True and flag2 == True and flag3 == True :
        return .005

    if flag1 == True and flag2 == True and flag4 == True:
        return .005

    if flag1 == True and flag3 == True and flag4 == True:
        return .005

    if flag2 == True and flag3 == True and flag4 == True:
        return .005
    else:
        return 0

def basicLidarReward(state):
    depthFlag = False
    longFlag = False
    leftFlag = False
    rightFlag = False
    depth = state.depth_buffer
    if depth is not None:
        depth = np.asarray(depth)
        np.set_printoptions(threshold=sys.maxsize)

        centerd= depth[240,320]
        #print(depth) # if you have the cursor on, you wind up giving a depth of zero for the center... duh.
        centerdx= depth[240,0:640]
        leftdx= statistics.mean(depth[240,0:320])
        rightdx= statistics.mean(depth[240,320:640])
        averagedx = statistics.mean(centerdx)
        #print(centerd, averagedx)      
        if centerd*1.1 <= averagedx and centerd <= 50:


            depthFlag = True

        """if centerd <= 100:

            longFlag = True
            print("long")
        if leftdx*1.2 < rightdx:
            leftFlag = True
            print("left")
        if rightdx*1.2 < leftdx:
            rightFlag = True
            print("right")"""

    return depthFlag, longFlag, leftFlag, rightFlag 
        

def yoloFunction(image):

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    lxArray = []
    boxes = []
    confidences = []
    classIDs = []

    (H, W) = image.shape[:2]
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = gimmetime.time()
    layerOutputs = net.forward(ln)

    end = gimmetime.time()
    # show timing information on YOLO
    #print("[INFO] YOLO took {:.6f} seconds".format(end - start))


    # loop over each of the layer outputs
    for output in layerOutputs:
	    # loop over each of the detections
	    for detection in output:
		    # extract the class ID and confidence (i.e., probability) of
		    # the current object detection
		    scores = detection[5:]
		    classID = np.argmax(scores)
		    confidence = scores[classID]
		    # filter out weak predictions by ensuring the detected
		    # probability is greater than the minimum probability
		    if confidence > .5: #magic number alert
			    # scale the bounding box coordinates back relative to the
			    # size of the image, keeping in mind that YOLO actually
			    # returns the center (x, y)-coordinates of the bounding
			    # box followed by the boxes' width and height
			    box = detection[0:4] * np.array([W, H, W, H])
			    (centerX, centerY, width, height) = box.astype("int")
			    # use the center (x, y)-coordinates to derive the top and
			    # and left corner of the bounding box
			    x = int(centerX - (width / 2))
			    y = int(centerY - (height / 2))
			    # update our list of bounding box coordinates, confidences,
			    # and class IDs
			    boxes.append([x, y, int(width), int(height)])
			    confidences.append(float(confidence))
			    classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, .5,
	.3) #.3 = threshold, .5 = confidence

    

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            lx = x+(.5*w)
            
		    # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            if LABELS[classIDs[i]] == "person":
                lxArray.append(lx)
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			    0.5, color, 2)
    # show the output image
    #cv2.imshow("Image", image)
    #cv2.waitKey(sleep_time)
    return image, lxArray

def labelFunction(screen, labels):
    lxArray = []
    for l in labels:

        if l.object_name in ["Cacodemon", "DoomImp", "Zombieman","shotgunGuy","MarineChainsawVzd","ChaingunGuy", "Demon","HellKnight"]:
            #print(l.object_name)
            lx = l.x+.5*l.width
            ly = l.y
            #print(lx)
            draw_bounding_box(screen, l.x, l.y, l.width, l.height, doom_red_color)
            lxArray.append(lx)
        elif l.object_name in ["Medkit", "GreenArmor"]:
            draw_bounding_box(screen, l.x, l.y, l.width, l.height, doom_blue_color)
            continue
        else:
            continue
    #cv2.imshow('ViZDoom Label Buffer', screen)
    #cv2.waitKey(sleep_time)
    return screen, lxArray



def perform_learning_step(epoch):
    """ Makes an action according to eps-greedy policy, observes the result
    (next state, reward) and learns from the transition"""

    def exploration_rate(epoch):
        """# Define exploration rate change over time"""
        start_eps = 0.2
        end_eps = 0.1
        const_eps_epochs = 0.1 * epochs  # 10% of learning time
        eps_decay_epochs = 0.6 * epochs  # 60% of learning time

        if epoch < const_eps_epochs:
            return start_eps
        elif epoch < eps_decay_epochs:
            # Linear decay
            return start_eps - (epoch - const_eps_epochs) / \
                               (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else:
            return end_eps



    # Gets the state
    state = game.get_state()

    yoloxArray = []
    yolox = False
    labelxArray = []
    labelx = False
    depthFlag = False
    longFlag = False
    leftFlag = False
    rightFlag = False
    

    # Screen buffer, given in selected format. This buffer is always available.
    # Using information from state.labels draw bounding boxes.
    screen = state.screen_buffer
    image, yoloxArray = yoloFunction(screen)
    

    # Labels buffer, always in 8-bit gray channel format.
    # Shows only visible game objects (enemies, pickups, exploding barrels etc.), each with unique label.
    # Labels data are available in state.labels.

    image, labelxArray = labelFunction(screen, state.labels)

    #if labels is not None:
    #    cv2.imshow('ViZDoom Labels Buffer', color_labels(state.labels_buffer))
    #    cv2.waitKey(sleep_time)

    if labelxArray != []:
        labelx = min(labelxArray, key=lambda x:abs(x-320))

    if yoloxArray != []:
        yolox = min(yoloxArray, key=lambda x:abs(x-320))        

    screen = color.rgb2gray(screen)
    #print(lx,320)

    s1 = preprocess(screen)


    
    depthFlag, longFlag, leftFlag, rightFlag = basicLidarReward(state)

      
        
    #degrees = math.atan2(lx, 320)#*57.2958 #https://en.wikipedia.org/wiki/Atan2 in Radians

    # With probability eps make a random action.
    eps = exploration_rate(epoch)
    if labelx and epoch < 10:    
        a = autoAim(labelx)
    elif yolox and epoch <10:
        a = autoAim(yolox)
    elif depthFlag:
        #nextActions = np.random.randint(3,4, size=1)
        #print(nextActions)
        a = 3 #nextActions[0]
    elif random() <= eps:
        a = randint(0, len(actions) - 1)
        #if a == 1 and epoch < 10:
        #    a = 2
    else:
        #Choose the best action according to the network.
        a = randint(0, len(actions) - 1)
        s1 = s1.reshape([1, 1, resolution[0], resolution[1]])
        #a = get_best_action(s1)
    
    reward = game.make_action(actions[a], frame_repeat)
    #print(reward)

    isterminal = game.is_episode_finished()
    s2 = preprocess(color.rgb2gray(game.get_state().screen_buffer)) if not isterminal else None
    if not isterminal:
        s2 = preprocess(color.rgb2gray(game.get_state().screen_buffer)) 
        #s2 = color.rgb2gray(s2)
    else:
        s2 = None

    # Remember the transition that was just experienced.
    memory.add_transition(s1, a, s2, isterminal, reward)

    learn_from_memory()
   



# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.RGB24)
    game.set_depth_buffer_enabled(True)
    #game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_labels_buffer_enabled(True)
    game.clear_available_game_variables()

    game.init()
    print("Doom initialized.")
    return game

def initialize_vizdoom_test(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(True)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.RGB24)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    #game.set_labels_buffer_enabled(True)
    game.clear_available_game_variables()

    game.init()
    print("Doom initialized.")
    return game

if __name__ == '__main__':
    # Create Doom instance
    game = initialize_vizdoom(config_file_path)


    # Action = which buttons are pressed
    n = game.get_available_buttons_size()
    n = game.get_available_buttons_size()
    newActions = []    
    actions = [list(a) for a in it.product([0, 1], repeat=n)]
    for i in actions:

        if sum(i) <= 1:
            newActions.append(i)
        #elif i[0] == 1 and sum(i) <=2:
        #    newActions.append(i)
        #elif i[1] == 1 and sum(i) <=2:
        #    newActions.append(i)        
    actions = newActions
    print(actions)


    # Create replay memory which will store the transitions
    memory = ReplayMemory(capacity=replay_memory_size)

    if load_model:
        print("Loading model from: ", model_savefile)
        model = torch.load(model_savefile)
    else:
        model = Net(len(actions))
    
    optimizer = torch.optim.SGD(model.parameters(), learning_rate)

    print("Starting the training!")
    time_start = time()
    if not skip_learning:
        for epoch in range(epochs):
            print("\nEpoch %d\n-------" % (epoch + 1))
            #f.write("Epoch:%d\n" %epoch)
            train_episodes_finished = 0
            train_scores = []
            game = initialize_vizdoom(config_file_path)
            print("Training...")
            game.new_episode()
            for learning_step in trange(learning_steps_per_epoch, leave=False):
                perform_learning_step(epoch)
                if game.is_episode_finished():
                    score = game.get_total_reward()
                    train_scores.append(score)
                    game.new_episode()
                    train_episodes_finished += 1

            print("%d training episodes played." % train_episodes_finished)

            train_scores = np.array(train_scores)

            print("Results: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()), \
                  "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())
            b = ''.join(["Results: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()), \
                  "min: %.1f," % train_scores.min(), "max: %.1f,\n" % train_scores.max()])
            #f.write(b)

            print("\nTesting...")
            test_episode = []
            test_scores = []
            game = initialize_vizdoom_test(config_file_path)
            for test_episode in trange(test_episodes_per_epoch, leave=False):
                game.new_episode()
                while not game.is_episode_finished():
                    state = game.get_state()
                    # Labels buffer, always in 8-bit gray channel format.
                    # Shows only visible game objects (enemies, pickups, exploding barrels etc.), each with unique label.
                    # Labels data are available in state.labels.
                    #labels = state.labels_buffer
                    # Screen buffer, given in selected format. This buffer is always available.
                    # Using information from state.labels draw bounding boxes.
                    screen = state.screen_buffer

                    state = preprocess(screen)


                    state = state.reshape([1, 1, resolution[0], resolution[1]])
                    best_action_index = get_best_action(state)

                    game.make_action(actions[best_action_index], frame_repeat)
                r = game.get_total_reward()
                test_scores.append(r)

            test_scores = np.array(test_scores)
            print("Results: mean: %.1f +/- %.1f," % (test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),"max: %.1f" % test_scores.max())
            a = ''.join(["Results: mean: %.1f +/- %.1f," % (test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(), "max: %.1f\n" % test_scores.max()])
                        
            #f.write(a)

            print("Saving the network weights to:", model_savefile)
            torch.save(model, model_savefile)

            print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))
            #f.write(str("Total elapsed time: " +str(((time() - time_start) / 60.0)))+"\n")

    game.close()
    print("======================================")
    print("Training finished. It's time to watch!")

    # Reinitialize the game with window visible
    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()

    for _ in range(episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():

            state = preprocess(game.get_state().screen_buffer)

            state = state.reshape([1, 1, resolution[0], resolution[1]])


            best_action_index = get_best_action(state)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)

    plotGrowth(timestr, 0)
    plotGrowth(timestr, 1)
