from gym_torcs import TorcsEnv
import numpy as np
import time
from agent import Model
from PIL import Image

img_dim = [64, 64, 3]
n_action = 1        # steer only (float, left and right 1 ~ -1)
steps = 1000        # maximum step for a game
batch_size = 32
n_epoch = 100

images_all = np.zeros((0, img_dim[0], img_dim[1], img_dim[2]))
actions_all = np.zeros((0, n_action))
rewards_all = np.zeros((0,))

img_list = []
action_list = []
reward_list = []

def get_teacher_action(ob):
    """ Compute steer from image for getting data of demonstration """
    steer = ob.angle*10/np.pi
    steer -= ob.trackPos*0.10
    return np.array([steer])

def img_reshape(input_img):
    """ (3, 64, 64) --> (64, 64, 3) """
    _img = np.transpose(input_img, (1, 2, 0))
    _img = np.flipud(_img)
    _img = np.reshape(_img, (img_dim[0], img_dim[1], img_dim[2]))
    return _img


def main():
    env = TorcsEnv(vision=True, throttle=False)
    ob = env.reset(relaunch=True)
    model = Model()

    '''get Date from expert'''
    print('Collecting data from teacher (fake AI) ... ')

    for i in range(steps):
        if i == 0:
            act = np.array([0.0])
        else:
            act = get_teacher_action(ob)
        if i % 100 == 0:
            print("step:", i)
        # if i > 50: # quick stop for quick debug
        #     break
        ob, _, done, _ = env.step(act)
        img_list.append(ob.img)
        action_list.append(act)

    env.end()

    print("#"*50)
    print('saving Expert\'s Data')
    for i in range(len(img_list)):
        print(img_list[i].shape)
        print(type(img_list[i]))
        # Image.imsave('image/teacher/im_{}_{}.png'.format(i, action_list[i]),img_list[i])
        img = Image.fromarray(img_reshape(img_list[i]),'RGB')
        img.save('image/teacher/im_{}_{}.png'.format(i, action_list[i]))

    exit()
    
    # TODO change code like pytorch 

    '''get Date from expert + Agent'''
    for n_episode in range(n_episode):
        ob_list = []
        
        env = TorcsEnv(vision=True, throttle=False)
        ob = env.reset(relaunch=True)
        print("#"*50)
        print("# Episode: %d start" % episode)

        model.eval()
        with torch.no_grad():
            for i in range(steps):
                act_lrnr = model.inf(img_reshape(ob.img))
                act_expt = get_teacher_action(ob)
                act = beta*act_expt + (1-beta)*act_lrnr

                ob, _, done, _ = env.step(act)
                
                if done is True:
                    break
                else:
                    ob_list.append(ob)
            env.end()

        for ob in ob_list:
            images_all = np.concatenate([images_all, img_reshape(ob.img)], axis=0)
            # Dataset AGGregation: bring learner’s and expert’s trajectory distributions
            # closer by labelling additional data points resulting from applying the current policy
            actions_all = np.concatenate([actions_all, np.reshape(get_teacher_action(ob), [1, n_action])], axis=0)

        model.train()
        train(model, data)

if __name__ == "__main__":
    main()