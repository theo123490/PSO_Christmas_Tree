import matplotlib.pyplot as plt
import numpy as np
import cv2

def Resize(image, width=None, height=None, inter=cv2.INTER_AREA, show=False, image_name='image'):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    
    if show==True:
        cv2.imshow(image_name, cv2.resize(image, dim, interpolation=inter))
        cv2.waitKey()
        return

    return cv2.resize(image, dim, interpolation=inter)

def resize_multiple(image_list, image_name_list,width=None, height=None, inter=cv2.INTER_AREA, show=False, image_name='image'):
    if len(image_list) != len(image_name_list):
        raise ValueError('Image_list and image_name_list length are not the same!')

    resize_list = []
    for i in image_list:
        resize_list.append(Resize(i, width=width,height=height))
    
    for i in range(len(resize_list)):
        cv2.imshow(str(image_name_list[i]),resize_list[i])
    cv2.waitKey()

    return


class Particle():
    def __init__(self,image,W=0.5,c1=0.8,c2=0.9):
        self.y_dim,self.x_dim = image.shape

        self.W = W
        self.c1 = c1
        self.c2 = c2 
        self.image = image
        self.position = np.array([int(np.random.rand()*self.x_dim), int(np.random.rand()*self.y_dim)])
        self.pbest_position = self.position
        self.pbest_value = 0
        self.velocity = np.array([0,0])
        self.calc_fitness()

    def __repr__(self):
        return str("I am at {} my pbest is {} \n ".format(self.position, self.pbest_position))

    def move(self):
        self.position = self.position + self.velocity
        self.position = self.position.astype(int)
    
    def calc_fitness(self):
        range_vals=30
        flag = (self.position[0]>=range_vals) and (self.position[1]>=range_vals) and (self.position[0]<=(self.y_dim-range_vals)) and (self.position[1]<=(self.x_dim-range_vals))
        if flag:
            self.fitness = self.image[self.position[0]-range_vals:self.position[0]+range_vals,self.position[1]-range_vals:self.position[1]+range_vals].mean()
        else:
            self.fitness = 0

    def set_velocity(self, gbest_position):
        noise = 50
        new_velocity = (self.W*self.velocity) + (self.c1*np.random.rand()) * (self.pbest_position - self.position) + (np.random.rand()*self.c2) * (gbest_position - self.position) +np.array([int(np.random.rand()*noise),int(np.random.rand()*noise)]) 
        self.velocity = new_velocity

    def iterate(self, gbest_position):
        self.set_velocity(gbest_position)
        self.move()
        self.calc_fitness()        


class World():
    def __init__(self,n_particle,image):
        
        self.particles = []
        self.image = image
        self.gbest_value = 0
        self.gbest_position = [0,0]
        self.gbest_graph = []
        for i in range(n_particle):
            self.particles.append(Particle(image))

        self.set_pbest()
        self.set_gbest()
        self.frames = []


    def set_pbest(self):
        for particle in self.particles:
            if particle.fitness >= particle.pbest_value:
                particle.pbest_value = particle.fitness
                particle.pbest_position = particle.position
    
    def set_gbest(self):
        for particle in self.particles:
            if self.gbest_value <= particle.pbest_value:
                self.gbest_position = particle.pbest_position
                self.gbest_value = particle.pbest_value
    
    def move_particles(self):
        for particle in self.particles:
            particle.iterate(self.gbest_position)

    def iterate_world(self,image):
        self.move_particles()
        self.set_pbest()
        self.set_gbest()
        self.gbest_graph.append(self.gbest_value)
        self.frames.append(self.draw_particles(image))

    def plot_history(self):
        plt.plot(self.gbest_graph)
        plt.show()

    def draw_particles(self,img):
        new_image = img.copy()

        for particle in self.particles:

            new_image = cv2.circle(new_image,tuple(particle.position),5,(255,255,0),-1)

        return new_image        


img = cv2.imread('img3.jpg')
ori_img = img.copy()
img = cv2.blur(img,(60,60))

hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

h,s,v = cv2.split(hsv)
mask = cv2.inRange(h, 60,70)
new_h = h+(255-75)
new_h = new_h+((new_h>255)*-255)
new_h = new_h.astype('uint8')

loss_img = cv2.blur(mask,(60,60))
a=World(800,loss_img)

# for i in range(10):
#     print("first particles fitness {}".format(a.particles[i].fitness))

for i in range(50):
    # print("iteration {}".format(i))
    a.iterate_world(ori_img)

# for i in range(10):
#     print("next particles fitness {}".format(a.particles[i].fitness))

particle_image = a.draw_particles(ori_img)

# a.plot_history()
height, width, layers = ori_img.shape
out = cv2.VideoWriter('project3.avi',cv2.VideoWriter_fourcc(*'DIVX'), 10, (width,height))
 
print("writing video")
for i in range(len(a.frames)):
    out.write(a.frames[i])
out.release()
print("finish writing video")

# cv2.imshow('particles',particle_image)
# cv2.waitKey()


