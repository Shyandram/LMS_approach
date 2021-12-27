from numpy import abs, sqrt, power
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
def f(pos):
    #return [0.0,pos[1]]
    return [-2/3*pos[0]+4/3, -3/2*pos[1]+4/2]
def g(pos):
    #return [pos[0],pos[1]]
    return [3*pos[0]+5, 1/3*pos[1]-5/3]
def h(pos):
    #return [pos[0],2.0]
    return [1/2*pos[0]+20/2, 2*pos[1]-20]
def get_error(tg_line,pos):
    hw = tg_line
    # dhw = np.array([hw[1]-pos[0],hw[0]-pos[1]])
    dhw = np.array([(hw[1]-pos[0])/abs(hw[1]-pos[0])*abs(hw[0]-pos[1]),(hw[0]-pos[1])/abs((hw[0]-pos[1]))*abs(hw[1]-pos[0])])
    #er = dhw[0]*dhw[1]/(sqrt(np.power(dhw[0],2)+np.power(dhw[1],2)))
    return dhw
def update(pos,ee,lr):
    # if pos[0]>0:
    #     factor_x*=-1
    # if pos[1]<0:
    #     factor_y*=-1
    if ee is not np.array([0.0,0.0]):
        pos -= np.multiply(-lr*2,ee)
    return pos

# def update3(pos,tg_line,lr):
#     x = pos[0]
#     y = pos[1]
#     x_ = x
#     y_ = y
#     gx = 0
#     gy = 0
#     er_x = []
#     er_y = []
#     for i, fun in enumerate(tg_line):
#         hw = fun
#         b = abs(hw[0])
#         a = abs(hw[1])
#         # gx -= (-b**3+3*(b**2)*y-3*b*(y**2)+y**3)/(sqrt((a-x)**2+(b-y)**2)*((a-x)**2+(b-y)**2))
#         # gy += (-a**3+3*(a**2)*x-3*a*(x**2)+x**3)/(sqrt((a-x)**2+(b-y)**2)*((a-x)**2+(b-y)**2))
#         #gx -= (-a*x+x*b-b**2+a*b)*(b-y)/(sqrt(2*x**2-2*a*x-2*x*b+a**2+b**2)*(2*x**2-2*a*x-2*x*b+a**2+b**2))
#         #gy += (-b*y+y*a-a**2+a*b)*(a-x)/(sqrt(2*y**2-2*b*y-2*y*a+a**2+b**2)*(2*y**2-2*b*y-2*y*a+a**2+b**2))
#         # gx += (b-y)*(-1)/sqrt(power((a-x),2)+power((b-y),2))+(a-x)*(b-y)*(-0.5)/power(sqrt((a-x)**2+(b-y)**2),3)
#         # gy += (a-x)*(-1)/sqrt(power((a-x),2)+power((b-y),2))+(a-x)*(b-y)*(-0.5)/power(sqrt((a-x)**2+(b-y)**2),3)
#         gx = -2*(a-x)
#         gy = -2*(b-y)
#         er_x.append(gx)
#         er_y.append(gy)
#     gx1 = np.sum(er_x)/3+0.1*np.sum(er_y)/3
#     gy1 = 0.1*np.sum(er_x)/3+np.sum(er_y)/3
#     x_ -= lr*gx1/3
#     y_ -= lr*gy1/3

    # for i, fun in enumerate(tg_line):
        
    #     hw = fun
    #     b = hw[0]
    #     a = hw[1]
    #     gx += (-b**3+3*(b**2)*y-3*b*(y**2)+y**3)/(sqrt((a-x)**2+(b-y)**2)*((a-x)**2+(b-y)**2))
    #     gy += (-a**3+3*(a**2)*x-3*a*(x**2)+x**3)/(sqrt((a-x)**2+(b-y)**2)*((a-x)**2+(b-y)**2))
    #     # gx = (-a*x+x*b-b**2+a*b)*(b-y)/(sqrt(2*x**2-2*a*x-2*x*b+a**2+b**2)*(2*x**2-2*a*x-2*x*b+a**2+b**2))
    #     # gy = (-b*y+y*a-a**2+a*b)*(a-x)/(sqrt(2*y**2-2*b*y-2*y*a+a**2+b**2)*(2*y**2-2*b*y-2*y*a+a**2+b**2))
    #     x_ += lr*gx
    #     y_ -= lr*gy
    return [x_,y_]

# def update4(pos,tg_line,lr):
#     x = pos[0]
#     y = pos[1]
#     x_ = x
#     y_ = y
#     gx = 0
#     gy = 0
#     hw = tg_line
#     b = hw[0]
#     a = hw[1]
#     gx -= (-b**3+3*(b**2)*y-3*b*(y**2)+y**3)/(sqrt((a-x)**2+(b-y)**2)*((a-x)**2+(b-y)**2))
#     gy -= (-a**3+3*(a**2)*x-3*a*(x**2)+x**3)/(sqrt((a-x)**2+(b-y)**2)*((a-x)**2+(b-y)**2))
#     # gx = (-a*x+x*b-b**2+a*b)*(b-y)/(sqrt(2*x**2-2*a*x-2*x*b+a**2+b**2)*(2*x**2-2*a*x-2*x*b+a**2+b**2))
#     # gy = (-b*y+y*a-a**2+a*b)*(a-x)/(sqrt(2*y**2-2*b*y-2*y*a+a**2+b**2)*(2*y**2-2*b*y-2*y*a+a**2+b**2))
#     x_ -= lr*gx
#     y_ -= lr*gy
#     return [x_,y_]
def get_error2(tg_line):
    hw = tg_line
    return hw
def update2(pos,ee,lr):
    for i in [0,1]:
        pos[i] -= np.multiply(lr*2*ee[1-i],pos[i])
    return pos

df = pd.DataFrame(columns  = ['x','y'])
xxx=[]
yyy=[]
loss_list = []
mu = 0.01
error = 0
itr = 1+100*10
position = [-14,-15]
xxx.append(position[0])
yyy.append(position[1])
plt.annotate('org',xy=position)
print('org_position:\t',position)

plt.plot([-50,50],[f([-50,1])[0],f([50,1])[0]])
plt.plot([-50,50],[g([-50,1])[0],g([50,1])[0]])
plt.plot([h([-50,-50])[1],h([50,50])[1]],[h([-50,1])[0],h([50,1])[0]])
plt.xlim(-20,20)
plt.ylim(-20,20)
#plt.savefig('pr_hw2-3_onlyline.png')
for e in range(itr):
    # for i, fun in enumerate([f(position),g(position),h(position)]):
    #     print('\tfunction ',i+1)
    #     error = get_error(fun)
    #     print('\terror:',error)
    #     position = update(position,error,mu)
    #     print('position:',position)

    error = np.array([0.0,0.0])
    func = [f(position), g(position),h(position),]
    error_list = []
    sq_error = 0
    # ## Method1
    # # func = [f(position),]
    for i, fun in enumerate(func):
        cnt = i+1
        error += get_error(fun,position)
        sq_error += get_error(fun,position) **2
        error_list.append(get_error(fun,position))
    error/=cnt
    position = update(position,error,mu)
    
    # # Method 2
    # func = [f(position),g(position),h(position)]
    # for i, fun in enumerate(func):
    #     position = update2(position,fun,mu)
    
    ##Method 3
    
    # position = update3(position,func,mu)

    # for i, fun in enumerate(func):
    #     cnt = i+1
    #     error += get_error(fun,position)
    # error/=cnt

    # for i, fun in enumerate(func):
    #     cnt = i+1
    #     error += get_error(fun,position)
    #     position = update4(position,fun,mu)
    #     plt.scatter(position[0],position[1],s=5,c='magenta')
    # error/=cnt

    
    # if itr % 300 == 0:
    #     mu /= 10


    df.loc[e,'x'] = position[0]
    df.loc[e,'y'] = position[1]
    if e%10 == 0:
        xxx.append(position[0])
        yyy.append(position[1])
        loss_list.append((np.log((sum(sq_error))/3)))
        plt.scatter(position[0],position[1],s=7,c='black')
        
    if e%100 == 0:
        print('itr:', e+1)
        print('\terror:',sum(error))
        print('\tloss:',(sum(sq_error))/3)
        print('position:',position)
        print('error_list',error_list)
        plt.annotate(str(e+1),xy=position)

plt.plot(xxx,yyy)
plt.grid(True)
# plt.xlim(min(xxx),max(xxx))
# plt.ylim(min(yyy),max(yyy))
plt.savefig('pr_hw2-3.png')
# plt.figure()
# plt.plot(range(0,len(loss_list)*10,10),loss_list)
plt.show()