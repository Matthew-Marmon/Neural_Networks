import numpy as np
import matplotlib.pyplot as plt
#Matthew Marmon

# no datset
#objective function:
def J(x,w):
    return ((x*w)**4)/4 - 4*((x*2)**3)/3 + 3*((x*w)**2)/2
def dJ(x,w):
    return ((x*w)**3 - 4*(x*w)**2 + 3*x*w)*x
#part 1

x = 1
w = -2
J1 = []
while w <= 5:
    J1.append(J(x, w))
    w += 0.1
plt.plot(np.arange(-2, 5.1, 0.1), J1)
plt.xlabel('w')
plt.ylabel('J(x,w)')
plt.title('Objective Function J when x_1 = 1'  )
plt.grid()
plt.show()

#part 2
eta = 0.1
x = 1
epochs =100
w = [-1, 0.2, 0.9, 4]
for i in range(len(w)):
    J1 = []
    current_w = w[i]  # Reset current_w to the current value for each iteration
    for j in range(epochs):
        J1.append(J(x, current_w))
        current_w -= eta * dJ(x, current_w)
    plt.plot(np.arange(0, epochs), J1, label=f'w={current_w:.2f}')
    plt.xlabel('Epochs')
    plt.ylabel('J(x,w)')
    plt.title('Objective Function J when x_1 = 1 at w={:.2f}'.format(w[i]))
    plt.legend()
    plt.grid()
    plt.show()
    print(f'Final w for w={w[i]:.2f} is {current_w:.2f}')
#part 3 - evaluation of different learning rates

w = 0.2
x = 1
etas = [0.001, 0.01, 1, 5]
epochs = 100

try:
    for eta in etas:
        J1 = []
        current_w = w 
        epoch = 0
        for j in range(epochs):
            J1.append(J(x, current_w))
            current_w -= eta * dJ(x, current_w)
            epoch += 1
        plt.plot(np.arange(0, epochs), J1, label=f'eta={eta}')
        plt.xlabel('Epochs')
        plt.ylabel('J(x,w)')
        plt.title(f'Objective Function J when x_1 = 1 at w={w} with different learning rates')
        plt.legend()
        plt.grid()
        plt.show()
        print(f'Final w for eta={eta} is {current_w:.2f}')
except OverflowError:
    plt.plot(np.arange(0, epoch), J1, label=f'eta={eta}')
    plt.xlabel('Epochs')
    plt.ylabel('J(x,w)')
    plt.title(f'Objective Function J when x_1 = 1 at w={w} with eta={eta} (overflow)')
    plt.legend()
    plt.grid()
    plt.show()
    print(f'Overflow error encountered for eta={eta} with final w={current_w:.2f}')

#part 4 - adaptive learning rate
w = 0.2
eta = 5
p1 = 0.9
p2 = 0.999
x =1
delta = 10**-8
s =0
r =0
epochs =100
J1= []
for i in range(epochs):
    J1.append(J(x, w))
    s = p1 * s + (1 - p1) * dJ(x, w)
    r = p2 * r + (1 - p2) * dJ(x, w)**2
    w -= eta * s / (np.sqrt(r) + delta)
plt.plot(np.arange(0, epochs), J1, label=f'Adaptive Learning Rate')
plt.xlabel('Epochs')
plt.ylabel('J(x,w)')
plt.title(f'Objective Function J with adaptive learning rate')
plt.legend()
plt.grid()
plt.show()


