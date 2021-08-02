from simanneal import anneal
import numpy as np
import pandas as pd
import math

def inputs():
    a = np.loadtxt('shujuku.csv',delimiter=',',dtype='float32')
    b = pd.DateFrame(a,columns=[i for i in 200])
    c = b.mean()
    return list(c.columns),c.values

def outputs(best_angle):
    x = math.cos(best_angle*np.pai/200)
    y = math.sin(best_angle*np.pai/200)
    return (x,y)

if __name__ == '__main__':
    status,energy = inputs()
    SA = Annealer(initial_state=status[0])
    best_state,best_energy = SA.anneal(energy)
    fianl = outputs(best_state)
    np.savetxt('outcome.csv',np.array([fianl,best_energy]),delimiter='')

