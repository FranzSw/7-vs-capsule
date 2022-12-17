import matplotlib.pyplot as plt
import numpy as np
from datetime import date, datetime
plt.ion()
plt.show()

def addPlot(x,y, marker=".", linestyle='None', label=None):
    plt.plot(x, y, label=label, marker=marker, linestyle=linestyle)


def addLinePlot(x,y, label=None):
    addPlot(x,y, marker=".", linestyle='-', label=label)

def updatePlot():
    plt.legend()
    
    plt.draw()
    plt.pause(0.001)

def refreshPlot(data: list[tuple[np.ndarray, np.ndarray, str]]):
    clearPlot()

    for x,y, label in data:
        addPlot(x,y, label=label)
        
    updatePlot()


def clearPlot():
    plt.clf()

def save():
    plt.savefig(str(datetime.now().strftime("%Y-%m-%d %H_%M_%S"))+'.png')