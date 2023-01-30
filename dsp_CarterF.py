import csv
import os 
import sys
import matplotlib.pyplot as plt 
import numpy as np 

def importData(flg, title): 

    t = [] # column 0
    data1 = [] # column 1
    data2 = [] # column 2

    with open(os.path.join(sys.path[0], title), "r") as f:
        # open the csv file
        reader = csv.reader(f)
        for row in reader:
            # read the rows 1 one by one
            t.append(float(row[0])) # leftmost column
            data1.append(float(row[1])) # second column
            #data2.append(float(row[2])) # third column
    if flg == 1: 
        for i in range(len(t)):
            # print the data to verify it was read
            print(str(t[i]) + ", " + str(data1[i]))
    return t, data1

def calcSampRate(t):
    Fs = len(t)/t[-1]
    tF = t[-1]
    return Fs, tF

def calcMovAvg(t,s,b): 
    # moving average for n samples over signal length of t 
    # buff = np.zeros(10)
    yAvg = np.zeros(len(t))
    for i in range(len(t)): 
        if i < b: 
            yAvg[i] = 0 
        else:     
            # collect last b values
            buff = s[i-b:i]
            yAvg[i] = np.average(buff) 
    return yAvg

def calcIIR(t,s, b, A, B): 
    # moving average for n samples over signal length of t 
    # buff = np.zeros(10)
    yIIR = np.zeros(len(t))
    for i in range(len(t)): 
        if i < b: 
            yIIR[i] = 0 
        else:     
            # collect last b values
            newAvg = A*s[i-b]+B*s[i]
            yIIR[i] = newAvg 
    return yIIR

def calcFIR_WinSinc(s, fH, N, Fs, flg):
    #from __future__ import print_function
    #from __future__ import division

    #import numpy as np

    # Example code, computes the coefficients of a high-pass windowed-sinc filter.

    # Configuration.
    #Fs = 3000 # Sampling rate
    #fH = 0.4  # Cutoff frequency as a fraction of the sampling rate.
    #N = 59  # Filter length, must be odd.

    # Compute sinc filter.
    h = np.sinc(2 * fH/Fs * (np.arange(N) - (N - 1) / 2))

    # Apply window.
    h *= np.blackman(N)

    # Normalize to get unity gain.
    h /= np.sum(h)

    # Create a high-pass filter from the low-pass filter through spectral inversion.
    if flg == 1: 
        h = -h
        h[(N - 1) // 2] += 1

    #print(h)

    # Applying the filter to a signal s can be as simple as writing
    sFIR = np.convolve(s, h, mode='same')
    
    return sFIR, fH, N

    
def runFFT(Fs,t,s):
    #Fs = 10000 # sample rate
    Ts = 1.0/Fs; # sampling interval
    ts = np.arange(0,t[-1],Ts) # time vector
    y = s # the data to make the fft from
    n = len(y) # length of the signal
    k = np.arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(int(n/2))] # one side frequency range
    Y = np.fft.fft(y)/n # fft computing and normalization
    Y = Y[range(int(n/2))]
    return Y, frq

def plotData(t,s,yAvg, title): 
    fig, (ax1, ax2) = plt.subplots(2, 1)   
    fig.tight_layout(pad=3.0)
    ax1.set_title('Raw Signal vs Time')
    ax1.plot(t,s,'b-*')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Signal')
    ax2.set_title('Filtered Signal vs Time')
    ax2.plot(t,yAvg,'b-*')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Signal')
    plt.show()
    path = "./HW2/Plots/"
    nameSansExt = title.split('.')
    figName = nameSansExt[0]
    extName = "_MovAvg.png"
    totalSavePath = path + figName + extName
    fig.savefig(totalSavePath, dpi = 100)

def plotMovAvgData(t,s,yAvg, title, b): 
    fig, ax = plt.subplots()   
    fig.tight_layout(pad=3.0)
    nameSansExt = title.split('.')
    plotTitle = f"{nameSansExt[0]} vs. Moving Average w/ size {b}"
    ax.set_title(plotTitle)
    ax.plot(t,s,'k-*', label="Raw Data")
    ax.plot(t,yAvg,'r', label=f"Moving Average w/ size {b}")
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Signal')
    ax.legend()
    plt.show()
    path = "./HW2/Plots/"
    figName = nameSansExt[0]
    extName = "_MovAvgCombined.png"
    totalSavePath = path + figName + extName
    fig.savefig(totalSavePath, dpi = 100)

def plotIIR(t,s,yIIR, title, A, B): 
    fig, ax = plt.subplots()   
    fig.tight_layout(pad=3.0)
    nameSansExt = title.split('.')
    plotTitle = f"{nameSansExt[0]} vs. IIR Filter w/ A: {A} and B: {B}"
    ax.set_title(plotTitle)
    ax.plot(t,s,'k-*', label="Raw Data")
    ax.plot(t,yIIR,'r', label=f"IIR Filter A: {A} and B: {B}")
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Signal')
    ax.legend()
    plt.show()
    path = "./HW2/Plots/"
    figName = nameSansExt[0]
    extName = "_IIRCombined.png"
    totalSavePath = path + figName + extName
    fig.savefig(totalSavePath, dpi = 100)

def plotFIR(t,s, sFIR, title, fH, N): 
    fig, ax = plt.subplots()   
    fig.tight_layout(pad=3.0)
    nameSansExt = title.split('.')
    plotTitle = f"{nameSansExt[0]} vs. FIR Filter w/ cutoff: {fH} and samples: {N}"
    ax.set_title(plotTitle)
    ax.plot(t,s,'k-*', label="Raw Data")
    ax.plot(t,sFIR,'r', label=f"FIR Filter w/ cutoff: {fH} and samples: {N}")
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Signal')
    ax.legend()
    plt.show()
    path = "./HW2/Plots/"
    figName = nameSansExt[0]
    extName = "_FIRCombined.png"
    totalSavePath = path + figName + extName
    fig.savefig(totalSavePath, dpi = 100)
 
def plotFFT(Y, frq, t, y, title): 
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(t,y,'b')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Amplitude')
    ax2.loglog(frq,abs(Y),'b') # plotting the fft
    ax2.set_xlabel('Freq (Hz)')
    ax2.set_ylabel('|Y(freq)|')
    plt.show()
    path = "./HW2/Plots/"
    nameSansExt = title.split('.')
    figName = nameSansExt[0]
    extName = "_FFT.png"
    totalSavePath = path + figName + extName
    fig.savefig(totalSavePath, dpi = 100)

def plotFFTComb(Y, Y1, frq, frq1, title, b): 
    fig, ax = plt.subplots()   
    fig.tight_layout(pad=3.0)
    nameSansExt = title.split('.')
    plotTitle = f"{nameSansExt[0]} FFT Plot vs. FFT Moving Average w/ size {b}"
    ax.set_title(plotTitle)
    ax.loglog(frq,abs(Y),'k-*', label="Raw Data")
    ax.loglog(frq1,abs(Y1),'r', label=f"Moving Average w/ size {b}")
    ax.set_xlabel('Freq [Hz]')
    ax.set_ylabel('|Y(freq)|')
    ax.legend()
    plt.show()
    path = "./HW2/Plots/"
    figName = nameSansExt[0]
    extName = "_FFTCombined.png"
    totalSavePath = path + figName + extName
    fig.savefig(totalSavePath, dpi = 100)

def plotFFTCombFIR(Y, Y2, frq, frq2, title, Fh, N): 
    fig, ax = plt.subplots()   
    fig.tight_layout(pad=3.0)
    nameSansExt = title.split('.')
    plotTitle = f"{nameSansExt[0]} FFT Plot vs. FFT FIR w/ cutoff {Fh} and samples {N}"
    ax.set_title(plotTitle)
    ax.loglog(frq,abs(Y),'k-*', label="Raw Data")
    ax.loglog(frq2,abs(Y2),'r', label=f"FFT FIR w/ cutoff {Fh} and samples {N}")
    ax.set_xlabel('Freq [Hz]')
    ax.set_ylabel('|Y(freq)|')
    ax.legend()
    plt.show()
    path = "./HW2/Plots/"
    figName = nameSansExt[0]
    extName = "_FFTCombinedFIR.png"
    totalSavePath = path + figName + extName
    fig.savefig(totalSavePath, dpi = 100)


if __name__ == "__main__":     
    sigDat = ["sigA.csv", "sigB.csv", "sigC.csv", "sigD.csv"]
    for i in range(len(sigDat)): 
        # define stuff
        A = .5
        B = .5
        N = 59
        fH = 40
        flg = 0 # 1 for high pass window sinc, 0 for low pass
        

        # calc  stuff 
        t, s = importData(0, sigDat[i])
        Fs, tF = calcSampRate(t)
        yIIR = calcIIR(t,s,1, A, B)
        Y, frq = runFFT(Fs, t, s)
        b = 100*(i+1) # size of moving avg buffer
        yAvg = calcMovAvg(t,s,b)
        Y1, frq1 = runFFT(Fs, t, yAvg)
        sFIR, fH, N = calcFIR_WinSinc(s, fH, N, Fs, flg)
        Y2, frq2 = runFFT(Fs, t, sFIR)
        
        # print stuff 
        print("This is the Sampling Rate for", sigDat[i], ":", Fs)
        print("These are the first timesteps for", sigDat[i], ":", t[0:4])         
        print("This is the final time for", sigDat[i], ":", t[-1])
        print("This is the number of samples for", sigDat[i], ":", len(t))
    
        # plot stuff 
        plotData(t, s, yAvg, sigDat[i])
        plotMovAvgData(t,s, yAvg, sigDat[i], b)
        plotIIR(t,s,yIIR, sigDat[i], A, B)
        plotFIR(t, s, sFIR, sigDat[i], fH, N)
        plotFFT(Y, frq, t,s, sigDat[i])
        plotFFTComb(Y, Y1, frq, frq1, sigDat[i], b)
        plotFFTCombFIR(Y, Y2, frq, frq2, sigDat[i], fH, N)

    
    
    
    