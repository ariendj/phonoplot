#!/usr/bin/env python3
"""
PLEASE READ
To use this script you need to edit HOME to the directory where the .wav file is.
The .wav file can be any monotonic frequency sweep either up or down in frequency but
it must be trimmed at both ends to remove any leading silence. Frequencies below 1kHz are
ignored since virually all cartridges are well behaved below 1kHz.


The info_line should be alpha-numeric with entries separated by " / " only.  The script
will save a .png file that is named from the info line, replacing " / " with "_".  As
example "this / is / a / test" will create a file named "this_is_a_test.png"

plotstyle =

                1 - traditional
                2 - dual axis (twinx)
                3 - dual plot

Version 15 Alpha


"""

from scipy import signal
from scipy.io.wavfile import read
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
import librosa
import argparse


parser = argparse.ArgumentParser()

# Add arguments for each variable with their default values
parser.add_argument('--file', type=str, default='wavefile.wav')
parser.add_argument('--infoline', type=str, default='Cart / Load / Record')
parser.add_argument('--roundlvl', type=int, default=1)
parser.add_argument('--plotstyle', type=int, default=2)
parser.add_argument('--plotdataout', type=int, default=0)
parser.add_argument('--riaamode', type=int, default=1)
parser.add_argument('--riaainv', type=int, default=0)
parser.add_argument('--str100', type=int, default=0)
parser.add_argument('--normalize', type=int, default=1000)
parser.add_argument('--onekfstart', type=int, default=0)
parser.add_argument('--endf', type=int, default=50000)
parser.add_argument('--ovdylim', type=int, default=0)
parser.add_argument('--ovdylimvalue', type=list, default=[-35,5])
parser.add_argument('--topdb', type=int, default=100)
parser.add_argument('--framelength', type=int, default=512)
parser.add_argument('--hoplength', type=int, default=128)

args = parser.parse_args()

# Access the values of the arguments using their names
_FILE = args.file
infoline = args.infoline
roundlvl = args.roundlvl
plotstyle = args.plotstyle
plotdataout = args.plotdataout
riaamode = args.riaamode
riaainv = args.riaainv
str100 = args.str100
normalize = args.normalize
onekfstart = args.onekfstart
endf = args.endf
ovdylim = args.ovdylim
ovdylimvalue = args.ovdylimvalue
topdb = args.topdb
framelength = args.framelength
hoplength = args.hoplength



def align_yaxis(ax1, ax2):
    y_lims = np.array([ax.get_ylim() for ax in [ax1, ax2]])

    # force 0 to appear on both axes, comment if don't need
    y_lims[:, 0] = y_lims[:, 0].clip(None, 0)
    y_lims[:, 1] = y_lims[:, 1].clip(0, None)

    # normalize both axes
    y_mags = (y_lims[:,1] - y_lims[:,0]).reshape(len(y_lims),1)
    y_lims_normalized = y_lims / y_mags

    # find combined range
    y_new_lims_normalized = np.array([np.min(y_lims_normalized), np.max(y_lims_normalized)])

    # denormalize combined range to get new axes
    new_lim1, new_lim2 = y_new_lims_normalized * y_mags
    return new_lim1, new_lim2



def ft_window(n):       #Matlab's flat top window
    w = []
    a0 = 0.21557895
    a1 = 0.41663158
    a2 = 0.277263158
    a3 = 0.083578947
    a4 = 0.006947368
    pi = np.pi

    for x in range(0,n):
        w.append(a0 - a1*np.cos(2*pi*x/(n-1)) + a2*np.cos(4*pi*x/(n-1)) - a3*np.cos(6*pi*x/(n-1)) + a4*np.cos(8*pi*x/(n-1)))
    return w



def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def createplotdata(insig, Fs):
    fout = []
    aout = []
    foutx = []
    aoutx = []
    fout2 = []
    aout2 = []
    fout3 = []
    aout3 = []


    def interpolate(f, a, minf, maxf, fstep):
        f_out = []
        a_out = []
        amp = 0
        count = 0
        for x in range(minf,(maxf)+1,fstep):
            for y in range(0,len(f)):
                if f[y] == x:
                    amp = amp + a[y]
                    count = count + 1
            if count != 0:
                f_out.append(x)
                a_out.append(20*np.log10(amp/count))
            amp = 0
            count = 0
        return f_out, a_out


    def rfft(insig, Fs, minf, maxf, fstep):
        freq = []
        amp = []
        freqx = []
        ampx = []
        freq2h = []
        amp2h = []
        freq3h = []
        amp3h = []

        F = int(Fs/fstep)
        win = ft_window(F)

        if chinfile == 1:
            for x in range(0,len(insig)-F,F):
                y = abs(np.fft.rfft(insig[x:x+F]*win))
                f = np.argmax(y) #use largest bin
                if f >=minf/fstep and f <=maxf/fstep:
                    freq.append(f*fstep)
                    amp.append(y[f])
                if 2*f<F/2-2 and f > minf/fstep and f < maxf/fstep:
                    f2 = np.argmax(y[(2*f)-2:(2*f)+2])
                    freq2h.append(f*fstep)
                    amp2h.append(y[2*f-2+f2])
                if 3*f<F/2-2 and f > minf/fstep and f < maxf/fstep:
                    f3 = np.argmax(y[(3*f)-2:(3*f)+2])
                    freq3h.append(f*fstep)
                    amp3h.append(y[3*f-2+f3])


        else:
            for x in range(0,len(insig[0])-F,F):
                y0 = abs(np.fft.rfft(insig[0,x:x+F]*win))
                y1 = abs(np.fft.rfft(insig[1,x:x+F]*win))
                f0 = np.argmax(y0) #use largest bin
                f1 = np.argmax(y1) #use largest bin
                if f0 >=minf/fstep and f0 <=maxf/fstep:
                    freq.append(f0*fstep)
                    freqx.append(f1*fstep)
                    amp.append(y0[f0])
                    ampx.append(y1[f1])
                if 2*f0<F/2-2 and f0 > minf/fstep and f0 < maxf/fstep:
                    f2 = np.argmax(y0[(2*f0)-2:(2*f0)+2])
                    freq2h.append(f0*fstep)
                    amp2h.append(y0[2*f0-2+f2])
                if 3*f0<F/2-2 and f0 > minf/fstep and f0 < maxf/fstep:
                    f3 = np.argmax(y0[(3*f0)-2:(3*f0)+2])
                    freq3h.append(f0*fstep)
                    amp3h.append(y0[3*f0-2+f3])

        return freq, amp, freqx, ampx, freq2h, amp2h, freq3h, amp3h



    def normstr100(f, a):
        fmin = 40
        fmax = 500
        slope = -6.02
        for x in range(find_nearest(f, fmin), (find_nearest(f, fmax))):
            a[x] = a[x] + 20*np.log10(1*((f[x])/fmax)**((slope/20)/np.log10(2)))
        return a


    def chunk(insig, Fs, fmin, fmax, step, offset):
        f, a, fx, ax, f2, a2, f3, a3 = rfft(insig, Fs, fmin, fmax, step)
        f, a = interpolate(f, a, fmin, fmax, step)
        fx, ax = interpolate(fx, ax, fmin, fmax, step)
        f2, a2 = interpolate(f2, a2, fmin, fmax, step)
        f3, a3 = interpolate(f3, a3, fmin, fmax, step)
        a = [x - offset for x in a]
        ax = [x - offset for x in ax]
        a2 = [x - offset for x in a2]
        a3 = [x - offset for x in a3]

        return f, a, fx, ax, f2, a2, f3, a3


    def concat(f, a, fx, ax, f2, a2, f3, a3, fout, aout, foutx, aoutx, fout2, aout2, fout3, aout3):
        fout = fout + f
        aout = aout + a
        foutx = foutx + fx
        aoutx = aoutx + ax
        fout2 = fout2 + f2
        aout2 = aout2 + a2
        fout3 = fout3 + f3
        aout3 = aout3 + a3

        return fout, aout, foutx, aoutx, fout2, aout2, fout3, aout3

    if onekfstart == 0:


        f, a, fx, ax, f2, a2, f3, a3 = chunk(insig, Fs, 20, 45, 5, 26.03)
        fout, aout, foutx, aoutx, fout2, aout2, fout3, aout3 = concat(f, a, fx, ax, f2, a2, f3, a3, fout, aout, foutx, aoutx, fout2, aout2, fout3, aout3)

        f, a, fx, ax, f2, a2, f3, a3 = chunk(insig, Fs, 50, 90, 10, 19.995)
        fout, aout, foutx, aoutx, fout2, aout2, fout3, aout3 = concat(f, a, fx, ax, f2, a2, f3, a3, fout, aout, foutx, aoutx, fout2, aout2, fout3, aout3)

        f, a, fx, ax, f2, a2, f3, a3 = chunk(insig, Fs, 100, 980, 20, 13.99)
        fout, aout, foutx, aoutx, fout2, aout2, fout3, aout3 = concat(f, a, fx, ax, f2, a2, f3, a3, fout, aout, foutx, aoutx, fout2, aout2, fout3, aout3)

    f, a, fx, ax, f2, a2, f3, a3 = chunk(insig, Fs, 1000, endf, 100, 0)
    fout, aout, foutx, aoutx, fout2, aout2, fout3, aout3 = concat(f, a, fx, ax, f2, a2, f3, a3, fout, aout, foutx, aoutx, fout2, aout2, fout3, aout3)


    if str100 == 1:
        aout = normstr100(fout, aout)
        aout2 = normstr100(fout2, aout2)
        aout3 = normstr100(fout3, aout3)
        if chinfile == 2:
            aoutx = normstr100(foutx, aoutx)

    i = find_nearest(fout, normalize)
    norm = aout[i]
    aout = aout-norm #amplitude is in dB so normalize by subtraction at [i]
    aoutx = aoutx-norm
    aout2 = aout2-norm
    aout3 = aout3-norm

    sos = signal.iirfilter(3,.5, btype='lowpass', output='sos') #filter some noise
    aout = signal.sosfiltfilt(sos,aout)
    aout2 = signal.sosfiltfilt(sos,aout2)
    aout3 = signal.sosfiltfilt(sos,aout3)

    if chinfile == 2 and len(aoutx) >1:
        aoutx = signal.sosfiltfilt(sos,aoutx)

    return fout, aout, foutx, aoutx, fout2, aout2, fout3, aout3



def ordersignal(sig, Fs):
    F = int(Fs/100)
    win = ft_window(F)

    if chinfile == 1:
        y = abs(np.fft.rfft(sig[0:F]*win))
        minf = np.argmax(y)
        y = abs(np.fft.rfft(sig[len(sig)-F:len(sig)]*win))
        maxf = np.argmax(y)
    else:
        y = abs(np.fft.rfft(sig[0,0:F]*win))
        minf = np.argmax(y)
        y = abs(np.fft.rfft(sig[0][len(sig[0])-F:len(sig[0])]*win))
        maxf = np.argmax(y)

    if maxf < minf:
        maxf,minf = minf,maxf
        sig = np.flipud(sig)

    return sig, minf, maxf



def riaaiir(sig, Fs, mode, inv):
    if Fs == 96000:
        at = [1, -0.66168391, -0.18158841]
        bt = [0.1254979638905360, 0.0458786797031512, 0.0018820452752401]
        ars = [1, -0.60450091, -0.39094593]
        brs = [0.90861261463964900, -0.52293147388301200, -0.34491369168550900]
    if inv == 1:
        at,bt = bt,at
        ars,brs = brs,ars
    if mode == 1:
        sig = signal.lfilter(brs,ars,sig)
    if mode == 2:
        sig = signal.lfilter(bt,at,sig)
    if mode == 3:
        sig = signal.lfilter(bt,at,sig)
        sig = signal.lfilter(brs,ars,sig)
    return sig



def openaudio(_FILE):

    global chinfile
    chinfile = 1

    srinfile = librosa.get_samplerate(_FILE)

    audio, Fs = librosa.load(_FILE, sr=None, mono=False)


    if len(audio.shape) == 2:
        chinfile = 2
        filelength = audio.shape[1] / Fs
    else:
        filelength = audio.shape[0] / Fs

    print('Input File:   ' + str(_FILE))
    print('Sample Rate:  ' + str("{:,}".format(srinfile) + 'Hz'))

    if Fs <96000:
        print('              Resampling to 96,000Hz')
        audio = librosa.resample(audio, orig_sr=Fs, target_sr=96000)
        Fs = 96000

    print('Channels:     ' + str(chinfile))
    print(f"Length:       {filelength}s")

    if riaamode != 0:
        audio = riaaiir(audio, Fs, riaamode, riaainv)


    audio, index = librosa.effects.trim(audio, top_db=topdb, frame_length=framelength, hop_length=hoplength)

    print(f"In/Out (s):   {index / Fs} \n")

    audio, minf, maxf = ordersignal(audio, Fs)

    return audio, Fs, minf, maxf





if __name__ == "__main__":



    input_sig, Fs, minf, maxf = openaudio(_FILE)


    freqout, ampout, freqoutx, ampoutx, freqout2h, ampout2h, freqout3h, ampout3h = createplotdata(input_sig, Fs)




    deltah = round((max(ampout)), roundlvl)
    deltal = abs(round((min(ampout)), roundlvl))


    print('Min Freq:     ' + str("{:,}".format(minf * 100) + 'Hz'))
    print('Max Freq:     ' + str("{:,}".format(maxf * 100) + 'Hz'))

    print('Min Ampl:     ' + str(round((min(ampout)), 4)) + 'dB')
    print('Max Ampl:     ' + str(round((max(ampout)), 4)) + 'dB')
    print('Delta:        ' + str(round((max(ampout) - min(ampout)), 4)) + 'dB')

    print('Frequency:    ' + str(len(freqout)) + ' list elements')
    print('Amplitude:    ' + str(len(ampout)) + ' list elements')
    print('Amplitude 2h: ' + str(len(ampout2h)) + ' list elements')
    print('Amplitude 3h: ' + str(len(ampout3h)) + ' list elements')


    if plotdataout == 1:

        dampout = [*ampout, *[''] * (len(freqout) - len(ampout))]
        dampout2h = [*ampout2h, *[''] * (len(freqout) - len(ampout2h))]
        dampout3h = [*ampout3h, *[''] * (len(freqout) - len(ampout3h))]

        print('\nPlot Data: (freq, ampl, 2h, 3h)\n')

        dataout = list(zip(freqout, dampout, dampout2h, dampout3h))
        for fo, ao, ao2, ao3 in dataout:
            print(fo, ao, ao2, ao3, sep=', ')



    plt.rcParams["xtick.minor.visible"] =  True
    plt.rcParams["ytick.minor.visible"] =  True

    if plotstyle == 1:
        fig, ax1 = plt.subplots(1, 1, figsize=(14,6))

        if ovdylim == 1:
            ax1.set_ylim(*ovdylimvalue)


        ax1.semilogx(freqout,ampout,color = 'r', label = 'Freq Response')

        ax1.semilogx(freqout2h,ampout2h,color = '#ff6666', label = '2nd Harmonic')
        ax1.semilogx(freqout3h,ampout3h,color = '#ffb266', label = '3rd Harmonic')

        ax1.semilogx(freqoutx,ampoutx,color = 'r', linestyle = (0, (3, 1, 1, 1)))


        ax1.set_ylabel("Amplitude (dB)")
        ax1.set_xlabel("Frequency (Hz)")

        ax1.legend(loc=3)


    if plotstyle == 2:
        fig, ax1 = plt.subplots(1, 1, figsize=(14,6))
        ax2 = ax1.twinx()

        if max(ampout) <7:
            ax1.set_ylim(-25, 7)

        if max(ampout) < 4:
            ax1.set_ylim(-25,5)

        if max(ampout) < 2:
            ax1.set_ylim(-29,3)

        if max(ampout) < 0.5:
            ax1.set_ylim(-30,2)

        if ovdylim == 1:
            ax1.set_ylim(*ovdylimvalue)


        ax1.semilogx(freqout,ampout,color = 'b', label = 'Freq Response')

        ax2.semilogx(freqout2h,ampout2h,color = '#6666ff', label = '2nd Harmonic', alpha = 0.6)
        ax2.semilogx(freqout3h,ampout3h,color = '#66b2ff', label = '3rd Harmonic', alpha = 0.6)

        ax1.semilogx(freqoutx,ampoutx,color = 'b', linestyle = (0, (3, 1, 1, 1)), label = 'Crosstalk')


        new_lim1, new_lim2 = align_yaxis(ax1, ax2)
        ax1.set_ylim(new_lim1)
        ax2.set_ylim(new_lim2)

        ax1.set_ylabel("Amplitude (dB)")
        ax2.set_ylabel("Distortion (dB)")
        ax1.set_xlabel("Frequency (Hz)")

        #fig.legend(loc=3, bbox_to_anchor=(0.899, 0.11))
        fig.legend(loc=3, bbox_to_anchor=(0.125, 0.11))

    if plotstyle == 3:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(14,6))

        ax2.grid(True, which="major", axis="both", ls="-", color="black")
        ax2.grid(True, which="minor", axis="both", ls="-", color="gainsboro")

        if max(ampout) <= 1.75 and min(ampout) >= -1.75:
                ax1.set_ylim(-2,2)

        ax1.semilogx(freqout,ampout,color = 'b', label = 'Freq Response')
        ax2.semilogx(freqout2h,ampout2h,color = 'g', label = '2nd Harmonic')
        ax2.semilogx(freqout3h,ampout3h,color = 'r', label = '3rd Harmonic')

        ax1.set_ylabel("Amplitude (dB)")
        ax2.set_ylabel("Distortion (dB)")
        ax2.set_xlabel("Frequency (Hz)")

        ax1.legend(loc=4)
        ax2.legend(loc=4)


    ax1.grid(True, which="major", axis="both", ls="-", color="black")
    ax1.grid(True, which="minor", axis="both", ls="-", color="gainsboro")



    bbox_args = dict(boxstyle="round", color='b', fc='w', ec='b', alpha=1)
    arrow_args = dict(arrowstyle="->")



    ax1.annotate('+' + str(deltah) + ', ' + u"\u2212" + str(deltal) + ' dB', color='b', \
             xy=(freqout[0],(ampout[0]-1)), xycoords='data', \
             xytext=(-15, -20), textcoords='offset points', \
             ha="left", va="center", \
             bbox=bbox_args, \
             #arrowprops=arrow_args \
             )



    ax1.set_xticks([0,20,100,1000,10000,20000,50000])
    ax1.set_xticklabels(['0','20','100','1k','10k','20k','50k'])

    plt.autoscale(enable=True, axis='x')

    ax1.set_title(infoline + "\n", fontsize=16)


    '''
    mod_date = datetime.datetime.fromtimestamp(os.path.getmtime(_FILE))

    plt.figtext(.17, .13, infoline + "\n" + _FILE + "\n" + \
            mod_date.strftime("%b %d, %Y %H:%M:%S"), fontsize=8)
    '''

    plt.savefig(infoline.replace(' / ', '_') +'.png', bbox_inches='tight', pad_inches=.75, dpi=96)

    plt.show()

    print('\nDone!')
