import matplotlib.pyplot as plt
import numpy as np

from reader import read_mp3
from musicprocessor import MusicProcessor

filename = './dataset/Bboy Practice Breaks/Get Lifted - Dj Fleg.mp3'

mp3 = read_mp3(filename)

# Segment
dur = mp3.duration()
fs = mp3.framerate
time = dur/4
segment = mp3.segment(start=time,duration=0.5)

# Implement DFT
mp = MusicProcessor(segment.ys)
mX,pX = mp.DFT()
y = mp.IDFT(mX,pX)

# Plot
plt.figure(figsize=(12,8))

plt.subplot(4,1,1)
plt.plot(time + np.arange(M)/float(fs), segment.ys)
plt.axis([time, time + M/float(fs), min(segment.ys), max(segment.ys)])
plt.ylabel('amplitude')
plt.xlabel('time (sec)')
plt.title('input sound: x')