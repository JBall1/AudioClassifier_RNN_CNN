import pandas as pd
import pathlib
import os
import librosa

wavFiles = pathlib.Path.home() /'audioFiles'
#Change labels 'Scooter' and 'other' and add others as needed.
#sample rate set to 22050 from preprocessing, change as needed and get sample rate with getSampleRate func.
samplerate = '22050'


#gets all file names in dir
def getFilesInDir(directory):
    files = []
    files = os.listdir(directory)
    return files


#Count instances of certian type by file name
def countInst(keyword,arr):
    s = sum(keyword in s for s in arr)
    return s

def getSampleRate(file):
    X,sr = librosa.load(file)
    return sr

filesNames = getFilesInDir(wavFiles)
scooterCounter = countInst('scooter',filesNames)
otherCounter = countInst('other',filesNames)

sample_rate = ['samplerate']*len(filesNames)
label_scooter = ['Scooter']*scooterCounter
label_other = ['Other']*otherCounter
label = label_scooter + label_other
length = [1]*len(filesNames)



df = pd.DataFrame({'fname':filesNames, 'sr':sample_rate, 'label': label, 'length': length})
df.to_csv('metadata.csv',index=True)
