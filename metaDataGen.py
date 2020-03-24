import pandas as pd
import pathlib
import os

wavFiles = pathlib.Path.home() /'audioFiles'

set_1 = 0
#Gets all file names in given directory
def getFilesInDir(directory):
   
    files = []
    files = os.listdir(directory)
    s = sum('scooter' in s for s in files)
    print(s)
    
    return files

#File names, sr, labels, length 

fileNames = getFilesInDir(wavFiles)
sample_rate = ['22050']*len(fileNames)
#Scooters, others
label_scooter = ['Scooter']*67
label_other = ['Other']*(len(fileNames)-67)
length = [1]*len(fileNames)

df = pd.DataFrame({'fileName': fileNames, 'sr': sample_rate, 'label': label_other+label_scooter, 'length':length})
#df.set_index('fileName',inplace=True)
#print(df)
df.to_csv('Desktop/Audio Data/a.csv',index=True)
