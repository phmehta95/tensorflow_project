#Classifier script
#Importing OS and python unzip
import os
import zipfile

#Unzipping training and testing data
local_zip = '/home/pruthvi/Desktop/tensorflow_project/rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/home/pruthvi/Desktop/tensorflow_project/test_data/')
zip_ref.close()

local_zip = '/home/pruthvi/Desktop/tensorflow_project/rps-test-set.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/home/pruthvi/Desktop/tensorflow_project/test_data/')
zip_ref.close()

#Checking number of files in training and testing dataset
rock_dir = os.path.join('test_data/rps/rock')
paper_dir = os.path.join('test_data/rps/paper')
scissors_dir = os.path.join('test_data/rps/scissors')

print('total training rock images:', len(os.listdir(rock_dir)))
print('total training paper images:', len(os.listdir(paper_dir)))
print('total training scissors images:', len(os.listdir(scissors_dir)))

#Printing the names of the first 10 files in the training dataset
rock_files = os.listdir(rock_dir)
print(rock_files[:10])

paper_files = os.listdir(paper_dir)
print(paper_files[:10])

scissors_files = os.listdir(scissors_dir)
print(scissors_files[:10])

#Iterate over the training images

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

pic_index = 2


next_rock = [os.path.join(rock_dir, fname) 
                for fname in rock_files[pic_index-2:pic_index]]
next_paper = [os.path.join(paper_dir, fname) 
                for fname in paper_files[pic_index-2:pic_index]]
next_scissors = [os.path.join(scissors_dir, fname) 
                for fname in scissors_files[pic_index-2:pic_index]]

print('ROCK NAME',next_rock)
print('PAPER NAME',next_paper)
print('SCISSORS NAME',next_scissors)
print(next_rock+next_paper+next_scissors)
for i, img_path in enumerate(next_rock+next_paper+next_scissors):
  print(i,img_path)
  img = mpimg.imread(img_path)
  plt.imshow(img)
  plt.axis('Off')
  plt.show()


