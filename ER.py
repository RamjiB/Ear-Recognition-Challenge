
import os,shutil
import cv2,keras
import numpy as np
from imgaug import augmenters as iaa
import skimage.io as skio
from keras.applications.xception import Xception
from keras.models import Model

PATH = 'Dataset/'
tr_FN = 'Train Dataset'
trainset = "training_set"
testset = "testing_set"
validation_percentage = 20;
total_subs = 150
epochs = 2

def data_au(x,f):
    flip = iaa.Sequential([iaa.Fliplr(0.5)]).augment_images(x)
    skio.imsave(f + 'flip.png',flip)
    crop = iaa.Sequential([iaa.Crop(percent=(0, 0.1))]).augment_images(x)
    skio.imsave(f + 'crop.png',crop)
    GB = iaa.Sequential([iaa.GaussianBlur(sigma=(0, 3.0))]).augment_images(x)
    skio.imsave(f + 'GB.png',GB)
    GN = iaa.Sequential([iaa.AdditiveGaussianNoise(scale=(0.0, 0.2))]).augment_images(x)
    skio.imsave(f + 'GN.png',GN)
    CN = iaa.Sequential([iaa.ContrastNormalization(0.5, per_channel=0.5)]).augment_images(x)
    skio.imsave(f + 'CN.png',CN)
    BR = iaa.Sequential([iaa.Multiply((0.8,1.2))]).augment_images(x)
    skio.imsave(f + 'BR.png',BR)
    Scale = iaa.Sequential([iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8,1.2)})]).augment_images(x)
    skio.imsave(f + 'Scale.png',Scale)
    rotate = iaa.Sequential([iaa.Affine(rotate=(-45, 45))]).augment_images(x)
    skio.imsave(f + 'rotate.png',rotate)


def createFolder(folderName):
    """
    Safely create folder when needed
    :param folderName : the directory that you  want to safely create
    :return: None
    """
    if not os.path.exists(folderName):
        try:
            os.makedirs(folderName)
        except OSError as exc:  # Guard against race condition
            if exc.errno != exc.errno.EEXIST:
                raise

def split_trianing_testing(images,image_path,subject):
    #take only .png files and store it in a list
    ears = []
    for image in images:
        if (image.endswith(".png")):
            ears.append(image)
        else:
        	file = os.path.join(PATH,trainset,subject,image)
       		shutil.move(os.path.join(image_path,image),file)
    no_trian_data = int(0.6 * len(ears))
    for i,_ in enumerate(ears):
       	if (i <= no_trian_data-1):
       		file = os.path.join(PATH,trainset,subject,ears[i])
       		shutil.move(os.path.join(image_path,ears[i]),file)
       	else:
       		file = os.path.join(PATH,testset,subject,ears[i])
       		shutil.move(os.path.join(image_path,ears[i]),file)

#take 150 subjects from given trianing dataset and split 60-40 from every subject and
#for training and testing purpose
def creating_dataset_without_aug():
	mode = tr_FN
	mode_path = os.path.join(PATH,mode)
	subjects = os.listdir(mode_path)
	for subject in subjects:
		if (int(subject) <= 150):
			createFolder(os.path.join(PATH,trainset,subject))
			createFolder(os.path.join(PATH,testset,subject))
			image_path  = os.path.join(mode_path,subject)
			images = os.listdir(image_path)
			#splitting the trainig and testing in dataset as 60 and 40 percent
			split_trianing_testing(images,image_path,subject)
			if (len(os.listdir(image_path)) == 0):
				os.rmdir(image_path)
	if (len(os.listdir(mode_path)) == 0):
		os.rmdir(mode_path)

#function for data augmnetation
def data_after_augmentation():
	print('data_augmentation starts')
	mode_path = os.path.join(PATH,trainset)
	subjects = os.listdir(mode_path)
	for subject in subjects:
		image_path = os.path.join(mode_path,subject)
		images = os.listdir(image_path)
		for image in images:
			if image.endswith(".png"):
				file = os.path.join(image_path,image)
				img = cv2.resize(cv2.imread(file),(100,100))
				data_au(img,os.path.splitext(file)[0])

#function for collecting final dataset
def data(mode):
	print('train data')
	total_images = []
	label = []
	x_tr = []
	y_tr = []
	x_va = []
	y_va = []
	mode_path = os.path.join(PATH,mode)
	subjects = os.listdir(mode_path)
	for subject in subjects:
		image_path = os.path.join(mode_path,subject)
		images = os.listdir(image_path)
		for image in images:
			if image.endswith(".png"):
				file = os.path.join(image_path,image)
				total_images.append(cv2.resize(cv2.imread(file),(100,100)))
				label.append(int(subject))

		if(mode == trainset):
			total_images = np.array(total_images)
			label = np.array(label)

			[x_t,y_t,x_v,y_v] = split_data(total_images,label,validation_percentage)

			no_training_images = x_t.shape[0]
			no_validation_images = x_v.shape[0]
			x_tr.append(x_t)
			y_tr.append(y_t)
			x_va.append(x_v)
			y_va.append(y_v)
			total_images = []
			label = []

	if (mode == trainset):
		no_subjects = len(subjects)
		x_tr = np.reshape(np.array(x_tr),(no_subjects*no_training_images,100,100,3))
		y_tr = np.reshape(np.array(y_tr),(no_subjects*no_training_images,1))
		x_va = np.reshape(np.array(x_va),(no_subjects*no_validation_images,100,100,3))
		y_va = np.reshape(np.array(y_va),(no_subjects*no_validation_images,1))
		return x_tr,y_tr,x_va,y_va
	else:
		total_images = np.array(total_images)
		label = np.reshape(np.array(label),(len(label),1))
		return total_images,label


#function for splitting training and validation from trainset
def split_data(data,label,valid_len):
    valid_len = int(valid_len*len(data)/100)
    return (data[0:len(data)-valid_len],label[0:len(data)-valid_len],
            data[len(data)-valid_len:len(data)],label[len(data)-valid_len:len(data)])


creating_dataset_without_aug()
print("--------------------------------------------------------")
print('dataset splitted as 60-40 from training set')

data_after_augmentation()
print("--------------------------------------------------------")
print('data augmentation done for splitted train set (60 %)')

[x_train,y_train,x_valid,y_valid] = data(trainset)
print('final trianing dataset collected')
print("x_train: ",x_train.shape)
print("y_train: ",y_train.shape)
print("x_valid: ",x_valid.shape)
print("y_valid",y_valid.shape)


# [x_train,y_train,x_valid,y_valid] = split_data(train,label,validation_percentage)
print ("-------------------------------------------------------")


[x_test,y_test] = data(testset)
print("--------------------------------------------------------")
print('final test data collected')
print (x_test.shape)
print(y_test.shape)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, total_subs+1,dtype='int32')
y_valid = keras.utils.to_categorical(y_valid, total_subs+1,dtype='int32')

y_train = y_train[:,0:total_subs]
y_valid = y_valid[:,0:total_subs]

print(y_train[0])
print(y_valid[0])

#create model and fit the model
base_model = Xception(include_top=False, weights='imagenet',input_shape=x_train[0].shape)

x = base_model.output
x = keras.layers.GlobalAveragePooling2D()(x)
predictions = keras.layers.Dense(total_subs,activation='softmax')(x)

model = Model(base_model.input,predictions)
model.summary()

#for freezing the Xception layers
for layer in base_model.layers:
	layer.trainable = False

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=70,
          epochs=epochs,
          verbose=1,
          validation_data=(x_valid, y_valid))

results = np.argmax(model.predict(x_test),axis = 1)
results = np.reshape(results,(len(results),1))

#accuracy

print("Accuracy: ",(sum(results == y_test))/len(results) * 100)
