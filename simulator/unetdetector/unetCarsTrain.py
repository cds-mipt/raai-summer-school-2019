from unetdetector.model import *
from unetdetector.data import *
from keras import callbacks
from keras.callbacks import ModelCheckpoint
from datetime import datetime
import os
import time

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_flag = False
tune_flag = False

image_width = 512
image_height = 512


n_epochs = 30
batch_size = 1
train_sample_len = 400
test_sample_len = 100
train_steps = train_sample_len//batch_size
test_steps = test_sample_len//batch_size
learning_rate = 1e-4

current_dir = os.getcwd()
project_dir = os.path.dirname(current_dir)

train_path = project_dir+'/data/dataset/train'
val_path = project_dir+'/data/dataset/test'
test_path = project_dir+'/data/dataset/test/color'

model_path = project_dir+'/unetdetector/models/unet2019-07-02-22-08-09.29-tloss-0.0171-tdice-0.9829.hdf5' #512


timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
log_filename = current_dir+'/logs/log-unet-'+ timestr +'.txt'
with open(log_filename, "a") as log_file:
    log_file.write("Experiment start: "+ log_filename + "\n")
    log_file.write("Input dataset(s):\n")
    log_file.write("Train path: "+ train_path +"+\n")
    log_file.write("Val path: "+ val_path +"+\n")


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
with open(log_filename, "a") as log_file:
    log_file.write("Augmentation args: "+ str(data_gen_args) + "\n")

trainGene = trainGenerator(batch_size=batch_size,
                        train_path=train_path,
                        image_folder='color',
                        mask_folder='mask',
                        aug_dict=data_gen_args,
                        image_color_mode='rgb',
                        target_size = (image_width,image_height),
                        save_to_dir = None) # project_dir+'/data/dataset/aug')
valGene = trainGenerator(batch_size=1,
                        train_path=val_path,
                        image_folder='color',
                        mask_folder='mask',
                        aug_dict=dict(),
                        image_color_mode='rgb',
                        target_size = (image_width,image_height),
                        save_to_dir = None)
testGene = testGenerator(test_path=test_path,
                         num_image = test_sample_len,
                         target_size = (image_width,image_height))

if not tune_flag:
    if train_flag:
        model_path = None

with open(log_filename, "a") as log_file:
    log_file.write("Model path: "+ str(model_path) + "\n")

model = unet_light(pretrained_weights = model_path, input_size = (image_width,image_width,3), learning_rate = learning_rate)

stringlist = []
model.summary(print_fn=lambda x: stringlist.append(x))
short_model_summary = "\n".join(stringlist)



with open(log_filename, "a") as log_file:
    log_file.write('\nModel summary:\n')
    log_file.write(short_model_summary)
    log_file.write('\nOptimizer:' +str(model.optimizer)+' Learning rate: '+str(learning_rate)+'\n')
    log_file.write('Loss:' + str(model.loss) + '\n')
    log_file.write('Metrics:' + str(model.metrics) + '\n')
    log_file.write('Image_Size: ' + str(image_width) + ' x '+str(image_height) + '\n')
    log_file.write('Batch_Size: ' + str(batch_size) + '\n')
    log_file.write('\nTraining process:\n')
    log_file.write('\nEpoch no, train dice, train loss, val dice, val loss\n')
if train_flag:

    model_checkpoint = ModelCheckpoint(project_dir+'/unetdetector/models/unet'+timestr+
                                       '.{epoch:02d}-tloss-{loss:.4f}-tdice-{dice_coef:.4f}.hdf5',
                                       monitor='loss', verbose=1, save_best_only=True, save_weights_only = True)
    callbacks = [model_checkpoint, callbacks.CSVLogger(log_filename, separator=',', append=True)]

    start_time = time.time()

    model.fit_generator(trainGene,steps_per_epoch=train_steps,epochs=n_epochs,validation_data=valGene, validation_steps=test_sample_len, callbacks=callbacks)

    end_time = time.time()
    duration = end_time - start_time
    with open(log_filename, "a") as log_file:
        log_file.write("Training time, sec: "+ str(duration) + "\n")
else:
    start_time = time.time()
    results = model.predict_generator(testGene,test_sample_len,verbose=1)
    end_time = time.time()
    duration = end_time - start_time
    with open(log_filename, "a") as log_file:
        log_file.write("Testing time, sec: " + str(duration) + "\n")
    saveResult(project_dir+"/data/dataset/results",results)
