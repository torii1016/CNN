import argparse
import sys
import numpy as np
from scipy.io import loadmat
import cv2

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

from model.convolutional_neural_network import ConvolutionalNeuralNetwork

from show_data import ShowData

def create_svhn_dataset(path):
    file = open(path, 'rb')
    data = loadmat(file)
    imgs = data['X']
    labels = data['y'].flatten()
    labels[labels == 10] = 0
    labels = (np.arange(10) == labels[:, None]).astype(np.float32)
    img_array = convert_imgs_to_array(imgs)
    file.close()
    return img_array, labels

def convert_imgs_to_array(img_array):
    rows = img_array.shape[0]
    cols = img_array.shape[1]
    chans = img_array.shape[2]
    num_imgs = img_array.shape[3]
    scalar = 1 / 255
    new_array = np.empty(shape=(num_imgs, rows, cols, chans), dtype=np.float32)
    for x in range(0, num_imgs):
        chans = img_array[:, :, :, x]
        #norm_vec = chans
        norm_vec = (255-chans)*1.0/255.0
        #norm_vec -= np.mean(norm_vec, axis=0)
        new_array[x] = norm_vec
    return new_array

def color_resize(dataset):

    output_dataset = np.empty(shape=(dataset.shape[0], 32, 32, 3), dtype=np.float32)
    for i in range(output_dataset.shape[0]):
        data = dataset[i].reshape(28,28)
        tmp = np.zeros([28,28,3])
        tmp[:,:,0] = data
        tmp[:,:,1] = data
        tmp[:,:,2] = data

        tmp = cv2.resize(tmp, (32,32))
        output_dataset[i,:,:,:] = tmp
    return output_dataset

def main():

    svhn_train_data, svhn_train_label = create_svhn_dataset("SVHN/train_32x32.mat")
    svhn_pre_train_data = svhn_train_data[:60000]
    svhn_pre_train_label = svhn_train_label[:60000]
    svhn_fine_tuning_data = svhn_train_data[svhn_train_data.shape[0]-10000:]
    svhn_fine_tuning_label = svhn_train_label[svhn_train_data.shape[0]-10000:]
    svhn_test_data, svhn_test_label = create_svhn_dataset("SVHN/test_32x32.mat")
    svhn_pre_test_data = svhn_train_data[6000:svhn_train_data.shape[0]-10000]
    svhn_pre_test_label = svhn_train_label[6000:svhn_train_data.shape[0]-10000]

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    mnist_train_data = np.concatenate([mnist.train.images, mnist.validation.images])
    mnist_train_label = np.concatenate([mnist.train.labels, mnist.validation.labels])

    mnist_train_data = color_resize(mnist_train_data)
    mnist_test_data = color_resize(mnist.test.images)

    cnn = ConvolutionalNeuralNetwork(32, 1000)
    cnn.set_model(0.001)

    show_data = ShowData()

    sess = tf.Session()
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess.run(init)
    saver.restore(sess, "model_m.dump")

    save_model = "model_m_s.dump"
    save_accuracy = "accuracy_curve_finetuning_mnist.png"
    save_loss = "loss_curve_finetuning_mnist.png"
    train_data = svhn_fine_tuning_data
    train_label = svhn_fine_tuning_label
    test_data = svhn_test_data
    test_label = svhn_test_label

    epoch = 500

    # Pre-training
    accuracy_list = []
    loss_list = []
    for i in range(epoch):
        choice_id = np.random.choice(train_data.shape[0], 100, replace=False)
        batch_data = train_data[choice_id]
        batch_label = train_label[choice_id]

        if i % 100 == 0:
            accuracy = 0
            for j in range(0, test_data.shape[0], 100):
                data = test_data[j:j+100]
                label = test_label[j:j+100]
                accuracy += int(cnn.test(sess, data, label)[0]*data.shape[0])
            print("step {}, training accuracy {}".format(i, accuracy/test_data.shape[0]*100.0))

        _, loss = cnn.train(sess, batch_data, batch_label)
        loss_list.append(loss)

        accuracy = 0
        for j in range(0, test_data.shape[0], 100):
            data = test_data[j:j+100]
            label = test_label[j:j+100]
            accuracy += int(cnn.test(sess, data, label)[0]*data.shape[0])
        accuracy_list.append(accuracy/test_data.shape[0]*100.0)

    print("test accuracy {}".format(accuracy/test_data.shape[0]*100.0))

    #show_data.show_accuracy_curve(accuracy_list, save_accuracy)
    #show_data.show_loss_curve(loss_list, save_loss)

    saver.save(sess, save_model)


    saver.restore(sess, "model_s.dump")

    save_model = "model_s_s.dump"
    save_accuracy = "accuracy_curve_finetuning.png"
    save_loss = "loss_curve_finetuning_svhn.png"

    accuracy_list_ = []
    loss_list_ = []
    for i in range(epoch):
        choice_id = np.random.choice(train_data.shape[0], 100, replace=False)
        batch_data = train_data[choice_id]
        batch_label = train_label[choice_id]

        if i % 100 == 0:
            accuracy = 0
            for j in range(0, test_data.shape[0], 100):
                data = test_data[j:j+100]
                label = test_label[j:j+100]
                accuracy += int(cnn.test(sess, data, label)[0]*data.shape[0])
            print("step {}, training accuracy {}".format(i, accuracy/test_data.shape[0]*100.0))

        _, loss = cnn.train(sess, batch_data, batch_label)
        loss_list_.append(loss)

        accuracy = 0
        for j in range(0, test_data.shape[0], 100):
            data = test_data[j:j+100]
            label = test_label[j:j+100]
            accuracy += int(cnn.test(sess, data, label)[0]*data.shape[0])
        accuracy_list_.append(accuracy/test_data.shape[0]*100.0)

    print("test accuracy {}".format(accuracy/test_data.shape[0]*100.0))

    show_data.show_accuracy_curves(accuracy_list, accuracy_list_, save_accuracy)
    #show_data.show_loss_curve(loss_list, save_loss)

    saver.save(sess, save_model)

if __name__ == '__main__':
    main()