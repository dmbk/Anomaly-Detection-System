import matplotlib.pyplot as plt
import Config
import Model
import DataPreProc
import argparse
import numpy as np

def TEST_get_training_set(conf):
    print("Starting")
    clips =  DataPreProc.get_training_set(conf)
    print("Total clips: %d" % len(clips))
    for clip in clips:
        print("Sequence size: %d" % len(clip))
        for image in clip:
            plt.imshow(np.uint8(image))
            plt.show()



def evaluate(conf):
    model = Model.get_model(conf, True)
    print("got model")
    test = DataPreProc.get_single_test(conf)
    print("got test")
    sz = test.shape[0] - 10
    sequences = np.zeros((sz, 10, 256, 256, 1))
    # apply the sliding window technique to get the sequences
    for i in range(0, sz):
        clip = np.zeros((10, 256, 256, 1))
        for j in range(0, 10):
            clip[j] = test[i + j, :, :, :]
        sequences[i] = clip

    # get the reconstruction cost of all the sequences
    reconstructed_sequences = model.predict(sequences,batch_size=4)
    sequences_reconstruction_cost = np.array([np.linalg.norm(np.subtract(sequences[i],reconstructed_sequences[i])) for i in range(0,sz)])
    sa = (sequences_reconstruction_cost - np.min(sequences_reconstruction_cost)) / np.max(sequences_reconstruction_cost)
    sr = 1.0 - sa

    # plot the regularity scores
    plt.plot(sr)
    plt.ylabel('regularity score Sr(t)')
    plt.xlabel('frame t')
    plt.show()



# Instantiate the parser
parser = argparse.ArgumentParser(description='Vidasa Extended -- Anomaly Detection System')



parser.add_argument('--data',type=str,
                    help='Data directory path')

args = parser.parse_args()

conf = Config.Config(data_dir=args.data)

evaluate(conf)
#TEST_get_training_set(conf)





