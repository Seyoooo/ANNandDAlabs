from util import *
from rbm import RestrictedBoltzmannMachine


class DeepBeliefNet():

    '''
    For more details : Hinton, Osindero, Teh (2006). A fast learning algorithm for deep belief nets. https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    network          : [top] <---> [pen] ---> [hid] ---> [vis]
                               `-> [lbl]
    lbl : label
    top : top
    pen : penultimate
    hid : hidden
    vis : visible
    '''

    def __init__(self, sizes, image_size, n_labels, batch_size):
        """
        Args:
          sizes: Dictionary of layer names and dimensions
          image_size: Image dimension of data
          n_labels: Number of label categories
          batch_size: Size of mini-batch
        """

        self.rbm_stack = {

            'vis--hid': RestrictedBoltzmannMachine(ndim_visible=sizes["vis"], ndim_hidden=sizes["hid"],
                                                   is_bottom=True, image_size=image_size, batch_size=batch_size),

            'hid--pen': RestrictedBoltzmannMachine(ndim_visible=sizes["hid"], ndim_hidden=sizes["pen"], batch_size=batch_size),

            'pen+lbl--top': RestrictedBoltzmannMachine(ndim_visible=sizes["pen"]+sizes["lbl"], ndim_hidden=sizes["top"],
                                                       is_top=True, n_labels=n_labels, batch_size=batch_size)
        }

        self.sizes = sizes

        self.image_size = image_size

        self.batch_size = batch_size

        self.n_gibbs_recog = 20  # 15 default

        self.n_gibbs_gener = 1000  # 200 default

        self.n_gibbs_wakesleep = 5

        self.print_period = 2000

        return

    def recognize(self, true_img, true_lbl):
        """Recognize/Classify the data into label categories and calculate the accuracy

        Args:
          true_imgs: visible data shaped (number of samples, size of visible layer)
          true_lbl: true labels shaped (number of samples, size of label layer). Used only for calculating accuracy, not driving the net
        """

        n_samples = true_img.shape[0]

        visible_layer = true_img  # visible layer gets the image data
        print(visible_layer.shape)
        print(true_lbl.shape)

        # initialise label units with the value of 0.1
        labels = np.ones(true_lbl.shape) / \
            10.  # start the net by telling you know nothing about labels

        # [TASK 4.2 - Finished]
        # Fix the image data in the visible layer and drive the network bottom to top. In the top RBM, run alternating Gibbs sampling \
        # and read out the labels (replace pass below and 'predicted_lbl' to your predicted labels).
        # NOTE : inferring entire train/test set may require too much compute memory (depends on your system). In that case, divide into mini-batches.

        # getting probs for first hidden layer, which later will be used to sample the second layer
        p_h1 = self.rbm_stack["vis--hid"].get_h_given_v_dir(visible_layer)[0]
        accuracy = np.zeros(self.n_gibbs_recog)
        for i in range(self.n_gibbs_recog):
            print(f"iteration {i+1} of {self.n_gibbs_recog}")

            # sampling second hidden layer
            h_2 = self.rbm_stack['hid--pen'].get_h_given_v_dir(p_h1)[1]

            # adding labels
            h_2_label = np.concatenate((h_2, labels), axis=1)

            # getting values from top layer (probability values)
            out = self.rbm_stack["pen+lbl--top"].get_h_given_v(h_2_label)[1]

            # checking the predicted labels, softmax values
            out_label = self.rbm_stack["pen+lbl--top"].get_v_given_h(out)[0][:, -labels.shape[1]:]

            # adding all the softmax results together. Later, the label with the largest sum from all iterations from all the epochs will win.
            if i == 0:
                pred_lbl = out_label.copy()
            else:
                pred_lbl += out_label.copy()

            acc = 100.*np.mean(np.argmax(pred_lbl, axis=1) == np.argmax(true_lbl, axis=1))
            print("accuracy = {:.2f}".format(acc))
            print()
            accuracy[i] = acc
        # uncomment below for saving mislabelsed digits
        '''
        fig, ax = plt.subplots(3, 4)
        a = 0
        b = 0
        col = 0
        row = 0
        while a < 12:
            print(a)
            if np.argmax(pred_lbl[b, :]) != np.argmax(true_lbl[b, :]):
                ax[row, col].imshow(visible_layer[b, :].reshape([28, 28]), cmap="bwr",
                                    vmin=0, vmax=1)
                ax[row, col].set_title(str(np.argmax(pred_lbl[b, :])), fontsize=16)
                ax[row, col].tick_params(left=False, right=False, labelleft=False,
                                         labelbottom=False, bottom=False)

                a += 1
                if row > 1:
                    row = 0
                    col += 1
                else:
                    row += 1

            b += 1
        fig.tight_layout(pad=1.0)
        plt.savefig(f"missclassified digits")
        '''

        # uncomment below for saving grid showing misclassified digits
        '''
        fig, ax = plt.subplots()

        grid = np.zeros((true_lbl.shape[1], true_lbl.shape[1]))
        for i in range(n_samples):
            pred = np.argmax(pred_lbl[i, :])
            true = np.argmax(true_lbl[i, :])
            if pred != true:
                grid[pred, true] += 1
        grid = 100 * grid / np.sum(np.sum(grid))
        im = ax.imshow(grid, cmap="bwr")
        ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        ax.set_xlabel('True label', fontsize=12)
        ax.set_ylabel('Predicted label', fontsize=12)
        ax.set_title('Percentage of wrongly predicted digits', fontsize=14)
        bar = plt.colorbar(im)

        plt.savefig(f"missclass_grid")
        '''

    def generate(self, true_lbl, name):
        """Generate data from labels

        Args:
          true_lbl: true labels shaped (number of samples, size of label layer)
          name: string used for saving a video of generated visible activations
        """
        train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(
            dim=[28, 28], n_train=1, n_test=3)
        print(train_imgs.shape)
        n_sample = true_lbl.shape[0]

        records = []
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_xticks([])
        ax.set_yticks([])

        labels = true_lbl

        # [TASK 4.2 - Finished]
        # Fix the label in the label layer and run alternating Gibbs sampling in the top RBM. From the top RBM, drive the network \
        # top to the bottom visible layer (replace 'vis' from random to your generated visible layer).

        # initializing by generatying a random image in the bottom layer, and propagating it forward
        # random_vis = np.random.choice([0, 1], self.sizes['vis']).reshape(-1, self.sizes['vis'])
        random_vis = train_imgs

        h_1 = self.rbm_stack["vis--hid"].get_h_given_v_dir(random_vis)[1]
        h_2 = self.rbm_stack["hid--pen"].get_h_given_v_dir(h_1)[1]

        # adding the desired label
        h_2_label = np.concatenate((h_2, labels), axis=1)

        save_every = 50
        for i in range(self.n_gibbs_gener):

            # getting values in from the top layer
            top = self.rbm_stack["pen+lbl--top"].get_h_given_v(h_2_label)[1]

            # getting back values in the penultimate layer
            h_2_label = self.rbm_stack["pen+lbl--top"].get_v_given_h(top)[0]

            # Fix the labels, making sure it always corresponds to the 1-hot encoding
            h_2_label[:, -labels.shape[1]:] = labels[:, :]

            # removing the labes, preparing for propagating down
            h_2_top_to_bottom = h_2_label[:, :-labels.shape[1]]

            # getting the first hidden layer
            h_1_top_to_bottom = self.rbm_stack["hid--pen"].get_v_given_h_dir(h_2_top_to_bottom)[1]

            # getting visible layer
            vis = self.rbm_stack["vis--hid"].get_v_given_h_dir(h_1_top_to_bottom)[1]
            # vis = np.random.rand(n_sample,self.sizes["vis"])

            records.append([ax.imshow(vis.reshape(self.image_size), cmap="bwr",
                                      vmin=0, vmax=1, animated=True, interpolation=None)])
            if int(i/save_every) == i/save_every:
                plt.savefig(f"{np.argmax(true_lbl)}_{int(i)}")

        anim = stitch_video(fig, records).save("%s.generate%d.mp4" % (name, np.argmax(true_lbl)))

        return records

    def train_greedylayerwise(self, vis_trainset, lbl_trainset, n_iterations):
        """
        Greedy layer-wise training by stacking RBMs. This method first tries to load previous saved parameters of the entire RBM stack.
        If not found, learns layer-by-layer (which needs to be completed) .
        Notice that once you stack more layers on top of a RBM, the weights are permanently untwined.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        try:

            self.loadfromfile_rbm(loc="trained_rbm", name="vis--hid")
            self.rbm_stack["vis--hid"].untwine_weights()

            self.loadfromfile_rbm(loc="trained_rbm", name="hid--pen")
            self.rbm_stack["hid--pen"].untwine_weights()

            self.loadfromfile_rbm(loc="trained_rbm", name="pen+lbl--top")

        except IOError:

            # [TASK 4.2 - Finished]
            #  Use CD-1 to train all RBMs greedily

            print("training vis--hid")
            """
            CD-1 training for vis--hid
            """

            # training first hiden layer, and getting the final probability of h1
            self.rbm_stack["vis--hid"].cd1(vis_trainset, n_iterations)
            p_h1 = self.rbm_stack["vis--hid"].get_h_given_v(vis_trainset)[0]

            self.savetofile_rbm(loc="trained_rbm", name="vis--hid")

            print("training hid--pen")
            """
            CD-1 training for hid--pen
            """
            self.rbm_stack["vis--hid"].untwine_weights()

            # training second hiden layer, and getting the final probability of h2
            self.rbm_stack["hid--pen"].cd1(p_h1, n_iterations)
            p_h2 = self.rbm_stack["hid--pen"].get_h_given_v(p_h1)[0]
            self.savetofile_rbm(loc="trained_rbm", name="hid--pen")

            print("training pen+lbl--top")
            """
            CD-1 training for pen+lbl--top
            """
            self.rbm_stack["hid--pen"].untwine_weights()
            # concatenating with labels for final layer training

            p_h2_label = np.concatenate((p_h2, lbl_trainset), axis=1)

            self.rbm_stack["pen+lbl--top"].cd1(p_h2_label, n_iterations)
            self.savetofile_rbm(loc="trained_rbm", name="pen+lbl--top")

        return

    def train_wakesleep_finetune(self, vis_trainset, lbl_trainset, n_iterations):
        """
        Wake-sleep method for learning all the parameters of network.
        First tries to load previous saved parameters of the entire network.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        print("\ntraining wake-sleep..")

        try:

            self.loadfromfile_dbn(loc="trained_dbn", name="vis--hid")
            self.loadfromfile_dbn(loc="trained_dbn", name="hid--pen")
            self.loadfromfile_rbm(loc="trained_dbn", name="pen+lbl--top")

        except IOError:

            self.n_samples = vis_trainset.shape[0]

            for it in range(n_iterations):

                # [TODO TASK 4.3] wake-phase : drive the network bottom to top using fixing the visible and label data.

                # [TODO TASK 4.3] alternating Gibbs sampling in the top RBM for k='n_gibbs_wakesleep' steps, also store neccessary information for learning this RBM.

                # [TODO TASK 4.3] sleep phase : from the activities in the top RBM, drive the network top to bottom.

                # [TODO TASK 4.3] compute predictions : compute generative predictions from wake-phase activations, and recognize predictions from sleep-phase activations.
                # Note that these predictions will not alter the network activations, we use them only to learn the directed connections.

                # [TODO TASK 4.3] update generative parameters : here you will only use 'update_generate_params' method from rbm class.

                # [TODO TASK 4.3] update parameters of top rbm : here you will only use 'update_params' method from rbm class.

                # [TODO TASK 4.3] update generative parameters : here you will only use 'update_recognize_params' method from rbm class.

                if it % self.print_period == 0:
                    print("iteration=%7d" % it)

            self.savetofile_dbn(loc="trained_dbn", name="vis--hid")
            self.savetofile_dbn(loc="trained_dbn", name="hid--pen")
            self.savetofile_rbm(loc="trained_dbn", name="pen+lbl--top")

        return

    def loadfromfile_rbm(self, loc, name):

        self.rbm_stack[name].weight_vh = np.load("%s/rbm.%s.weight_vh.npy" % (loc, name))
        self.rbm_stack[name].bias_v = np.load("%s/rbm.%s.bias_v.npy" % (loc, name))
        self.rbm_stack[name].bias_h = np.load("%s/rbm.%s.bias_h.npy" % (loc, name))
        print("loaded rbm[%s] from %s" % (name, loc))
        return

    def savetofile_rbm(self, loc, name):

        np.save("%s/rbm.%s.weight_vh" % (loc, name), self.rbm_stack[name].weight_vh)
        np.save("%s/rbm.%s.bias_v" % (loc, name),    self.rbm_stack[name].bias_v)
        np.save("%s/rbm.%s.bias_h" % (loc, name),    self.rbm_stack[name].bias_h)
        return

    def loadfromfile_dbn(self, loc, name):

        self.rbm_stack[name].weight_v_to_h = np.load("%s/dbn.%s.weight_v_to_h.npy" % (loc, name))
        self.rbm_stack[name].weight_h_to_v = np.load("%s/dbn.%s.weight_h_to_v.npy" % (loc, name))
        self.rbm_stack[name].bias_v = np.load("%s/dbn.%s.bias_v.npy" % (loc, name))
        self.rbm_stack[name].bias_h = np.load("%s/dbn.%s.bias_h.npy" % (loc, name))
        print("loaded rbm[%s] from %s" % (name, loc))
        return

    def savetofile_dbn(self, loc, name):

        np.save("%s/dbn.%s.weight_v_to_h" % (loc, name), self.rbm_stack[name].weight_v_to_h)
        np.save("%s/dbn.%s.weight_h_to_v" % (loc, name), self.rbm_stack[name].weight_h_to_v)
        np.save("%s/dbn.%s.bias_v" % (loc, name),        self.rbm_stack[name].bias_v)
        np.save("%s/dbn.%s.bias_h" % (loc, name),        self.rbm_stack[name].bias_h)
        return