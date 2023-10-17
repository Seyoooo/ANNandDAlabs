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
            
            'vis--hid' : RestrictedBoltzmannMachine(ndim_visible=sizes["vis"], ndim_hidden=sizes["hid"],
                                                    is_bottom=True, image_size=image_size, batch_size=batch_size),
            
            'hid--pen' : RestrictedBoltzmannMachine(ndim_visible=sizes["hid"], ndim_hidden=sizes["pen"], batch_size=batch_size),
            
            'pen+lbl--top' : RestrictedBoltzmannMachine(ndim_visible=sizes["pen"]+sizes["lbl"], ndim_hidden=sizes["top"],
                                                        is_top=True, n_labels=n_labels, batch_size=batch_size)
        }
        
        self.sizes = sizes

        self.image_size = image_size

        self.batch_size = batch_size
        
        self.n_gibbs_recog = 20
        
        self.n_gibbs_gener = 200
        
        self.n_gibbs_wakesleep = 5

        self.print_period = 2000
        
        return

    def recognize(self,true_img,true_lbl):

        """Recognize/Classify the data into label categories and calculate the accuracy

        Args:
          true_imgs: visible data shaped (number of samples, size of visible layer)
          true_lbl: true labels shaped (number of samples, size of label layer). Used only for calculating accuracy, not driving the net
        """
        
        n_samples = true_img.shape[0]
        
        vis = true_img # visible layer gets the image data
        
        lbl = np.ones(true_lbl.shape)/10. # start the net by telling you know nothing about labels        
        
        # [TODO TASK 4.2] fix the image data in the visible layer and drive the network bottom to top. In the top RBM, run alternating Gibbs sampling \
        # and read out the labels (replace pass below and 'predicted_lbl' to your predicted labels).
        # NOTE : inferring entire train/test set may require too much compute memory (depends on your system). In that case, divide into mini-batches.
        
        # Propagate the visible layer data through the RBMs from bottom to top
        vis = self.rbm_stack["vis--hid"].get_h_given_v_dir(vis)[0]
        vis = self.rbm_stack["hid--pen"].get_h_given_v_dir(vis)[0]
        vis = np.concatenate((vis, lbl), axis=1)

        for i in range(self.n_gibbs_recog):
            # Gibs learning for the top RBM
            h_0_prob, _ = self.rbm_stack["pen+lbl--top"].get_h_given_v(vis)
            if i == self.n_gibbs_recog - 1:
                # We don't take the probabilities!!
                _, vis = self.rbm_stack["pen+lbl--top"].get_v_given_h(h_0_prob)
            else:
                vis, _ = self.rbm_stack["pen+lbl--top"].get_v_given_h(h_0_prob)
        
        predictions = vis[:, -true_lbl.shape[1]:]

        predicted_lbl = np.argmax(predictions,axis=1)


            
        print ("accuracy = %.2f%%"%(100.*np.mean(predicted_lbl==np.argmax(true_lbl,axis=1))))
        
        return predictions

    def generate(self,true_lbl,name):
        
        train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(
            dim=[28, 28], n_train=1, n_test=3)
        print(train_imgs.shape)
        n_sample = true_lbl.shape[0]

        """Generate data from labels

        Args:
          true_lbl: true labels shaped (number of samples, size of label layer)
          name: string used for saving a video of generated visible activations
        """
        
        n_sample = true_lbl.shape[0]
        
        records = []        
        fig,ax = plt.subplots(1,1,figsize=(3,3))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_xticks([])
        ax.set_yticks([])

        labels = true_lbl

        # [TODO TASK 4.2] fix the label in the label layer and run alternating Gibbs sampling in the top RBM. From the top RBM, drive the network \ 
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
            # print(i)
            # getting values in from the top layer
            top = self.rbm_stack["pen+lbl--top"].get_h_given_v(h_2_label)[1]

            # getting back values in the penultimate layer
            h_2_label = self.rbm_stack["pen+lbl--top"].get_v_given_h(top)[0]

            # Fix the labels, making sure it always corresponds to the 1-hot encoding
            h_2_label[:, -labels.shape[1]:] = labels[:, :]

            # removing the labels, preparing for propagating down
            h_2_top_to_bottom = h_2_label[:, :-labels.shape[1]]
            # print(h_2_top_to_bottom.shape)

            # getting the first hidden layer
            h_1_top_to_bottom = self.rbm_stack["hid--pen"].get_v_given_h_dir(h_2_top_to_bottom)[1]
            # print(h_1_top_to_bottom.shape)

            # getting visible layer
            vis = self.rbm_stack["vis--hid"].get_v_given_h_dir(h_1_top_to_bottom)[1]
            # vis = np.random.rand(n_sample,self.sizes["vis"])

            if (i % save_every == 0 or i == self.n_gibbs_gener - 1):
                fig, ax = plt.subplots()
                ax.imshow(vis.reshape(self.image_size), cmap="bwr", vmin=0, vmax=1, interpolation=None)

                # Save the figure with a unique filename
                plt.savefig(f"plts/image_{np.argmax(true_lbl)}_{i}.png")

                # Close the figure to release resources
                plt.close()

        #anim = stitch_video(fig, records).save("%s.generate%d.mp4" % (name, np.argmax(true_lbl)))

        return records
       # I commented it
            
        return

    def train_greedylayerwise(self, vis_trainset, lbl_trainset, n_iterations, n_epochs=5):

        """
        Greedy layer-wise training by stacking RBMs. This method first tries to load previous saved parameters of the entire RBM stack. 
        If not found, learns layer-by-layer (which needs to be completed) .
        Notice that once you stack more layers on top of a RBM, the weights are permanently untwined.

        Args:
        vis_trainset: visible data shaped (size of training set, size of visible layer)
        lbl_trainset: label data shaped (size of training set, size of label layer)
        n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        try :

            self.loadfromfile_rbm(loc="trained_rbm",name="vis--hid")
            self.rbm_stack["vis--hid"].untwine_weights()            
            
            self.loadfromfile_rbm(loc="trained_rbm",name="hid--pen")
            self.rbm_stack["hid--pen"].untwine_weights()
            
            self.loadfromfile_rbm(loc="trained_rbm",name="pen+lbl--top")        

        except IOError :
            # If previously trained parameters are not found, perform greedy layer-wise training

            print("training vis--hid")
            """
            CD-1 training for vis--hid
            """
            self.rbm_stack["vis--hid"].cd1(vis_trainset, n_iterations, n_epochs=n_epochs)

            # Save trained parameters for vis--hid
            self.savetofile_rbm(loc="trained_rbm", name="vis--hid")

            print("training hid--pen")
            """
            CD-1 training for hid--pen
            """
            # Get hidden layer activations from vis--hid for use in hid--pen
            hidden_activations = self.rbm_stack["vis--hid"].get_h_given_v(vis_trainset)[0]
            self.rbm_stack["hid--pen"].cd1(hidden_activations, n_iterations, n_epochs=n_epochs)
 
            # Untwine the weights for the next layer
            self.rbm_stack["vis--hid"].untwine_weights()

            # Save trained parameters for hid--pen
            self.savetofile_rbm(loc="trained_rbm", name="hid--pen")

            print("training pen+lbl--top")
            """
            CD-1 training for pen+lbl--top
            """
            # Get hidden layer activations from hid--pen
            hidden_activations = self.rbm_stack["hid--pen"].get_h_given_v(hidden_activations)[0]

            # Concatenate labels with activations for pen+lbl--top
            visible_data = np.concatenate((hidden_activations, lbl_trainset), axis=1)
            self.rbm_stack["pen+lbl--top"].cd1(visible_data, n_iterations, n_epochs=n_epochs)

            # Untwine the weights for the next layer
            self.rbm_stack["hid--pen"].untwine_weights()

            # Save trained parameters for pen+lbl--top
            self.savetofile_rbm(loc="trained_rbm", name="pen+lbl--top")

        return    

    
    def loadfromfile_rbm(self,loc,name):
        
        self.rbm_stack[name].weight_vh = np.load("%s/rbm.%s.weight_vh.npy"%(loc,name))
        self.rbm_stack[name].bias_v    = np.load("%s/rbm.%s.bias_v.npy"%(loc,name))
        self.rbm_stack[name].bias_h    = np.load("%s/rbm.%s.bias_h.npy"%(loc,name))
        print ("loaded rbm[%s] from %s"%(name,loc))
        return
        
    def savetofile_rbm(self,loc,name):
        
        np.save("%s/rbm.%s.weight_vh"%(loc,name), self.rbm_stack[name].weight_vh)
        np.save("%s/rbm.%s.bias_v"%(loc,name),    self.rbm_stack[name].bias_v)
        np.save("%s/rbm.%s.bias_h"%(loc,name),    self.rbm_stack[name].bias_h)
        return
    
    def loadfromfile_dbn(self,loc,name):
        
        self.rbm_stack[name].weight_v_to_h = np.load("%s/dbn.%s.weight_v_to_h.npy"%(loc,name))
        self.rbm_stack[name].weight_h_to_v = np.load("%s/dbn.%s.weight_h_to_v.npy"%(loc,name))
        self.rbm_stack[name].bias_v        = np.load("%s/dbn.%s.bias_v.npy"%(loc,name))
        self.rbm_stack[name].bias_h        = np.load("%s/dbn.%s.bias_h.npy"%(loc,name))
        print ("loaded rbm[%s] from %s"%(name,loc))
        return
        
    def savetofile_dbn(self,loc,name):
        
        np.save("%s/dbn.%s.weight_v_to_h"%(loc,name), self.rbm_stack[name].weight_v_to_h)
        np.save("%s/dbn.%s.weight_h_to_v"%(loc,name), self.rbm_stack[name].weight_h_to_v)
        np.save("%s/dbn.%s.bias_v"%(loc,name),        self.rbm_stack[name].bias_v)
        np.save("%s/dbn.%s.bias_h"%(loc,name),        self.rbm_stack[name].bias_h)
        return
    
