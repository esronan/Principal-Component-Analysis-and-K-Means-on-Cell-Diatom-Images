import numpy as np
from matplotlib import pyplot as plt

#Exercise 1
print("Exercise 1")
print("See pop-up window for graphs")
toy_data = np.loadtxt("pca_toydata.txt")
diatoms = np.loadtxt("diatoms.txt")
dims = diatoms.shape

diatomxs = np.array([diatoms[:,2*x] for x in range(int(dims[1]/2))]).T
diatomys = np.array([diatoms[:,2*x-1] for x in range(int(dims[1]/2))]).T

def stacker(cellarray): #To complete cell outline in graph
	return np.hstack((cellarray, cellarray[0]))

#Exercise 1.a
plt.figure()
plt.axis("equal")
plt.plot(stacker(diatomxs[0]),stacker(diatomys[0]), marker = "x", mec = "black", color = "m")

plt.title("Plot of the first diatom data set")
# plt.show()

#Exercise 1.b
plt.figure()
plt.axis("equal")
plt.title("Plot of all diatomes superimposed on each other")
blues = plt.get_cmap("Blues")
for i in range(dims[0]):
    plt.plot(stacker(diatomxs[i]),stacker(diatomys[i]))#, color = blues(i/dims[0]))
plt.show()    


#Exercise 2

############################## Principal component analysis ##############################

class pca():
    def __init__(self,data_x,data_y = None):
        self.data = data_x
        self.mean = np.mean(self.data, axis = 0)
        self.std = np.std(self.data, axis = 0)
        self.stndrd = (self.data-self.mean)/self.std
        self.cntrd = self.data - self.mean
        self.dims = self.data.shape
        self.data_y = data_y
        
    def get_eig(self, centered = False, standardised = False, whitened = False):
        if standardised == True:
            data = self.stndrd
        elif centered == True:
            data = self.cntrd
        else:
            data = self.data
        cov_matrix = np.dot(data.T, data)/(self.dims[0]-1)
        eigval, eigvec = np.linalg.eig(cov_matrix)

        eigsort = np.argsort(-eigval)
        self.eigval = eigval[eigsort]
        self.eigvec = eigvec[:,eigsort]

        # if whitened = True:
        # 	self.eigval = np.diag(1. / np.sqrt(self.eigval+1E-18))
        # 	W = np.dot(np.dot(self.eigvec, self.eigval), self.eigvec.T)
        # 	self.data = np.dot(self.data, W)
    
    def plots(self, opt_title = ""):

        blues = plt.get_cmap("Blues")
        fig, axes = plt.subplots(1,3, figsize = (22,8))
        fig.suptitle(opt_title, fontsize = 16)
        plot_matrix = np.empty((3,5,self.dims[1]))
        for i in range(3):        
            for j in range(5):
                plot_matrix[i,j] = self.mean + (j-2)*abs(np.sqrt(self.eigval[i]))*self.eigvec[:,i]
                xs = plot_matrix[i,j,::2]
                ys = plot_matrix[i,j,1::2]

                axes[i].plot(stacker(xs), stacker(ys),label = "${}\u03C3_{}\u03B5_{}$".format(j-2,i,i),color = blues((j+1)/5))
            axes[i].axis("equal")
            axes[i].set_title("PC{}".format(i+1))
            axes[i].legend(loc = "upper right")
        plt.show()
        
            
    def projection(self, pc = 2, classifications = [], meanplot = False, centres = [], last_2 = False, opt_title = False):
        title = "Projection on two first principal components"
        data = self.data
        if last_2:
            data = self.data[:-2]
            title = "Projection on two first principal components (without two last data points)"
            
        projected = np.dot(data, self.eigvec[:,:2])
        
        if opt_title != False:
            title = title + opt_title
        
        plt.figure(figsize = (12,8))
        
        if classifications != []: 
            positives = projected[classifications == 0]
            negatives = projected[classifications == 1]

            plt.scatter(positives[:,0], positives[:,1], label = "Weed", color = "c", s = 30, alpha = 0.25)
            plt.scatter(negatives[:,0], negatives[:,1], label = "Crop", color = "m", s = 30, alpha = 0.25)

            if meanplot == True:
            	posmean, negmean = np.mean(positives, axis = 0), np.mean(negatives, axis = 0)
            	print("means",posmean,negmean)
            	plt.scatter([posmean[0],negmean[0]], [posmean[1],negmean[1]], label = "Means of each class", color = "black")


            if centres != []:
                pccentres = np.dot(centres, self.eigvec[:,:2])
                plt.scatter(pccentres[0,0],pccentres[0,1], label = "Class 1 centroid", color = "yellow", edgecolors = "black")
                plt.scatter(pccentres[1,0],pccentres[1,1], label = "Class 2 centroid", color = "orange", edgecolors = "black")

        else:    
            plt.scatter(projected[:,0],projected[:,1], s = 10)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.axis('equal')
        plt.title(title)
        plt.legend()
        plt.show()
        


#Exercise 2
print("Exercise 2")
diatompca = pca(diatoms)
diatompca.get_eig()
diatompca.plots(opt_title = "Outline Variance Across Principal Components")



#Exercise 3.A
print("Exercise 3.A")
print("See pop-up window for graphs")
print("No centering or standardisation")
diatompca.get_eig()
diatompca.plots("Outline Variance Across Principal Components (Un-centered/un-standardised Data)")
print("Centered data")
diatompca.get_eig(centered = True)
diatompca.plots("Outline Variance Across Principal Components (Centered Data)")
print("Standardised data")
diatompca.get_eig(standardised = True)
diatompca.plots("Outline Variance Across Principal Components (Standardised Data)")
# diatompca.get_eig(whitened = True)
# diatompca.projection("Whitened Data Projection")


#Exercise 3.B
print("Exercise 3.B")
print("See pop-up window for graphs")

# plt.scatter(toy_data[:,0],toy_data[:,1])
# plt.show()
toypca = pca(toy_data)
toypca.get_eig()
toypca.projection()

toypca2 = pca(toy_data[:-2])
toypca2.get_eig()
toypca2.projection()


# In[33]:


print("Exercise 4")
print("See pop-up window for graphs")


#Pesticide Data
pesticide_train = np.loadtxt('IDSWeedCropTrain.csv',delimiter=',')
pesticide_test = np.loadtxt('IDSWeedCropTest.csv',delimiter=',')
pesticide_xtrain = pesticide_train[:,:-1]
pesticide_ytrain = pesticide_train[:,-1]
pesticide_xtest = pesticide_test[:,:-1]
pesticide_ytest = pesticide_test[:,-1] 



############################## K-means algorithm ##############################


def k_means(data, iters = 13, k = 2):

    #initialise matrix of classifications, distances and centres
    classifications = np.zeros((data.shape[0]))
    distances = np.zeros((k, 1))
    centres = np.zeros((k, data.shape[1]))

    #initalise centres with the first 2 data points
    for i in range(k):
        centres[i] = data[i]

    #initalise convergence
    converged = False

    #iteratively calculate classifications and cluster centres
    while converged == False: 
        converged = True
        for x in range(data.shape[0]):
            for i in range(k):
                #Sum of squared differences
                total = 0
                for j in range (data.shape[1]):
                     total += (data[x,j] - centres[i,j])**2
                distances[i] = total
            if classifications[x] != np.argmin(distances):
                classifications[x] = np.argmin(distances)
                converged = False

        #Recalculate centres by taking the means of each cluster
        if converged == False:
            for i in range(k):
                centres[i] = np.mean(data[np.where(classifications == i)],axis=0)
    print("K-means complete.")
    return classifications, centres

classes, centres = k_means(pesticide_xtrain) 
print("2-Means cluster centres:", "\n1:", centres[0], "\n2:", centres[1])

classes = abs(classes-1) #Flip classifications (because k-means algorithm is agnostic to positive/negative class) so that the colours align with real classes (to allow for comparison)

pesticide_pca = pca(pesticide_xtrain)
pesticide_pca.get_eig(standardised = True)
pesticide_pca.projection(classifications = classes, centres = centres, opt_title = " (depicted with 2-means clustering classifications)")
pesticide_pca.projection(classifications = pesticide_ytrain, meanplot = True, centres = centres, opt_title = " (depicted with the correct labels)")

