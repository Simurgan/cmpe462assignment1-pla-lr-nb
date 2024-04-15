#sample = X_test.sample(1) # taking a random sample
#print("Value: ",sample.Diagnosis.values[0])
#
#prior_m = len(class_M)/len(X_train)
#prior_b = len(class_B)/len(X_train)
#
#posterior_m = prior_m
#posterior_b = prior_b
#
#for i in range (2, len(dataset.variables)): # for all features
#    likelihood_m = (1/(np.sqrt(2*np.pi*variances_m[i-2])))*np.exp(-((sample[dataset.variables.name[i]].values[0]-means_m[i-2])**2)/(2*variances_m[i-2]))
#    likelihood_b = (1/(np.sqrt(2*np.pi*variances_b[i-2])))*np.exp(-((sample[dataset.variables.name[i]].values[0]-means_b[i-2])**2)/(2*variances_b[i-2]))
#    posterior_m = posterior_m * likelihood_m
#    posterior_b = posterior_b * likelihood_b
#
#if(posterior_m > posterior_b):
#    print("Prediction: M")
#else:
#    print("Prediction: B")
