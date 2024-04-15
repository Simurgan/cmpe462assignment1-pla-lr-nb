import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class NaiveBayes(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(NaiveBayes, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Parameters for the class priors
        self.class_priors = nn.Parameter(torch.ones(num_classes) / num_classes, requires_grad=True)
        
        # Parameters for the mean and variance of each feature for each class
        self.class_means = nn.Parameter(torch.randn(num_classes, input_dim), requires_grad=True)
        self.class_variances = nn.Parameter(torch.ones(num_classes, input_dim), requires_grad=True)

    def forward(self, x):
        # Calculate log likelihoods for each class
        log_likelihoods = []
        for c in range(self.num_classes):
            class_mean = self.class_means[c]
            class_variance = self.class_variances[c]
            
            # Calculate log likelihood for this class
            log_likelihood = -0.5 * torch.sum(torch.log(2 * np.pi * class_variance) + 
                                               ((x - class_mean) ** 2) / class_variance, dim=1)
            log_likelihoods.append(log_likelihood)
        
        # Stack the log likelihoods along dimension 1
        log_likelihoods = torch.stack(log_likelihoods, dim=1)
        
        # Add log prior probabilities
        log_probabilities = log_likelihoods + torch.log(self.class_priors)
        
        # Calculate log-sum-exp for numerical stability
        max_log_prob = torch.max(log_probabilities, dim=1, keepdim=True)[0]
        log_prob_sum_exp = torch.log(torch.sum(torch.exp(log_probabilities - max_log_prob), dim=1, keepdim=True)) + max_log_prob
        
        # Calculate log probabilities
        log_probabilities_normalized = log_probabilities - log_prob_sum_exp
        
        return log_probabilities_normalized

    def predict(self, x):
        log_probs = self.forward(x)
        _, predicted = torch.max(log_probs, 1)
        return predicted

# Example usage
# Assuming X_train, y_train, X_test, y_test are your training and testing data
# X_train and X_test are torch tensors, y_train and y_test are numpy arrays

# # Convert y_train to one-hot encoding
# num_classes = len(np.unique(y_train))
# y_train_one_hot = np.eye(num_classes)[y_train]
# 
# # Define model
# input_dim = X_train.shape[1]
# model = NaiveBayesClassifier(input_dim, num_classes)
# 
# # Define loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
# 
# # Training loop
# num_epochs = 100
# for epoch in range(num_epochs):
#     # Forward pass
#     outputs = model(X_train)
#     loss = criterion(outputs, torch.tensor(y_train))
#     
#     # Backward pass and optimization
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     
#     if (epoch+1) % 10 == 0:
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
# 
# # Test the model
# with torch.no_grad():
#     predicted = model.predict(X_test)
#     accuracy = (predicted.numpy() == y_test).mean()
#     print(f'Test Accuracy: {accuracy:.4f}')
#    