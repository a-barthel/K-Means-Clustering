#! /usr/bin/env python
__author__ 	= "Andrew Barthel"
__email__	= "abarthe1@asu.edu"

###############################################################
###############################################################
###############################################################
#
# Andrew Barthel - 1217975070
# abarthe1@asu.edu
# Arizona State University
# CSE 575 - Statistical Machine Learning
# Fall 2019
#
# Project 2 : K-Means Unsupervised Learning
# Due - 9/25/19
#
# K-Means Algorithm for Unsupervised Learning
#
###############################################################
###############################################################
###############################################################





###############################################################
####### PROGRAM SETUP - IMPORTS AND GLOBAL VARIABLES ##########
###############################################################

# Import statements for libraries needed
import scipy.io
import sys
import math
import numpy as np
import random 
import matplotlib.pyplot as plt

###############################################################
###############################################################
###############################################################

# Utility function to find Euclidean Distance between two points.
# This takes in two points and returns the euclidean distance between them.
def euclidean_dist(point1, point2):
	return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

# Assign data points to centroids.
# This takes in the data sample and the points of the centroids.
# It then computes the distance for each point in the sample to
# each centroid and assigns it to the closest one.
# This returns a list of assignments corresponding to k values
def assign(data_sample, centroids):
	# Placeholder variables
	clusters = []
	length = []
	lowest = 999999
	index = 0

	# Compute the distance for each point to all centroids
	for x in data_sample:
		for y in centroids:
			length.append(euclidean_dist(x, y))
		for i in range(len(centroids)):
			if length[i] <= lowest:
				lowest = length[i]
				index = i
		clusters.append(index) # Assigns sample to centroid via index of centroid in the centroids list.
		length = []
		lowest = 999999
		index = 0
	
	# Return assignment
	return clusters

# Objective function J for the data set
# This takes in the data sample, the centroid points, the assignment to centroid
# via index of centroid list, and k value.
# It returns the objective function for that value of k.
def objective_function(data_sample, centroids, clusters, k):
	# Placeholder variables
	i = 0
	total = 0

	# Compute the J function defined by k-means algo.
	while i < k:
		j = 0
		for x in data_sample:
			if clusters[j] == i:
				total += euclidean_dist(x, centroids[clusters[j]])
			j += 1
		i += 1

	# Return the total objective function of this k value.
	return total

# Plot the objective function output.
# This takes in the obj function values for both runs of both strategies
# Plots the obj function vs k plots and returns zero.
def plotting(strategy1_run1, strategy2_run1, strategy1_run2, strategy2_run2):
	# Create k list to have x axis values
	k = [2,3,4,5,6,7,8,9,10]

	# Plot the two strategies with both runs.
	plt.figure("K Means")
	plt.subplot(211)
	plt.plot(k, strategy1_run1)
	plt.plot(k, strategy1_run2)
	plt.ylabel("Objective Function")
	plt.xlabel("K")
	plt.subplot(212)
	plt.plot(k, strategy2_run1)
	plt.plot(k, strategy2_run2)
	plt.ylabel("Objective Function")
	plt.xlabel("K")
	plt.show()

	# Return status success.
	return 0

# Update centroids based on new cluster assignment.
# Takes in data sample, cluster assignments via index, and k
# This function updates the centroids via the new assignments given.
# Returns a list of new, updated centroid points.
def update(data_sample, clusters, k):
	# Placeholder variables
	x_tot = 0
	y_tot = 0
	count = 1
	i = 0
	centers = []

	# Update centroids via mean points of all samples assigned to centroid.
	while i < k:
		j = 0
		for x in data_sample:
			if clusters[j] == i:
				x_tot += x[0]
				y_tot += x[1]
				count += 1
			j += 1
		centers.append((float(x_tot/count), float(y_tot/count)))
		x_tot = 0
		y_tot = 0
		count = 1
		i += 1

	# Return the new centers.
	return centers

# See if centroids move.
# Takes in the list of old and new centroids.
# If the centroids moved, it is denoted by one, if not, zero.
def diff(old_centroids, centroids):
	change = 0
	for i in range(len(centroids)):
		if (old_centroids[i][0] != centroids[i][0]) or (old_centroids[i][1] != centroids[i][1]):
			change = 1
	return change

# Function for the k means strategy one.
# Takes in the sample and k value.
# Returns the centroid points and cluster assignments via index.
# As defined by project handout.
def k_means_strategy_one(data_sample, k):
	# Placeholder variables.
	centroids = []
	indecies = []

	# Pick k random centroids from data.
	i = 0
	while i < k:
		index = random.randint(0, len(data_sample) - 1)
		for j in range(len(indecies)+1):
			if (len(indecies) != 0) and (index == indecies[j]):
				continue
			else:
				centroids.append(data_sample[index])
				indecies.append(index)
				i += 1
				break

	# Update centroids until no change.
	change = 1
	while change > 0:
		clusters = assign(data_sample, centroids)
		old_centroids = centroids
		centroids = update(data_sample, clusters, k)
		change = diff(old_centroids, centroids)
	
	# Return centroid points and index assignments
	return centroids, clusters

# Function for the k means strategy two.
# Takes in samples and k value.
# Returns a list of centroid points and cluster assignments via index.
# As defined by project handout.
def k_means_strategy_two(data_sample, k):
	# Placeholder variables.
	centroids = []
	farthest = 0
	total = 0
	index = 9999999

	# Initial random centroid.
	centroids.append(data_sample[random.randint(0, len(data_sample) - 1)])

	# Compute the other centroids by choosing farthest sample from previous centroids.
	while len(centroids) < k:
		for x in data_sample:
			for i in range(len(centroids)):
				total += euclidean_dist(x, centroids[i])
			if total > farthest:
				farthest = total
				hold = x
			total = 0
		farthest = 0
		centroids.append(hold)
	
	# Update centroids until no change.
	change = 1
	while change > 0:
		clusters = assign(data_sample, centroids)
		old_centroids = centroids
		centroids = update(data_sample, clusters, k)
		change = diff(old_centroids, centroids)

	# Return centroid points and cluster assignments via index.
	return centroids, clusters

# Main function
def main():

	# Open the data samples and parse what we need
	data_sample = scipy.io.loadmat('AllSamples.mat')
	data_sample = data_sample["AllSamples"]

	# Placeholder variables.
	method_one_run_one = []
	method_one_run_two = []
	method_two_run_one = []
	method_two_run_two = []

	# Run for k = 2,...,10
	for k in range(2, 11):
		method_one = k_means_strategy_one(data_sample, k)
		method_one_run_one.append(objective_function(data_sample, method_one[0], method_one[1], k))
		method_two = k_means_strategy_two(data_sample, k)
		method_two_run_one.append(objective_function(data_sample, method_two[0], method_two[1], k))
		method_one = k_means_strategy_one(data_sample, k)
		method_one_run_two.append(objective_function(data_sample, method_one[0], method_one[1], k))
		method_two = k_means_strategy_two(data_sample, k)
		method_two_run_two.append(objective_function(data_sample, method_two[0], method_two[1], k))

	# Plot the obj functions
	ret = plotting(method_one_run_one, method_two_run_one, method_one_run_two, method_two_run_two)
	
	# Exit on success - if plotting worked
	sys.exit(ret)

	


# Ensure to call main function
if __name__ == "__main__":
	main()
else:
	print('Something is really wrong....there is no main function, man?')
	sys.exit(1) #error