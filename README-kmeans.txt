Execute: python kmeans.py [input_image] [k] [output_image]

example: python kmeans.py Koala.jpg 5 output.jpg

(This will load the data from the input_image and convert it into RGB array. The program will then select k random points from the RGB array to be used as inital cluster centroids. Then it will assign the RGB points to nearest centroid and calculate the average of the points for each cluster, which will function as the cluster centroids for the next iteration. The last two steps will be performed in a loop until there is no further change or 100 iterations, whichever comes earlier. Finally the program will do this entire process 10 times and display the average compression percentage and the variance in the 10 iterations. It will also generate 10 output images and csv files compression.csv & compression_perc.csv)

Output:

Iteration 1 : Image size reduced from 780831 bytes to 197650 bytes
Iteration 2 : Image size reduced from 780831 bytes to 197647 bytes
Iteration 3 : Image size reduced from 780831 bytes to 197654 bytes
Iteration 4 : Image size reduced from 780831 bytes to 197653 bytes
Iteration 5 : Image size reduced from 780831 bytes to 197654 bytes
Iteration 6 : Image size reduced from 780831 bytes to 197647 bytes
Iteration 7 : Image size reduced from 780831 bytes to 197704 bytes
Iteration 8 : Image size reduced from 780831 bytes to 198015 bytes
Iteration 9 : Image size reduced from 780831 bytes to 197654 bytes
Iteration 10 : Image size reduced from 780831 bytes to 197649 bytes
Please find the output images and csv files compression.csv & compression_perc.csv
Average compression achieved : 74.68175571922734 %
Variance in compression : 0.00019351266320131964
