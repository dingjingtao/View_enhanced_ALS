package main;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

import algorithms.ItemPopularity;
import algorithms.MF_fastALS;
import algorithms.MFbpr;
import algorithms.MFbpr_multi_channel;
import algorithms.MFbpr_multi_channel;
import data_structure.DenseMatrix;
import data_structure.Rating;
import data_structure.SparseMatrix;
import utils.Printer;

public class mfbpr_pos extends main {
	public static void main(String[] argv)throws IOException{
		String dataset_name = "yelp";
		String method = "mfbpr_pos";
		double w0 = 0.2;
		boolean showProgress = true;
		boolean showLoss = false;
		int factors = 32;
		int maxIter = 10;
		double reg = 0.2;
		double alpha = 0.4;
		double lr = 0.01; 
		boolean adaptive = false;
		double parad = 0.4;
		SparseMatrix viewmatrix;
		boolean testmode = false ;
		double dstart = 0.5;
		double dend = 1;
		int dnum = 6;
		int showbound = 0 ;
		int showtime = 1;
		int paraK = 1;
		int Valset = 1000;
		double paraw = 0.5;
		int endn = 10;
		String viewfile = "";
		String datafile = "";
				
		if (argv.length > 0) {
			//dataset_name = argv[0];
			//method = argv[1];
			w0 = Double.parseDouble(argv[2]);
			showProgress = Boolean.parseBoolean(argv[3]);
			showLoss = Boolean.parseBoolean(argv[4]);
			factors = Integer.parseInt(argv[5]);
			maxIter = Integer.parseInt(argv[6]);
			reg = Double.parseDouble(argv[7]);
			if (argv.length > 8) alpha = Double.parseDouble(argv[8]);
			datafile = argv[9];
			viewfile = argv[10];
			testmode = Boolean.parseBoolean(argv[11]);
			dstart = Double.parseDouble(argv[12]);
			
//			dend = Double.parseDouble(argv[13]);
//			dnum = Integer.parseInt(argv[14]);
//			if(argv.length > 15)
//			{
			showbound = maxIter-10;
			showtime = 1;
//			}
			paraK = 1;
			paraw = Double.parseDouble(argv[13]);
		}
		if (!datafile.contains("tmall"))
			ReadRatings_HoldOneOut(datafile);
		else
			ReadRatings_HoldOneOut_Tmall(datafile);
					
		// Step 1. Construct data structure for sorting.
		long startTime = System.currentTimeMillis();
		ArrayList<ArrayList<Rating>> user_ratings = new ArrayList<ArrayList<Rating>>();
		BufferedReader reader = new BufferedReader(
				new InputStreamReader(new FileInputStream(viewfile)));
		String line;
		while((line = reader.readLine()) != null) {
			Rating rating = new Rating(line);
			 
			while (user_ratings.size()  < userCount) { // create a new user
				user_ratings.add(new ArrayList<Rating>());
			}
			user_ratings.get(rating.userId).add(rating);
		}
		reader.close();
										
		// Step 3. Generated splitted matrices (implicit 0/1 settings). 
		System.out.printf("Generate view/buy matrics.");
		
		startTime = System.currentTimeMillis();
		viewmatrix = new SparseMatrix(userCount, itemCount);
		
		for (int u = 0; u < userCount; u ++) {
			{
			ArrayList<Rating> ratings = user_ratings.get(u);
			for (int i = ratings.size() - 1; i >= 0; i --) {
				int userId = ratings.get(i).userId;
				int itemId = ratings.get(i).itemId;										
				viewmatrix.setValue(userId, itemId, 1);		
			}
			}
		}			
		System.out.printf("[%s]\n", Printer.printTime(
				System.currentTimeMillis() - startTime));			
		// Print some basic statistics of the dataset.
		System.out.println ("Data\t" + viewfile);
		System.out.printf("#Ratings\t %d (train)\n", 
				viewmatrix.itemCount());	
		
		ItemPopularity popularity = new ItemPopularity(trainMatrix, testRatings, topK, threadNum);
		evaluate_model(popularity, "Popularity");
		
		double init_mean = 0;
		double init_stdev = 0.01;
		adaptive = showLoss;
		if (testmode == true)
			System.out.printf("testmode wrong!\n");
		else {
			parad = dstart;
			System.out.printf("%s: showProgress=%s, factors=%d, maxIter=%d, reg=%.6f, w0=%.6f, alpha=%.2f,parad = %.6f,parak=%d,paraw = %.6f\n",
					method, showProgress, factors, maxIter, reg, w0, alpha,parad,paraK,paraw);	
			MFbpr_multi_channel bpr = new MFbpr_multi_channel(trainMatrix, testRatings, topK, threadNum, 
					factors, maxIter, w0, adaptive, reg, init_mean, init_stdev, showProgress,
					showbound,showtime,viewmatrix,parad,paraK,paraw);
			evaluate_model(bpr, "MFbpr_multi_channel");	
	    }	
	}
}