package main;

import java.io.IOException;
import java.util.ArrayList;

import data_structure.DenseMatrix;
import data_structure.Rating;
import data_structure.SparseMatrix;
import utils.Printer;
import algorithms.MF_fastALS;
import algorithms.MFbpr;
import algorithms.MF_VALS;
import algorithms.ItemPopularity;

public class main_MF extends main {
	public static void main(String argv[]) throws IOException {
		String dataset_name = "buy";
		String sidedataset_name = "ipv";
		String method = "FastALS";
		double w0 = 10;
		double w1 = 1;
		double w2 = 1;
		double r1 = 1;
		boolean showProgress = false;
		boolean showLoss = true;
		int factors = 64;
		int maxIter = 500;
		double reg = 0.01;
		double alpha = 0.75;
		double beta = 0.2;
		double ratio = 0;
		double gamma1 = 0;
		double gamma2 = 0;
		
		if (argv.length > 0) {
			dataset_name = argv[0];
			method = argv[1];
			w0 = Double.parseDouble(argv[2]);
			showProgress = Boolean.parseBoolean(argv[3]);
			showLoss = Boolean.parseBoolean(argv[4]);
			factors = Integer.parseInt(argv[5]);
			maxIter = Integer.parseInt(argv[6]);
			reg = Double.parseDouble(argv[7]);
			if (argv.length > 8) alpha = Double.parseDouble(argv[8]);
			if (argv.length > 9) {
				sidedataset_name = argv[9];
				w1 = Double.parseDouble(argv[10]);
			}
			if (argv.length > 11) {
				beta = Double.parseDouble(argv[11]);
			}
			if (argv.length > 12) {
				r1 = Double.parseDouble(argv[12]);
				gamma1 = Double.parseDouble(argv[12]);
			}
			if (argv.length > 13){
				w2 = Double.parseDouble(argv[13]);
				gamma2 = Double.parseDouble(argv[13]);
			}
		}
		//ReadRatings_GlobalSplit("data/" + dataset_name + ".rating", 0.1);
		if (!dataset_name.contains("tmall"))
			ReadRatings_HoldOneOut(dataset_name);
		else
			ReadRatings_HoldOneOut_Tmall(dataset_name);
		if (method.equalsIgnoreCase("vieweALS")) {
			ReadSideRatings(sidedataset_name, r1);
		}
		
		System.out.printf("%s: showProgress=%s, factors=%d, maxIter=%d, reg=%f, w0=%.2f, alpha=%.2f, w1=%.6f, r1=%.2f, w2=%.6f, beta=%.2f\n",
				method, showProgress, factors, maxIter, reg, w0, alpha, w1, r1, w2, beta);
		System.out.println("====================================================");
		
		ItemPopularity popularity = new ItemPopularity(trainMatrix, testRatings, topK, threadNum);
		evaluate_model(popularity, "Popularity");
		
		double init_mean = 0;
		double init_stdev = 0.01;
		
		if (method.equalsIgnoreCase("fastals")) {
			MF_fastALS fals = new MF_fastALS(trainMatrix, testRatings, topK, threadNum,
					factors, maxIter, w0, alpha, reg, init_mean, init_stdev, showProgress, showLoss);
			evaluate_model(fals, "MF_fastALS");
		}
		
		ratio = beta;
		
		if (method.equalsIgnoreCase("vieweALS")) {
			MF_VALS eALSplusView = new MF_VALS(trainMatrix, testRatings, trainSideMatrix, topK, threadNum,
					factors, maxIter, w0, w1, alpha, reg, beta, gamma1, gamma2, init_mean, init_stdev, showProgress, showLoss);
			evaluate_model(eALSplusView, "MF_vieweALS");
		}
		
		if (method.equalsIgnoreCase("bpr")) {
			MFbpr bpr = new MFbpr(trainMatrix, testRatings, topK, threadNum,
					factors, maxIter, w0, false, reg, init_mean, init_stdev, showProgress);
			evaluate_model(bpr, "MFbpr");
		}
	
	} // end main
}
