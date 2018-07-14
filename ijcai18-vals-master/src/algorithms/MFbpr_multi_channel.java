package algorithms;

import data_structure.Rating;
import data_structure.SparseMatrix;
import data_structure.DenseVector;
import data_structure.DenseMatrix;
import data_structure.Pair;

import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

import utils.Printer;

/**
 * Implement the multi-channel BPR loss.
 * Babak, Loni, et al. "Bayesian Personalized Ranking with Multi-Channel User Feedback." 
 * Proc. of RecSys 2016.
 * 
 * @author jingtaoding
 *
 */
public class MFbpr_multi_channel extends TopKRecommender {
	/** Model priors to set. */
	int factors = 10; 	// number of latent factors.
	int maxIter = 100; 	// maximum iterations.
	double lr = 0.01; 		// Learning rate
	boolean adaptive = false; 	// Whether to use adaptive learning rate 
	double reg = 0.01; 	// regularization parameters
  double init_mean = 0;  // Gaussian mean for init V
  double init_stdev = 0.1; // Gaussian std-dev for init V
  int showbound = 0;
  int showtime = 1;
  /** Model parameters to learn */
  public DenseMatrix U;	// latent vectors for users
  public DenseMatrix V;	// latent vectors for items
  int usercount =0;
  int itemcount =0;
  public Integer [][]viewdata;
  public Integer [][]buydata;
  boolean showProgress;
  public String onlineMode = "u";
  SparseMatrix viewMatrix;
  //parad =0 mean unfirom =1 means just view
  double parad = 0;
  int paraK = 1;
  double paraw = 0.5;
  Random rand = new Random();
	public MFbpr_multi_channel(SparseMatrix trainMatrix, ArrayList<Rating> testRatings,
			int topK, int threadNum, int factors, int maxIter, double lr, boolean adaptive, double reg, 
			double init_mean, double init_stdev, boolean showProgress,
			int showbound ,int showtime ,SparseMatrix viewMatrix,double parad,int paraK,double paraw) {
		super(trainMatrix, testRatings, topK, threadNum);
		this.factors = factors;
		this.maxIter = maxIter;
		this.lr = lr;
		this.adaptive = adaptive;
		this.reg = reg;
		this.init_mean = init_mean;
		this.init_stdev = init_stdev;
		this.showProgress = showProgress;
		this.showbound = showbound;
		this.showtime = showtime;
		this.viewMatrix = viewMatrix;
		this.parad = parad;
		this.paraK = paraK;
		this.paraw = paraw;
		// Init model parameters
		U = new DenseMatrix(userCount, factors);
		V = new DenseMatrix(itemCount, factors);
		U.init(init_mean, init_stdev);
		V.init(init_mean, init_stdev);
		this.usercount = userCount;
		this.itemcount = itemCount;
	}
	
	public MFbpr_multi_channel(SparseMatrix trainMatrix, ArrayList<Rating> testRatings,
			int topK, int threadNum, int factors, int maxIter, double lr, boolean adaptive, double reg, 
			double init_mean, double init_stdev, boolean showProgress ) {
		super(trainMatrix, testRatings, topK, threadNum);
		this.factors = factors;
		this.maxIter = maxIter;
		this.lr = lr;
		this.adaptive = adaptive;
		this.reg = reg;
		this.init_mean = init_mean;
		this.init_stdev = init_stdev;
		this.showProgress = showProgress;
		// Init model parameters
		U = new DenseMatrix(userCount, factors);
		V = new DenseMatrix(itemCount, factors);
		U.init(init_mean, init_stdev);
		V.init(init_mean, init_stdev);
		this.usercount = userCount;
		this.itemcount = itemCount;
	}
	
	//remove
	public void setUV(DenseMatrix U, DenseMatrix V) {
		this.U = U.clone();
		this.V = V.clone();
	}
	
	public void buildModel() {	
		int nonzeros = trainMatrix.itemCount();
		double hr_prev = 0;
		double Sbuy = trainMatrix.itemCount();
		double Sipv = viewMatrix.itemCount();
		double p = Sbuy/(Sbuy+paraw*Sipv);
		
		buydata = new Integer[userCount][];
		viewdata = new Integer[userCount][];
		for (int i =0;i<userCount;i++) {
			ArrayList<Integer> itemList = trainMatrix.getRowRef(i).indexList();
			ArrayList<Integer> viewList = viewMatrix.getRowRef(i).indexList();
			buydata[i] = itemList.toArray(new Integer [itemList.size()]);
			viewdata[i] = viewList.toArray(new Integer [viewList.size()]);
		}
		System.out.println("array has been available");
		
		for (int iter = 0; iter < maxIter; iter ++) {
			Long start = System.currentTimeMillis();
			Long mytimecount = 0L;
			rand = new Random();
			//System.out.print(V);
			// Each training epoch
			
			//Long mttime = System.nanoTime();
			for (int s = 0; s < nonzeros; s ++) { 
				// sample a user				
				int u = rand.nextInt(userCount); 
				//Long mttime = System.nanoTime();
				//ArrayList<Integer> itemList = trainMatrix.getRowRef(u).indexList();
				//ArrayList<Integer> viewList = viewMatrix.getRowRef(u).indexList();
				
				//if (trainMatrix.getRowLength(u) == 0)	continue;
				//if (viewMatrix.getRowLength(u) == 0)	continue;
				//mytimecount += System.nanoTime() - mttime;
				//if (itemList.size() ==0) continue;
				//if (viewList.size() ==0) continue;
				if (buydata[u].length ==0) continue;
				if (viewdata[u].length ==0) continue;
				
				double randq = rand.nextDouble();
				
				if (randq<p) {
					// sample a positive item
					//int i = trainMatrix.getRowRef(u).indexList().get(rand.nextInt(trainMatrix.getRowLength(u))); 
					//int i = itemList.get(rand.nextInt(itemList.size()));
					int i = buydata[u][rand.nextInt(buydata[u].length)];
					
					// One SGD step update
					update_ui_buy(u, i);
				}
				else {
					//int i = viewMatrix.getRowRef(u).indexList().get(rand.nextInt(viewMatrix.getRowLength(u)));
					//int i = viewList.get(rand.nextInt(viewList.size()));
					int i = viewdata[u][rand.nextInt(viewdata[u].length)];
					// One SGD step update
					update_ui_ipv(u, i);
				}

			}
			//mytimecount += System.nanoTime() - mttime;
			//System.out.println(mytimecount);
			// Show progress
			if (showProgress)
				if(iter >= showbound  | iter %50 == 0)
				showProgress(iter, start, testRatings);
			
			// Adjust the learning rate
			if (adaptive) {
				if (!showProgress)	evaluate(testRatings);
				double hr = ndcgs.mean();
				lr = hr > hr_prev ? lr * 1.05 : lr * 0.5;
				hr_prev = hr;
			}
			
		} // end for iter
		//output matrix
		/*
		FileWriter out = null;
		try {
		out = new FileWriter("yuguanghui/ui-factors/bprposIJCA");
		out.write("user factors:\n");
		int i =0;
		int j=0;
		for (i=0;i<usercount;i++) {
			for (j=0;j<factors;j++) {
				out.write(Double.toString(U.get(i, j)));
				out.write('\t');				
			}
			out.write('\n');
		}
		out.write("item factors:\n");
		for (i=0;i<itemcount;i++) {
			for (j=0;j<factors;j++) {
				out.write(Double.toString(V.get(i, j)));
				out.write('\t');				
			}
			out.write('\n');
		}
		out.close();
		System.out.println("matrix has been writen!\n");
		}catch (Exception e) {   

            e.printStackTrace();   

        }*/
		
	}
	
	public void runOneIteration() {
		int nonzeros = trainMatrix.itemCount();
		rand = new Random();
		// Each training epoch
		for (int s = 0; s < nonzeros; s ++) { 
			// sample a user
			int u = rand.nextInt(userCount); 
			ArrayList<Integer> itemList = trainMatrix.getRowRef(u).indexList();
			if (itemList.size() == 0)	continue;
			// sample a positive item
			int i = itemList.get(rand.nextInt(itemList.size())); 
			
			// One SGD step update
			update_ui(u, i);

		}
	}
	
	//One SGD step for a positive instance.
	private void update_ui(int u, int i) {
		//decide j from uniform or view
		double p = rand.nextDouble();
		int j = rand.nextInt(itemCount);
		ArrayList<Integer> viewList = viewMatrix.getRowRef(u).indexList();
		if(viewMatrix.getRowRef(u).indexList().size() == 0 || parad < p) {
			// sample a negative item (uniformly random)			
			while (trainMatrix.getValue(u, j) != 0) {
				j = rand.nextInt(itemCount);
			}
		}
		else {
			j = viewMatrix.getRowRef(u).indexList().get(rand.nextInt(viewMatrix.getRowRef(u).indexList().size()));
			//System.out.println(viewMatrix.getRowLength(u));
			//System.out.println(viewList.size());
		}
		
		// BPR update rules
		double y_pos = predict(u, i);  // target value of positive instance
    double y_neg = predict(u, j);  // target value of negative instance
    double mult = -partial_loss(y_pos - y_neg);
    
    for (int f = 0; f < factors; f ++) {
    	double grad_u = V.get(i, f) - V.get(j, f);
    	U.add(u, f, -lr * (mult * grad_u + reg * U.get(u, f)));
    	
    	double grad = U.get(u, f);
    	V.add(i, f, -lr * (mult * grad + reg * V.get(i, f)));
    	V.add(j, f, -lr * (-mult * grad + reg * V.get(j, f)));
    }
	}
	
	//One SGD step for a positive instance.
	
	private void update_ui_buy(int u, int i) {
		//decide j from uniform or view
		double p = rand.nextDouble();
		int j = rand.nextInt(itemCount);
		//ArrayList<Integer> viewList = viewMatrix.getRowRef(u).indexList();
		
        int jsam = 0;
        double jscore = 0;
		for (int k=0; k<paraK; k++){	
			j = rand.nextInt(itemCount);
			p = rand.nextDouble();
			if(viewMatrix.getRowRef(u).indexList().size() == 0 || parad < p) {		
				while (trainMatrix.getValue(u, j) != 0) {
					j = rand.nextInt(itemCount);
				}
			}
			else {
				//ArrayList<Integer> viewList = viewMatrix.getRowRef(u).indexList();
				//j = viewList.get(rand.nextInt(viewList.size()));
				j = viewdata[u][rand.nextInt(viewdata[u].length)];
				//j = viewMatrix.getRowRef(u).indexList().get(rand.nextInt(viewMatrix.getRowLength(u)));
			}
			
			if (k == 0) {
				jscore = predict(u,j);
				jsam = j;
			}				
			if (predict(u,j)>jscore) {
				jscore = predict(u,j);
				jsam = j;
			}
		}
		j = jsam;
		
		
		// BPR update rules
	double y_pos = predict(u, i);  // target value of positive instance
    double y_neg = predict(u, j);  // target value of negative instance
    double mult = -partial_loss(y_pos - y_neg);
    double grad = 0;
    double grad_u = 0;
    for (int f = 0; f < factors; f ++) {
    	grad_u = V.get(i, f) - V.get(j, f);
    	U.add(u, f, -lr * (mult * grad_u + reg * U.get(u, f)));
    	
    	grad = U.get(u, f);
    	V.add(i, f, -lr * (mult * grad + reg * V.get(i, f)));
    	V.add(j, f, -lr * (-mult * grad + reg * V.get(j, f)));
    }
    
    if(Double.isInfinite(grad)||Double.isInfinite(grad_u)) {
    	System.out.print("INfinite num has been catched \n\n\n");
    	System.exit(0);
    }
	}
	
	private void update_ui_ipv(int u, int i) {
		//decide j from uniform or view
		int j = rand.nextInt(itemCount);
		
        int jsam = 0;
        double jscore = 0;
		for (int k=0; k<paraK; k++){	
			j = rand.nextInt(itemCount);
			while (trainMatrix.getValue(u, j) != 0) {
				j = rand.nextInt(itemCount);
			}
			if (k == 0) {
				jscore = predict(u,j);
				jsam = j;
			}				
			if (predict(u,j)>jscore) {
				jscore = predict(u,j);
				jsam = j;
			}
		}
		j = jsam;
		// BPR update rules
		double y_pos = predict(u, i);  // target value of positive instance
    double y_neg = predict(u, j);  // target value of negative instance
    double mult = -partial_loss(y_pos - y_neg);
    double grad = 0;
    double grad_u = 0;
    for (int f = 0; f < factors; f ++) {
    	grad_u = V.get(i, f) - V.get(j, f);
    	U.add(u, f, -lr * (mult * grad_u + reg * U.get(u, f)));
    	
    	grad = U.get(u, f);
    	V.add(i, f, -lr * (mult * grad + reg * V.get(i, f)));
    	V.add(j, f, -lr * (-mult * grad + reg * V.get(j, f)));
    }
    
    if(Double.isInfinite(grad)||Double.isInfinite(grad_u)) {
    	System.out.print("INfinite num has been catched \n\n\n");
    	System.exit(0);
    }
	}
	
	
	@Override
	public double predict(int u, int i) {
		return U.row(u, false).inner(V.row(i, false));
	}
	
  // Partial of the ln sigmoid function used by BPR.
  private double partial_loss(double x) {
    double exp_x = Math.exp(-x);
    return exp_x / (1 + exp_x);
  }

  // Implement the Recsys08 method: Steffen Rendle, Lars Schmidt-Thieme,
  // "Online-Updating Regularized Kernel Matrix Factorization Models"
	public void updateModel(int u, int item) {
		trainMatrix.setValue(u, item, 1);
		rand = new Random();
		
		// user retrain
		ArrayList<Integer> itemList = trainMatrix.getRowRef(u).indexList();
		for (int iter = 0; iter < maxIterOnline; iter ++) {
			Collections.shuffle(itemList);
			
			for (int s = 0; s < itemList.size(); s ++) {
				// retrain for the user or for the (user, item) pair
				int i = onlineMode.equalsIgnoreCase("u") ? itemList.get(s) : item;
				// One SGD step update
				update_ui(u, i);
			}
		}
		
	}
}
