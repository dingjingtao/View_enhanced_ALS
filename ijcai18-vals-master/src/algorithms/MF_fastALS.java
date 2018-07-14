package algorithms;

import data_structure.Rating;
import data_structure.SparseMatrix;
import data_structure.DenseVector;
import data_structure.DenseMatrix;
import data_structure.Pair;
import data_structure.SparseVector;
import happy.coding.math.Randoms;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import java.util.HashMap;

import utils.Printer;

/**
 * Fast ALS for weighted matrix factorization (with imputation)
 * 
 * @author xiangnanhe
 */
public class MF_fastALS extends TopKRecommender {
	/** Model priors to set. */
	int factors = 10; // number of latent factors.
	int maxIter = 500; // maximum iterations.
	double reg = 0.01; // regularization parameters
	double w0 = 1;
	double init_mean = 0; // Gaussian mean for init V
	double init_stdev = 0.01; // Gaussian std-dev for init V

	/** Model parameters to learn */
	public DenseMatrix U; // latent vectors for users
	public DenseMatrix V; // latent vectors for items

	/** Caches */
	DenseMatrix SU;
	DenseMatrix SV;
	double[] prediction_users, prediction_items;
	double[] rating_users, rating_items;
	double[] w_users, w_items;

	boolean showProgress;
	boolean showLoss;

	// weight for each positive instance in trainMatrix
	SparseMatrix W;

	// weight for negative instances on item i.
	double[] Wi;

	// weight of new instance in online learning
	public double w_new = 1;

	double getMinK(double[] v, int left, int right, int k) {
		double tmp = 0;
		if (left < right) {
			int i = left - 1, j = left;
			boolean flag = true;
			for (; j < right; ++j) {
				if (v[j] < v[right]) {
					++i;
					tmp = v[i];
					v[i] = v[j];
					v[j] = tmp;
				}
				if (v[j] != v[right]) {
					flag = false;
				}
			}
			tmp = v[i + 1];
			v[i + 1] = v[right];
			v[right] = tmp;
			if (flag) // avoid StackOverFlowError (equal items from left to right, large)
				return v[right];
			if (k == i + 1)
				return v[i + 1];
			else if (k <= i)
				return getMinK(v, left, i, k);
			else {
				// System.out.println(i+","+right+","+v[i+2]+","+v[right]);
				return getMinK(v, i + 2, right, k);
			}
		} else
			return v[left];
	}

	public MF_fastALS(SparseMatrix trainMatrix, ArrayList<Rating> testRatings, int topK, int threadNum, int factors,
			int maxIter, double w0, double alpha, double reg, double init_mean, double init_stdev, boolean showProgress,
			boolean showLoss) {
		super(trainMatrix, testRatings, topK, threadNum);
		this.factors = factors;
		this.maxIter = maxIter;
		this.w0 = w0;
		this.reg = reg;
		this.init_mean = init_mean;
		this.init_stdev = init_stdev;
		this.showLoss = showLoss;
		this.showProgress = showProgress;

		// Set the Wi as a decay function w0 * pi ^ alpha
		double sum = 0, Z = 0;
		double[] p = new double[itemCount];
		for (int i = 0; i < itemCount; i++) {
			p[i] = trainMatrix.getColRef(i).itemCount();
			sum += p[i];
		}
		// convert p[i] to probability
		for (int i = 0; i < itemCount; i++) {
			p[i] /= sum;
			if (p[i] > 0)
				p[i] = Math.pow(p[i], alpha);
			Z += p[i];
		}
		// assign weight
		Wi = new double[itemCount];
		double Wimin = 0, Wimax = 0, N0 = 0;
		for (int i = 0; i < itemCount; i++) {
			Wi[i] = w0 * p[i] / Z;
			if (Wi[i] < Wimin)
				Wimin = Wi[i];
			if (Wi[i] > Wimax)
				Wimax = Wi[i];
			if (Wi[i] == 0)
				N0++;
		}
		double[] Wii = new double[itemCount];
		System.arraycopy(Wi, 0, Wii, 0, itemCount);
		double Wi_50 = getMinK(Wii, 0, itemCount - 1, itemCount / 2);
		double Wi_10 = getMinK(Wii, 0, itemCount - 1, itemCount / 10);
		double Wi_30 = getMinK(Wii, 0, itemCount - 1, itemCount / 10 * 3);
		double Wi_70 = getMinK(Wii, 0, itemCount - 1, itemCount / 10 * 7);
		double Wi_90 = getMinK(Wii, 0, itemCount - 1, itemCount / 10 * 9);
		System.out.println("Wi\tmin: " + Wimin + "\t10-percentle: " + Wi_10 + "\t30-percentle: " + Wi_30
				+ "\t50-percentle: " + Wi_50 + "\t70-percentle: " + Wi_70 + "\t90-percentle: " + Wi_90 + "\tmax: "
				+ Wimax + "\tN0: " + N0);

		// By default, the weight for positive instance is uniformly 1.
		W = new SparseMatrix(userCount, itemCount);
		for (int u = 0; u < userCount; u++)
			for (int i : trainMatrix.getRowRef(u).indexList())
				W.setValue(u, i, 1);

		// Init caches
		prediction_users = new double[userCount];
		prediction_items = new double[itemCount];
		rating_users = new double[userCount];
		rating_items = new double[itemCount];
		w_users = new double[userCount];
		w_items = new double[itemCount];

		// Init model parameters
		U = new DenseMatrix(userCount, factors);
		V = new DenseMatrix(itemCount, factors);
		U.init(init_mean, init_stdev);
		V.init(init_mean, init_stdev);
		initS();
	}

	public void setTrain(SparseMatrix trainMatrix) {
		this.trainMatrix = new SparseMatrix(trainMatrix);
		W = new SparseMatrix(userCount, itemCount);
		for (int u = 0; u < userCount; u++)
			for (int i : this.trainMatrix.getRowRef(u).indexList())
				W.setValue(u, i, 1);
	}

	// Init SU and SV
	private void initS() {
		SU = U.transpose().mult(U);
		// Init SV as V^T Wi V
		SV = new DenseMatrix(factors, factors);
		for (int f = 0; f < factors; f++) {
			for (int k = 0; k <= f; k++) {
				double val = 0;
				for (int i = 0; i < itemCount; i++)
					val += V.get(i, f) * V.get(i, k) * Wi[i];
				SV.set(f, k, val);
				SV.set(k, f, val);
			}
		}
	}

	// remove
	public void setUV(DenseMatrix U, DenseMatrix V) {
		this.U = U.clone();
		this.V = V.clone();
		initS();
	}

	public void buildModel() throws IOException {
		// System.out.println("Run for FastALS. ");
		double loss_pre = Double.MAX_VALUE;
		
		if (!showProgress) {
			for (int iter = 0; iter < maxIter; iter++) {
				Long start = System.currentTimeMillis();

				// Update user latent vectors
				for (int u = 0; u < userCount; u++) {
					update_user(u);
				}

				// Update item latent vectors
				for (int i = 0; i < itemCount; i++) {
					update_item(i);
				}
				if ((iter >= maxIter - 10) || (iter % 20 == 0))
					showProgress(iter, start, testRatings);
				// Show loss
				if (showLoss)
					loss_pre = showLoss(iter, start, loss_pre);
			}
		}
		else {
			BufferedWriter hituserpath = new BufferedWriter(new FileWriter("fastals-hitusers.txt"));
//			BufferedWriter viewitemscore = new BufferedWriter(new FileWriter("eals+view-viewitemscore-w0-"+w0+"-gamma-"+gamma1+"-"+gamma2+".txt"));
			BufferedWriter buyitemscore = new BufferedWriter(new FileWriter("fastals-buyitemscore-w0-"+w0+".txt"));
			BufferedWriter itemfactor = new BufferedWriter(new FileWriter("fastals-itemfactor-w0-"+w0+".txt"));
			BufferedWriter userfactor = new BufferedWriter(new FileWriter("fastals-userfactor-w0-"+w0+".txt"));
			
			for (int iter = 0; iter < maxIter; iter++) {
				Long start = System.currentTimeMillis();

				// Update user latent vectors
				for (int u = 0; u < userCount; u++) {
					update_user(u);
				}

				// Update item latent vectors
				for (int i = 0; i < itemCount; i++) {
					update_item(i);
				}
				
				if (iter >= maxIter - 10) {
					showProgressWithHitUsers(iter, start, testRatings, hituserpath);
//					viewitemscore.write("Iter="+iter+"\n");
					buyitemscore.write("Iter="+iter+"\n");
					userfactor.write("Iter="+iter+"\n");
					for (int n = 0; n < userCount; n++) {
//						ArrayList<Integer> itemViewList = trainSideMatrix.getRowRef(n).indexList();
//						for (int m:itemViewList) {
//							viewitemscore.write(n+"\t"+m+"\t"+predict(n, m)+"\n");
//						}
						ArrayList<Integer> itemBuyList = trainMatrix.getRowRef(n).indexList();
						for (int m:itemBuyList) {
							buyitemscore.write(n+"\t"+m+"\t"+predict(n, m)+"\n");
						}
						userfactor.write(n+"\t");
						for (int k=0; k < factors-1; k++) {
							userfactor.write(U.get(n, k)+",");
						}
						userfactor.write(U.get(n, factors-1)+"\n");
					}
					//
					itemfactor.write("Iter="+iter+"\n");
					for (int m = 0; m < itemCount; m++) {
						itemfactor.write(m+"\t");
						for (int k=0; k < factors-1; k++) {
							itemfactor.write(V.get(m, k)+",");
						}
						itemfactor.write(V.get(m, factors-1)+"\n");
					}
				}
				else if (iter % 20 == 0)
					showProgress(iter, start, testRatings);
					
				// Show loss
				if (showLoss)
					loss_pre = showLoss(iter, start, loss_pre);
			}
			hituserpath.close();
//			viewitemscore.close();
			buyitemscore.close();
			userfactor.close();
			itemfactor.close();
		}
		
//		for (int iter = 0; iter < maxIter; iter++) {
//			Long start = System.currentTimeMillis();
//
//			// Update user latent vectors
//			for (int u = 0; u < userCount; u++) {
//				update_user(u);
//			}
//
//			// Update item latent vectors
//			for (int i = 0; i < itemCount; i++) {
//				update_item(i);
//			}
//
//			// Show progress
//			if ((showProgress) && (iter >= maxIter - 10)) {
//				hituserpath.write("Iter: " + iter + "\n");
//				showProgressWithHitUsers(iter, start, testRatings, hituserpath);
//				hituserpath.write("\n\n");
//			} else if (showProgress)
//				showProgress(iter, start, testRatings);
//			else if ((!showProgress) && ((iter >= maxIter - 10) || (iter % 10 == 0)))
//				showProgress(iter, start, testRatings);
//			// Show loss
//			if (showLoss)
//				loss_pre = showLoss(iter, start, loss_pre);
//			//
//			if (iter == maxIter - 1) {
//				hituserpath.write("user factors:\n");
//				for (int u = 0; u < userCount; u++) {
//					for (int f = 0; f < factors; f++) {
//						hituserpath.write(U.get(u, f)+"\t");
//					}
//					hituserpath.write("\n");
//				}
//				hituserpath.write("item factors:\n");
//				for (int i = 0; i < itemCount; i++) {
//					for (int f = 0; f < factors; f++) {
//						hituserpath.write(V.get(i, f)+"\t");
//					}
//					hituserpath.write("\n");
//				}
//			}
//		} // end for iter
//		hituserpath.close();
	}

	// public void buildModel() {
	// //System.out.println("Run for FastALS. ");
	// double loss_pre = Double.MAX_VALUE;
	// for (int iter = 0; iter < maxIter; iter ++) {
	// Long start = System.currentTimeMillis();
	//
	// // Update user latent vectors
	// for (int u = 0; u < userCount; u ++) {
	// update_user(u);
	// }
	//
	// // Update item latent vectors
	// for (int i = 0; i < itemCount; i ++) {
	// update_item(i);
	// }
	//
	// // Show progress
	// if (showProgress)
	// showProgress(iter, start, testRatings);
	// // Show loss
	// if (showLoss)
	// loss_pre = showLoss(iter, start, loss_pre);
	//
	// } // end for iter
	//
	// }

	// Run model for one iteration
	public void runOneIteration() {
		// Update user latent vectors
		for (int u = 0; u < userCount; u++) {
			update_user(u);
		}

		// Update item latent vectors
		for (int i = 0; i < itemCount; i++) {
			update_item(i);
		}
	}

	protected void update_user(int u) {
		ArrayList<Integer> itemList = trainMatrix.getRowRef(u).indexList();
		if (itemList.size() == 0)
			return; // user has no ratings
		// prediction cache for the user
		for (int i : itemList) {
			prediction_items[i] = predict(u, i);
			rating_items[i] = trainMatrix.getValue(u, i);
			w_items[i] = W.getValue(u, i);
		}

		DenseVector oldVector = U.row(u);
		for (int f = 0; f < factors; f++) {
			double numer = 0, denom = 0;
			// O(K) complexity for the negative part
			for (int k = 0; k < factors; k++) {
				if (k != f)
					numer -= U.get(u, k) * SV.get(f, k);
			}
			// numer *= w0;

			// O(Nu) complexity for the positive part
			for (int i : itemList) {
				prediction_items[i] -= U.get(u, f) * V.get(i, f);
				numer += (w_items[i] * rating_items[i] - (w_items[i] - Wi[i]) * prediction_items[i]) * V.get(i, f);
				denom += (w_items[i] - Wi[i]) * V.get(i, f) * V.get(i, f);
			}
			denom += SV.get(f, f) + reg;

			// Parameter Update
			U.set(u, f, numer / denom);

			// Update the prediction cache
			for (int i : itemList)
				prediction_items[i] += U.get(u, f) * V.get(i, f);
		} // end for f

		// Update the SU cache
		for (int f = 0; f < factors; f++) {
			for (int k = 0; k <= f; k++) {
				double val = SU.get(f, k) - oldVector.get(f) * oldVector.get(k) + U.get(u, f) * U.get(u, k);
				SU.set(f, k, val);
				SU.set(k, f, val);
			}
		} // end for f
	}

	protected void update_item(int i) {
		ArrayList<Integer> userList = trainMatrix.getColRef(i).indexList();
		if (userList.size() == 0)
			return; // item has no ratings.
		// prediction cache for the item
		for (int u : userList) {
			prediction_users[u] = predict(u, i);
			rating_users[u] = trainMatrix.getValue(u, i);
			w_users[u] = W.getValue(u, i);
		}

		DenseVector oldVector = V.row(i);
		for (int f = 0; f < factors; f++) {
			// O(K) complexity for the w0 part
			double numer = 0, denom = 0;
			for (int k = 0; k < factors; k++) {
				if (k != f)
					numer -= V.get(i, k) * SU.get(f, k);
			}
			numer *= Wi[i];

			// O(Ni) complexity for the positive ratings part
			for (int u : userList) {
				prediction_users[u] -= U.get(u, f) * V.get(i, f);
				numer += (w_users[u] * rating_users[u] - (w_users[u] - Wi[i]) * prediction_users[u]) * U.get(u, f);
				denom += (w_users[u] - Wi[i]) * U.get(u, f) * U.get(u, f);
			}
			denom += Wi[i] * SU.get(f, f) + reg;

			// Parameter update
			V.set(i, f, numer / denom);
			// Update the prediction cache for the item
			for (int u : userList)
				prediction_users[u] += U.get(u, f) * V.get(i, f);
		} // end for f

		// Update the SV cache
		for (int f = 0; f < factors; f++) {
			for (int k = 0; k <= f; k++) {
				double val = SV.get(f, k) - oldVector.get(f) * oldVector.get(k) * Wi[i]
						+ V.get(i, f) * V.get(i, k) * Wi[i];
				SV.set(f, k, val);
				SV.set(k, f, val);
			}
		}
	}

	public double showLoss(int iter, long start, double loss_pre) {
		long start1 = System.currentTimeMillis();
		double loss_cur = loss();
		String symbol = loss_pre >= loss_cur ? "-" : "+";
		System.out.printf("Iter=%d [%s]\t [%s]loss: %.4f [%s]\n", iter, Printer.printTime(start1 - start), symbol,
				loss_cur, Printer.printTime(System.currentTimeMillis() - start1));
		return loss_cur;
	}

	// Fast way to calculate the loss function
	public double loss() {
		double L = reg * (U.squaredSum() + V.squaredSum());
		for (int u = 0; u < userCount; u++) {
			double l = 0;
			for (int i : trainMatrix.getRowRef(u).indexList()) {
				double pred = predict(u, i);
				l += W.getValue(u, i) * Math.pow(trainMatrix.getValue(u, i) - pred, 2);
				l -= Wi[i] * Math.pow(pred, 2);
			}
			l += SV.mult(U.row(u, false)).inner(U.row(u, false));
			L += l;
		}

		return L;
	}

	@Override
	public double predict(int u, int i) {
		return U.row(u, false).inner(V.row(i, false));
	}

	@Override
	public void updateModel(int u, int i) {
		trainMatrix.setValue(u, i, 1);
		W.setValue(u, i, w_new);
		if (Wi[i] == 0) { // an new item
			Wi[i] = w0 / itemCount;
			// Update the SV cache
			for (int f = 0; f < factors; f++) {
				for (int k = 0; k <= f; k++) {
					double val = SV.get(f, k) + V.get(i, f) * V.get(i, k) * Wi[i];
					SV.set(f, k, val);
					SV.set(k, f, val);
				}
			}
		}

		for (int iter = 0; iter < maxIterOnline; iter++) {
			update_user(u);

			update_item(i);
		}
	}

	/*
	 * // Raw way to calculate the loss function public double loss() { double L =
	 * reg * (U.squaredSum() + V.squaredSum()); for (int u = 0; u < userCount; u ++)
	 * { double l = 0; for (int i : trainMatrix.getRowRef(u).indexList()) { l +=
	 * Math.pow(trainMatrix.getValue(u, i) - predict(u, i), 2); } l *= (1 - w0); for
	 * (int i = 0; i < itemCount; i ++) { l += w0 * Math.pow(predict(u, i), 2); } L
	 * += l; } return L; }
	 */
}
