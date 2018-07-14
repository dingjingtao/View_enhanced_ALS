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
 * View-enhanced eALS for weighted matrix factorization
 * 
 * @author jingtaoding
 */
public class MF_VALS extends TopKRecommender {
	/** Model priors to set. */
	int factors = 10; // number of latent factors.
	int maxIter = 500; // maximum iterations.
	double reg = 0.01; // regularization parameters
	double w0 = 1;
	double init_mean = 0; // Gaussian mean for init V
	double init_stdev = 0.01; // Gaussian std-dev for init V
	double gamma1 = 0;
	double gamma2 = 0;
	/** The number of users. */
	public int userCount_side;
	/** The number of items. */
	public int itemCount_side;

	public SparseMatrix trainSideMatrix;

	/** Model parameters to learn */
	public DenseMatrix U; // latent vectors for users
	public DenseMatrix V; // latent vectors for items

	/** Caches */
	double[] Cu;
	double[] Vu;
	double[] Ru;
	double[] GR;
	DenseVector GvR;
	DenseVector LvR;
	DenseVector T;
	DenseVector Told;
	DenseVector DU;
	DenseMatrix EU;
	DenseMatrix HU;
	DenseVector DV;
	DenseMatrix EV;
	DenseMatrix HV;

	double[] prediction_users, prediction_items;
	double[] rating_users, rating_items;
	double[] prediction_users_side, prediction_items_side;
	double[] rating_users_side, rating_items_side;
	double[] w_users, w_items;
	double[] w_users_side, w_items_side;

	boolean showProgress;
	boolean showLoss;

	// weight for each positive instance in trainMatrix
	SparseMatrix W;

	// weight for negative instances on item i.
	double[] Si;
	double[] Ci;

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

	public MF_VALS(SparseMatrix trainMatrix, ArrayList<Rating> testRatings, SparseMatrix trainSideMatrix,
			int topK, int threadNum, int factors, int maxIter, double w0, double w1, double alpha, double reg,
			double beta, double gamma1, double gamma2, double init_mean, double init_stdev, boolean showProgress,
			boolean showLoss) {
		super(trainMatrix, testRatings, topK, threadNum);
		this.trainSideMatrix = new SparseMatrix(trainSideMatrix);
		this.userCount_side = trainSideMatrix.length()[0];
		this.itemCount_side = trainSideMatrix.length()[1];
		this.factors = factors;
		this.maxIter = maxIter;
		this.w0 = w0;
		this.reg = reg;
		this.gamma1 = gamma1;
		this.gamma2 = gamma2;
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
		Si = new double[itemCount];
		double Wimin = 0, Wimax = 0, N0 = 0;
		for (int i = 0; i < itemCount; i++) {
			Si[i] = w0 * p[i] / Z;
			if (Si[i] < Wimin)
				Wimin = Si[i];
			if (Si[i] > Wimax)
				Wimax = Si[i];
			if (Si[i] == 0)
				N0++;
		}
		

		double[] pv = new double[itemCount];

		double sum1 = 0, Z1 = 0;
		for (int i = 0; i < itemCount; i++) {
			pv[i] = trainSideMatrix.getColRef(i).itemCount();
			sum1 += pv[i];
		}
		// convert pv[i] to probability
		for (int i = 0; i < itemCount; i++) {
			pv[i] /= sum1;
			if (pv[i] > 0)
				pv[i] = Math.pow(pv[i], beta);
			Z1 += pv[i];
		}
		// assign weight
		Ci = new double[itemCount];
		for (int i = 0; i < itemCount; i++) {
			Ci[i] = w1 * pv[i] / Z1;
		}
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
		prediction_users_side = new double[userCount_side];
		prediction_items_side = new double[itemCount_side];
		w_users = new double[userCount];
		w_items = new double[itemCount];
		w_users_side = new double[userCount_side];
		w_items_side = new double[itemCount_side];

		// Init model parameters
		U = new DenseMatrix(userCount, factors);
		V = new DenseMatrix(itemCount, factors);
		U.init(init_mean, init_stdev);
		V.init(init_mean, init_stdev);
		initCache();
	}

	public void setTrain(SparseMatrix trainMatrix) {
		this.trainMatrix = new SparseMatrix(trainMatrix);
		W = new SparseMatrix(userCount, itemCount);
		for (int u = 0; u < userCount; u++)
			for (int i : this.trainMatrix.getRowRef(u).indexList())
				W.setValue(u, i, 1);
	}

	// Init Cu,GvR,LvR,T,DU,EU,HU,DV,EV,HV
	private void initCache() {
		Cu = new double[userCount];
		Vu = new double[userCount];
		Ru = new double[userCount];
		GR = new double[userCount];
		GvR = new DenseVector(userCount);
		LvR = new DenseVector(userCount);
		for (int u = 0; u < userCount; u++) {
			double val1 = 0;
			double val2 = 0;
			for (int i : trainSideMatrix.getRowRef(u).indexList()) {
				Cu[u] += Ci[i];
				Vu[u] += 1;
				val1 += predict(u, i);
				val2 += Ci[i] * predict(u, i);
			}
			Ru[u] = trainMatrix.getRowRef(u).indexList().size();
			GvR.set(u, val1);
			LvR.set(u, val2);
		}
		T = U.transpose().mult(LvR);
		Told = T.clone();
		//
		DU = new DenseVector(factors);
		EU = U.transpose().mult(U);
		HU = new DenseMatrix(factors, factors);
		for (int f = 0; f < factors; f++) {
			double val1 = 0;
			for (int u = 0; u < userCount; u++)
				val1 += U.get(u, f) * Cu[u];
			DU.set(f, val1);
			for (int k = 0; k <= f; k++) {
				double val = 0;
				for (int u = 0; u < userCount; u++)
					val += U.get(u, f) * U.get(u, k) * Cu[u];
				HU.set(f, k, val);
				HU.set(k, f, val);
			}
		}
		//
		DV = new DenseVector(factors);
		EV = V.transpose().mult(V);
		// Init SV as V^T Wi V
		HV = new DenseMatrix(factors, factors);
		for (int f = 0; f < factors; f++) {
			double val1 = 0;
			for (int i = 0; i < itemCount; i++)
				val1 += V.get(i, f);
			DV.set(f, val1);
			for (int k = 0; k <= f; k++) {
				double val = 0;
				for (int i = 0; i < itemCount; i++)
					val += V.get(i, f) * V.get(i, k) * Si[i];
				HV.set(f, k, val);
				HV.set(k, f, val);
			}
		}
	}

	// remove
	public void setUV(DenseMatrix U, DenseMatrix V) {
		this.U = U.clone();
		this.V = V.clone();
		initCache();
	}

	public void buildModel() throws IOException {
		// System.out.println("Run for FastALS. ");
		double loss_pre = Double.MAX_VALUE;
//		ArrayList<Double> loss_pre = new ArrayList<Double>();
//		loss_pre.add(Double.MAX_VALUE);
//		loss_pre.add(Double.MAX_VALUE);
//		if (!showHit) {
//			if (showProgress)
//				showProgress(iter, start, testRatings);
//			else if ((!showProgress) && ((iter >= maxIter - 10)||(iter % 20 == 0)))
//				showProgress(iter, start, testRatings);
//		}
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
			BufferedWriter hituserpath = new BufferedWriter(new FileWriter("eals+view-hitusers.txt"));
			BufferedWriter viewitemscore = new BufferedWriter(new FileWriter("eals+view-viewitemscore-w0-"+w0+"-gamma-"+gamma1+"-"+gamma2+".txt"));
			BufferedWriter buyitemscore = new BufferedWriter(new FileWriter("eals+view-buyitemscore-w0-"+w0+"-gamma-"+gamma1+"-"+gamma2+".txt"));
			BufferedWriter itemfactor = new BufferedWriter(new FileWriter("eals+view-itemfactor-w0-"+w0+"-gamma-"+gamma1+"-"+gamma2+".txt"));
			BufferedWriter userfactor = new BufferedWriter(new FileWriter("eals+view-userfactor-w0-"+w0+"-gamma-"+gamma1+"-"+gamma2+".txt"));
			
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
					viewitemscore.write("Iter="+iter+"\n");
					buyitemscore.write("Iter="+iter+"\n");
					userfactor.write("Iter="+iter+"\n");
					for (int n = 0; n < userCount; n++) {
						ArrayList<Integer> itemViewList = trainSideMatrix.getRowRef(n).indexList();
						for (int m:itemViewList) {
							viewitemscore.write(n+"\t"+m+"\t"+predict(n, m)+"\n");
						}
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
			viewitemscore.close();
			buyitemscore.close();
			userfactor.close();
			itemfactor.close();
		}
		
	}


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
		ArrayList<Integer> itemViewList = trainSideMatrix.getRowRef(u).indexList();
		if (itemList.size() == 0)
			return; // user has no ratings
		// prediction cache for the user
		for (int i : itemList) {
			prediction_items[i] = predict(u, i);
			rating_items[i] = trainMatrix.getValue(u, i);
			w_items[i] = W.getValue(u, i);
		}
		DenseVector oldVector1 = U.row(u);

		for (int i : itemViewList) {
			prediction_items[i] = predict(u, i);
		}

		for (int f = 0; f < factors; f++) {
			double numer = 0, denom = 0;
			double pd = 0;
			// O(K) complexity for the negative part
			for (int k = 0; k < factors; k++) {
				if (k != f) {
					numer += U.get(u, k) * HV.get(f, k) + Cu[u] * U.get(u, k) * EV.get(f, k);
					pd += U.get(u, k) * DV.get(k);

				}
			}
			// numer *= w0;

			// O(Nu) complexity for the positive part
			for (int i : itemList) {
				prediction_items[i] -= U.get(u, f) * V.get(i, f);
				numer += -(w_items[i] * rating_items[i] - (w_items[i] - Si[i]) * prediction_items[i]) * V.get(i, f)
						- (gamma1 + gamma2) * Cu[u] * V.get(i, f);
				denom += (w_items[i] - Si[i]) * V.get(i, f) * V.get(i, f);
			}
			denom += HV.get(f, f) + reg;
			double cq = 0, r = 0, cr = 0, q = 0;
			for (int i : itemViewList) {
				prediction_items[i] -= U.get(u, f) * V.get(i, f);
				numer += -Si[i] * V.get(i, f) * prediction_items[i]
						+ (itemCount - Vu[u]) * Ci[i] * V.get(i, f) * prediction_items[i]
						+ ((gamma1 + gamma2) * Ru[u] - gamma2 * (itemCount - Vu[u])) * Ci[i] * V.get(i, f)
						- Cu[u] * V.get(i, f) * prediction_items[i] - gamma2 * Cu[u] * V.get(i, f)
						- pd * Ci[i] * V.get(i, f) - Ci[i] * prediction_items[i] * DV.get(f);
				cq += Ci[i] * V.get(i, f);
				r += prediction_items[i];
				cr += Ci[i] * prediction_items[i];
				q += V.get(i, f);

				denom += - Si[i] * V.get(i, f) * V.get(i, f) + (itemCount - Vu[u]) * Ci[i] * V.get(i, f) * V.get(i, f)
						- Cu[u] * V.get(i, f) * V.get(i, f) - 2 * Ci[i] * V.get(i, f) * DV.get(f);
			}
			numer += cq * r + cr * q + gamma2 * Cu[u] * DV.get(f);
			denom += Cu[u] * EV.get(f, f) + 2 * cq * q;
			// Parameter Update MOdified!!!
			U.set(u, f, -numer / denom);

			// Update the prediction cache
			for (int i : itemList)
				prediction_items[i] += U.get(u, f) * V.get(i, f);
			for (int i : itemViewList)
				prediction_items[i] += U.get(u, f) * V.get(i, f);
		} // end for f
		double tmp1 = 0, tmp2 = 0;
		for (int i : itemViewList) {
			tmp1 += prediction_items[i];
			tmp2 += Ci[i] * prediction_items[i];
		}
		GvR.set(u, tmp1);
		LvR.set(u, tmp2);

		// Update the SU cache
		for (int f = 0; f < factors; f++) {
			double val = U.get(u, f) * LvR.get(u);
			if (u == 0) {
				T.set(f, val);
				Told.set(f, val);
			} else {
				T.set(f, T.get(f) + val);
				Told.set(f, Told.get(f) + val);
			}
			double val0 = DU.get(f) - oldVector1.get(f) * Cu[u] + U.get(u, f) * Cu[u];
			DU.set(f, val0);
			for (int k = 0; k <= f; k++) {
				double val1 = EU.get(f, k) - oldVector1.get(f) * oldVector1.get(k) + U.get(u, f) * U.get(u, k);
				EU.set(f, k, val1);
				EU.set(k, f, val1);
				double val2 = HU.get(f, k) - oldVector1.get(f) * oldVector1.get(k) * Cu[u]
						+ U.get(u, f) * U.get(u, k) * Cu[u];
				HU.set(f, k, val2);
				HU.set(k, f, val2);
			}
		} // end for f
	}

	protected void update_item(int i) {
		ArrayList<Integer> userList = trainMatrix.getColRef(i).indexList();
		ArrayList<Integer> userViewList = trainSideMatrix.getColRef(i).indexList();
		int Ri = userList.size();
		int Vi = userViewList.size();
		if (userList.size() == 0)
			return; // item has no ratings.
		// prediction cache for the item
		for (int u : userList) {
			prediction_users[u] = predict(u, i);
			rating_users[u] = trainMatrix.getValue(u, i);
			w_users[u] = W.getValue(u, i);
		}
		int[] ind_u = new int[Vi];
		int cnt = 0;
		double[] GRold = new double[Vi];
		double[] GvRold = new double[Vi];
		double[] LvRold = new double[Vi];
		double DVold = 0;
		
		for (int u : userViewList) {
			ind_u[cnt] = u;
			cnt += 1;
			prediction_users[u] = predict(u, i);
			GR[u] = 0;
			for (int k = 0; k < factors; k++) {
				GR[u] += U.get(u, k) * DV.get(k);
				// if (i == 0) {
				// Told.set(k, Told.get(k) - 2 * Ci[i] * prediction_users[u] * U.get(u, k));
				// }
				Told.set(k, Told.get(k) - Ci[i] * U.get(u, k) * prediction_users[u]);
			}

		}
		DenseVector oldVector1 = V.row(i);

		for (int f = 0; f < factors; f++) {
			for (int n = 0; n < Vi; n++) {
				GRold[n] = GR[ind_u[n]] - U.get(ind_u[n], f) * DV.get(f);
				GvRold[n] = GvR.get(ind_u[n]) - prediction_users[ind_u[n]];
				LvRold[n] = LvR.get(ind_u[n]) - Ci[i] * prediction_users[ind_u[n]];
			}
			DVold = DV.get(f) - V.get(i, f);
			// O(K) complexity for the w0 part
			double numer = 0, denom = 0;
			for (int k = 0; k < factors; k++) {
				if (k != f)
					numer += V.get(i, k) * EU.get(f, k) * Si[i] + V.get(i, k) * HU.get(f, k);
			}
			// numer *= Si[i];

			// O(Ni) complexity for the positive ratings part
			for (int u : userList) {
				prediction_users[u] -= U.get(u, f) * V.get(i, f);
				numer += -(w_users[u] * rating_users[u] - (w_users[u] - Si[i]) * prediction_users[u]) * U.get(u, f)
						- (gamma1 + gamma2) * Cu[u] * U.get(u, f);
				denom += (w_users[u] - Si[i]) * U.get(u, f) * U.get(u, f);
			}
			denom += Si[i] * EU.get(f, f) + HU.get(f, f) + reg;
			for (int u : userViewList) {
				prediction_users[u] -= U.get(u, f) * V.get(i, f);
				numer += -Si[i] * prediction_users[u] * U.get(u, f) - Ci[i] * GR[u] * U.get(u, f)
						+ Ci[i] * GvR.get(u) * U.get(u, f) + LvR.get(u) * U.get(u, f)
						+ Ci[i] * (prediction_users[u] * (itemCount - Vu[u]) + (gamma1 + gamma2) * Ru[u]
								- gamma2 * (itemCount - Vu[u])) * U.get(u, f)
						- (prediction_users[u] + gamma2) * Cu[u] * U.get(u, f);
				denom += -Si[i] * U.get(u, f) * U.get(u, f) + Ci[i] * (itemCount - Vu[u]) * U.get(u, f) * U.get(u, f)
						- Cu[u] * U.get(u, f) * U.get(u, f);
			}
			numer += -T.get(f) + gamma2 * DU.get(f);
			// Parameter update MOdified!!!
			V.set(i, f, - numer / denom);
			// Update the prediction cache for the item
			for (int u : userList)
				prediction_users[u] += U.get(u, f) * V.get(i, f);
			for (int u : userViewList)
				prediction_users[u] += U.get(u, f) * V.get(i, f);
			double tf = 0, tfp1 = 0;
			int fp1 = f + 1;
			if (fp1 >= factors) {
				fp1 = 0;
			}
			DV.set(f, DVold + V.get(i, f));
			for (int n = 0; n < Vi; n++) {
				GR[ind_u[n]] = GRold[n] + U.get(ind_u[n], f) * DV.get(f);
				GvR.set(ind_u[n], GvRold[n] + prediction_users[ind_u[n]]);
				LvR.set(ind_u[n], LvRold[n] + Ci[i] * prediction_users[ind_u[n]]);
				tf += Ci[i] * U.get(ind_u[n], f) * prediction_users[ind_u[n]];
				tfp1 += Ci[i] * U.get(ind_u[n], fp1) * prediction_users[ind_u[n]];
			}
			Told.set(f, T.get(f) - tf);
			T.set(fp1, Told.get(fp1) + tfp1);

		} // end for f

		for (int u : userViewList) {
			for (int k = 0; k < factors; k++) {
				Told.set(k, Told.get(k) + Ci[i] * U.get(u, k) * prediction_users[u]);
			}
		}
		// Update the SV cache
		for (int f = 0; f < factors; f++) {
//			double val0 = DV.get(f) - oldVector1.get(f) + V.get(i, f);
//			DV.set(f, val0);
			for (int k = 0; k <= f; k++) {
				double val1 = EV.get(f, k) - oldVector1.get(f) * oldVector1.get(k) + V.get(i, f) * V.get(i, k);
				EV.set(f, k, val1);
				EV.set(k, f, val1);
				double val2 = HV.get(f, k) - oldVector1.get(f) * oldVector1.get(k) * Si[i]
						+ V.get(i, f) * V.get(i, k) * Si[i];
				HV.set(f, k, val2);
				HV.set(k, f, val2);
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
				l -= Si[i] * Math.pow(pred, 2);
				l -= 2 * Cu[u] * (gamma1 + gamma2) * pred;
			}
			for (int i : trainSideMatrix.getRowRef(u).indexList()) {
				double pred = predict(u, i);
				l -= (Si[i] + Cu[u]) * Math.pow(pred, 2);
				l += (itemCount - Vu[u]) * Ci[i] * Math.pow(pred, 2);
				// l -= Cu[u] * Math.pow(pred, 2);
			}
			for (int k = 0; k < factors; k++) {
				l += (2 * gamma2 * Cu[u] - 2 * LvR.get(u)) * U.get(u, k) * DV.get(k);
			}
			l += HV.mult(U.row(u, false)).inner(U.row(u, false));
			l += Cu[u] * (Math.pow(gamma1, 2) - Math.pow(gamma2, 2)) * Ru[u] + 2 * (gamma1 + gamma2) * Ru[u] * LvR.get(u)
					- 2 * Cu[u] * gamma2 * GvR.get(u) + 2 * GvR.get(u) * LvR.get(u)
					+ Cu[u] * (itemCount - Vu[u]) * Math.pow(gamma2, 2) - 2 * gamma2 * (itemCount - Vu[u]) * LvR.get(u);
			l += Cu[u] * EV.mult(U.row(u, false)).inner(U.row(u, false));
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
		if (Si[i] == 0) { // an new item
			Si[i] = w0 / itemCount;
			// Update the SV cache
			for (int f = 0; f < factors; f++) {
				for (int k = 0; k <= f; k++) {
					double val = HV.get(f, k) + V.get(i, f) * V.get(i, k) * Si[i];
					HV.set(f, k, val);
					HV.set(k, f, val);
				}
			}
		}

		for (int iter = 0; iter < maxIterOnline; iter++) {
			update_user(u);

			update_item(i);
		}
	}

	// Raw way to calculate the loss function
	public double loss_slow() {
		double L = reg * (U.squaredSum() + V.squaredSum());
		for (int u = 0; u < userCount; u++) {
			double l = 0;
			for (int i : trainMatrix.getRowRef(u).indexList()) {
				l += Math.pow(trainMatrix.getValue(u, i) - predict(u, i), 2) - Si[i] * Math.pow(predict(u, i), 2);
			}
			for (int i : trainSideMatrix.getRowRef(u).indexList()) {
				l -= Si[i] * Math.pow(predict(u, i), 2);
				for (int j : trainMatrix.getRowRef(u).indexList()) {
					l += Ci[i] * (Math.pow(gamma1 - (predict(u, j) - predict(u, i)), 2)
							- Math.pow(gamma2 - (predict(u, i) - predict(u, j)), 2));
				}
				for (int j : trainSideMatrix.getRowRef(u).indexList()) {
					l -= Ci[i] * Math.pow(gamma2 - (predict(u, i) - predict(u, j)), 2);
				}
				for (int j = 0; j < itemCount; j++) {
					l += Ci[i] * Math.pow(gamma2 - (predict(u, i) - predict(u, j)), 2);
				}
			}
			// l *= (1 - w0);
			for (int i = 0; i < itemCount; i++) {
				l += Si[i] * Math.pow(predict(u, i), 2);
			}
			L += l;
		}
		return L;
	}
}
