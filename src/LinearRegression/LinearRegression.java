package LinearRegression;

import Jama.Matrix;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;

/**
 * Simple Linear Regression implementation
 */
public class LinearRegression {
    public static void linearRegression() throws Exception {
//        Matrix trainingData = MatrixData.getDataMatrix("/Users/mounika/Documents/workspace/DataMiningHW1/src/LinearRegression/TestAlgo.java");
        Matrix trainingData = MatrixData.getDataMatrix("/Users/mounika/Documents/workspace/DataMiningHW1/data/linear_regression/linear-regression-train.csv");
        // getMatrix(Initial row index, Final row index, Initial column index, Final column index)
        
        Matrix train_x = trainingData.getMatrix(0, trainingData.getRowDimension() - 1, 0, trainingData.getColumnDimension() - 2);
        Matrix train_y = trainingData.getMatrix(0, trainingData.getRowDimension()-1, trainingData.getColumnDimension()-1, trainingData.getColumnDimension()-1);

        Matrix testData = MatrixData.getDataMatrix("/Users/mounika/Documents/workspace/DataMiningHW1/data/linear_regression/linear-regression-test.csv");
        Matrix test_x = testData.getMatrix(0, testData.getRowDimension() - 1, 0, testData.getColumnDimension() - 2);

        /* Linear Regression */
        /* 2 step process */
        // 1) find beta
        Matrix beta = getBeta(train_x, train_y);
        // 2) predict y for test data using beta calculated from train data
//        Matrix predictedY = modifiedX(test_x).times(beta);
        Matrix predictedY = test_x.times(beta);
        // Output
        printOutput(predictedY);
        
        System.out.println("Done");
    }
    
    public static Matrix modifiedX(Matrix matX){
       	int nRows = matX.getRowDimension();
    	int nCols = matX.getColumnDimension();
    	
    	Matrix newX = new Matrix(nRows, nCols+1);
    	
    	// Set the first column to 1
    	for (int r=0; r<nRows; r++){
    		newX.set(r, 0, 1);	
    	}

    	// Copy existing matrix to new matrix from column 1
        for (int r=0; r<nRows; r++) {
            for(int c=0; c<nCols; c++){
            	double orgElem = matX.get(r, c);
            	newX.set(r, c+1, orgElem);
            }
        }
        //System.out.print(newX.get(0, 1));
        return newX;
    }
    /**  @params: X and Y matrix of training data
     * returns value of beta calculated using the formula beta = (X^T*X)^ -1)*(X^T*Y)
     */
    private static Matrix getBeta(Matrix trainX, Matrix trainY) {
    	/****************Please Fill Missing Lines Here*****************/
    	//get batch gradient descent beta
    	int nCols = trainX.getColumnDimension();
    	double eta = 0.005;
    	Matrix oldBeta = new Matrix(1, nCols);
    		for(int c=0; c<nCols; c++){
    			oldBeta.set(0, c, 1);
    		}
    	Matrix newBeta = new Matrix(1,nCols);
    	int n = 0;
//    	while(n<1){
         while(!compareBeta(oldBeta,newBeta)){
    		oldBeta = newBeta;
    		Matrix calMat = gradientBeta(trainX,trainY, oldBeta);
    		newBeta = oldBeta.minus(calMat.timesEquals(eta));
    		n++;
    		System.out.println(n);
    		if (n == 5) {
    			break;
    		}
    	}
    	//return newBeta;
    	//Matrix xT = modifiedX(trainX).transpose();
    	//Matrix result = xT.times(modifiedX(trainX)).inverse().times(xT.times(trainY));
    	
    	return newBeta.transpose(); 	
    }
    
    private static boolean compareBeta(Matrix oldBeta, Matrix newBeta){
    	oldBeta.print(0, 2);
    	newBeta.print(0, 2);
    	boolean test = true;
    	int nCols = oldBeta.getColumnDimension();
    	for(int c=0; c<nCols; c++){
    		double oldB =oldBeta.get(0, c);
    		double newB = newBeta.get(0, c);
    		if( Math.abs(oldB - newB) < 0.001 ) { // abs(old - new) < eplsilon)Math.abs(d - 1.0) < 0.001);
    			test = test & true;
    		}
    		else {
    			test = test & false;
    		}
    	}
    	System.out.println(test);
    	return test;
    }
    
    private static Matrix gradientBeta(Matrix trainX, Matrix trainY, Matrix beta){
    	int nRows = trainX.getRowDimension();
    	int nCols = trainX.getColumnDimension();
    	
    	Matrix xi = new Matrix(1, nCols);
    	Matrix bi = new Matrix(1, nCols);
    	Matrix sumXi = new Matrix(1, nCols);
    	double yi = 0;
    	for(int c=0; c<nCols; c++){
			sumXi.set(0, c, 0);
		}
    	for (int r=0;r<nRows;r++){
    		for(int c=0;c<nCols;c++){
    			xi.set(0, c, trainX.get(r, c));  			
    			bi.set(0, c, beta.get(0, c));
    		}
    		yi = trainY.get(r, 0);
    		Matrix calMat = bi.times(xi.transpose());
    		double calVal = calMat.get(0, 0);
    		calVal -= yi;
    		sumXi.plusEquals(xi.times(2 * calVal));
    	}
    	return sumXi;
    } 
    
//    
//    private static double mean_square_error(Matrix predictedY, Matrix actualY){
//    	double MSE;
//    	Matrix subMat = actualY.minus(predictedY);
//    	for(r)
//    	return MSE;
//    }
////    
//    private static Matrix gradient_descent(Matrix x, Matrix y, Matrix theta, double alpha, int num_iters){
//    	int m = y.getRowDimension();
//    	Matrix xTrans = x.transpose();
//    	
//    	for (int i=0; i<num_iters; i++) {
//    		Matrix hypothesis = x.times(theta);
//    		Matrix loss = hypothesis.minus(y);
//    		// formula for cost = (loss^2)/(2*m)
//    		Matrix cost = squareMat(loss,m);
//   		    Matrix gradient = calculateMat(xTrans, loss, m);
//   		    Matrix temp = matSub(theta, alpha);
//   		    theta = theta.minus(gradient.times(alpha));
//   		    
//   		    // Check for convergence
//   		    // Get new MSE, using trainX, theta
//    	}
//        return theta;
//    }
    //to subtract a scalar from matrix
//    private static Matrix matSub(Matrix A, double alpha){
//    	int nRows = A.getRowDimension();
//    	int nCols = A.getColumnDimension();
//    	for (int r=0; r<nRows; r++){
//    		for(int c=0; c<nCols; c++){
//    			A.set(r, c, A.get(r,c)-alpha);
//    		}
//    	}
//    	return A;
//    }
    
//    private static Matrix squareMat(Matrix A, int m){
//    	int nRows = A.getRowDimension();
//    	int nCols = A.getColumnDimension();
//    	for (int r=0; r<nRows; r++){
//    		for(int c=0; c<nCols; c++){
//    			double calcValue = (A.get(r,c) * A.get(r,c))/(2 * m);
//    			A.set(r,c,calcValue);
//    		}
//    	}
//    return A;
//    }


    /**
     * @params: predicted Y matrix
     * outputs the predicted y values to the text file named "linear-regression-output"
     */
    public static void printOutput(Matrix predictedY) throws IOException {
        FileWriter fStream = new FileWriter("/Users/mounika/Documents/workspace/DataMiningHW1/output/linear_regression/linear-regression-outputGrad.txt");     // Output File
        BufferedWriter out = new BufferedWriter(fStream);
        for (int row =0; row<predictedY.getRowDimension(); row++) {
            out.write(String.valueOf(predictedY.get(row, 0)));
            out.newLine();
        }
        out.close();
    }
}
