package LinearRegression;

import Jama.Matrix;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

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
       // Matrix closedBeta = getClosedBeta(train_x, train_y);
       // Matrix bgdBeta = getbgdBeta(train_x, train_y);
        Matrix sgdBeta = getsgdBeta(train_x, train_y);
        // 2) predict y for test data using beta calculated from train data
        //Matrix predictedClosedY = modifiedX(test_x).times(closedBeta);
        //Matrix predictedbgdY = test_x.times(bgdBeta);
        Matrix predictedsgdY = test_x.times(sgdBeta);
        // Output
        //printClosedOutput(predictedClosedY);
        //printbgdOutput(predictedbgdY);
        printsgdOutput(predictedsgdY);
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
    
    private static Matrix getsgdBeta(Matrix trainX, Matrix trainY) {
    	int nCols = trainX.getColumnDimension();
    	double eta = 0.0001;
    	Matrix oldBeta = new Matrix(1, nCols);
    	Matrix newBeta = new Matrix(1,nCols);
    		for(int c=0; c<nCols; c++){
    			oldBeta.set(0, c, 1);
    			newBeta.set(0, c, 1);
    		}
    	 int n = 0;
    	 while(true) {
    		 oldBeta = newBeta;
    		 newBeta = gradientSBeta(trainX,trainY, oldBeta, eta);
    		 if(compareBetaEqual(oldBeta, newBeta)) {
    			 break;
    		 }
    		n++;
    		System.out.println(n);
    		if (n == 5000) {
    			//return newBeta.transpose();
    			break;
    		}
    	}
    	return newBeta.transpose(); 
    }
    
    private static Matrix gradientSBeta(Matrix trainX, Matrix trainY, Matrix beta, double eta){
    	int nRows = trainX.getRowDimension();
    	int nCols = trainX.getColumnDimension();
    	Matrix xi = new Matrix(1, nCols);
    	Matrix bi = new Matrix(1, nCols);
    	double yi = 0;
//    	Matrix sumXi = new Matrix(1, nCols);
//    	for(int r=0; r<nRows; r++){
//			yi.set(r, 0, 0);
//		}
    	for (int r=0;r<nRows;r++){
    		for(int c=0;c<nCols;c++){
    			xi.set(0, c, trainX.get(r, c));  			
    			bi.set(0, c, beta.get(0, c));
    		}
    		yi = trainY.get(r, 0);
    		Matrix calMat = bi.times(xi.transpose());
    		double calVal = calMat.get(0, 0);
    		calVal =yi-calVal;
    		bi.plusEquals(xi.times(2 * eta * calVal));
    	}
    	return bi;
    } 
    private static boolean compareBetaEqual(Matrix oldBeta, Matrix newBeta){
    	boolean test = true;
    	int nCols = oldBeta.getColumnDimension();
    	for(int c=0; c<nCols; c++){
    		double oldB =oldBeta.get(0, c);
    		double newB = newBeta.get(0, c);
    		if(Math.abs(oldB - newB) < 0.0000001) {
    			test = test & true;
    		}
    		else {
    			test = test & false;
    		}
    	}
    	return test;
    }
    private static Matrix getClosedBeta(Matrix trainX, Matrix trainY) {
    	Matrix xT = modifiedX(trainX).transpose();
    	Matrix result = xT.times(modifiedX(trainX)).inverse().times(xT.times(trainY));
    	return result;
    	}
    private static Matrix getbgdBeta(Matrix trainX, Matrix trainY) {
    	/****************Please Fill Missing Lines Here*****************/
    	//get batch gradient descent beta
    	int nCols = trainX.getColumnDimension();
    	double eta = 0.00001;
    	Matrix oldBeta = new Matrix(1, nCols);
    		for(int c=0; c<nCols; c++){
    			oldBeta.set(0, c, 1);
    		}
    	Matrix newBeta = new Matrix(1,nCols);
    	for(int c=0; c<nCols; c++){
			newBeta.set(0, c, 1);
		}
    	int n = 0;
    	 while(true) {
    		oldBeta = newBeta;
    		Matrix sumXi = gradientBeta(trainX,trainY, oldBeta);
    		newBeta = oldBeta.minus(sumXi.timesEquals(eta));
    		if(compareBeta(oldBeta, newBeta)) {
    			break;
    		}
    		n++;
    		if (n == 5000) {
    			break;
    		}
    	}
    	return newBeta.transpose(); 	
    }
    
    private static boolean compareBeta(Matrix oldBeta, Matrix newBeta){
    	boolean test = true;
    	int nCols = oldBeta.getColumnDimension();
    	for(int c=0; c<nCols; c++){
    		double oldB =oldBeta.get(0, c);
    		double newB = newBeta.get(0, c);
    		if( Math.abs(oldB - newB) < 0.0001 ) {
    			test = test & true;
    		}
    		else {
    			test = test & false;
    		}
    	}
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
//    		xi.print(0, 4);
//    		bi.print(0, 2);
    		
    		yi = trainY.get(r, 0);
//    		System.out.println(yi);
    		Matrix calMat = bi.times(xi.transpose());
//    		calMat.print(0,  4);
    		double calVal = calMat.get(0, 0);
//    		System.out.println(calVal);
    		calVal -= yi;
//    		System.out.println(calVal);
//    		System.out.println("Xi");
//    		xi.times(2 * calVal).print(0, 4);
    		sumXi.plusEquals(xi.times(2 * calVal));
//    		sumXi.print(0, 4);
    	}
//    	sumXi.print(0, 4);
    	return sumXi;
    } 
    
    /**
     * @params: predicted Y matrix
     * outputs the predicted y values to the text file named "linear-regression-output"
     */
    public static void printClosedOutput(Matrix predictedY) throws IOException {
        FileWriter fStream = new FileWriter("/Users/mounika/Documents/workspace/DataMiningHW1/output/linear_regression/linear-regression-outputClosed.txt");     // Output File
        BufferedWriter out = new BufferedWriter(fStream);
        for (int row =0; row<predictedY.getRowDimension(); row++) {
            out.write(String.valueOf(predictedY.get(row, 0)));
            out.newLine();
        }
        out.close();
    }
    
    public static void printbgdOutput(Matrix predictedY) throws IOException {
        FileWriter fStream = new FileWriter("/Users/mounika/Documents/workspace/DataMiningHW1/output/linear_regression/linear-regression-outputbgd.txt");     // Output File
        BufferedWriter out = new BufferedWriter(fStream);
        for (int row =0; row<predictedY.getRowDimension(); row++) {
            out.write(String.valueOf(predictedY.get(row, 0)));
            out.newLine();
        }
        out.close();
    }
    
    public static void printsgdOutput(Matrix predictedY) throws IOException {
        FileWriter fStream = new FileWriter("/Users/mounika/Documents/workspace/DataMiningHW1/output/linear_regression/linear-regression-outputsgd.txt");     // Output File
        BufferedWriter out = new BufferedWriter(fStream);
        for (int row =0; row<predictedY.getRowDimension(); row++) {
            out.write(String.valueOf(predictedY.get(row, 0)));
            out.newLine();
        }
        out.close();
    }
}
