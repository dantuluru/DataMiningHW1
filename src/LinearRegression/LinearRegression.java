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
        // Matrix trainingData = MatrixData.getDataMatrix("/Users/mounika/Documents/workspace/DataMiningHW1/src/LinearRegression/TestAlgo.java");
        Matrix trainingData = MatrixData.getDataMatrix("/Users/mounika/Documents/workspace/DataMiningHW1/data/linear_regression/linear-regression-train.csv");
        // getMatrix(Initial row index, Final row index, Initial column index, Final column index)
        
        Matrix train_x = trainingData.getMatrix(0, trainingData.getRowDimension() - 1, 0, trainingData.getColumnDimension() - 2);
        Matrix train_y = trainingData.getMatrix(0, trainingData.getRowDimension()-1, trainingData.getColumnDimension()-1, trainingData.getColumnDimension()-1);

        Matrix testData = MatrixData.getDataMatrix("/Users/mounika/Documents/workspace/DataMiningHW1/data/linear_regression/linear-regression-test.csv");
        Matrix test_x = testData.getMatrix(0, testData.getRowDimension() - 1, 0, testData.getColumnDimension() - 2);

        /* Linear Regression */
        /* 2 step process */
        // 1) find beta
        normalization(train_x,test_x,train_y);
        Matrix closedBeta = getClosedBeta(train_x, train_y);
        Matrix bgdBeta = getbgdBeta(train_x, train_y);
        Matrix sgdBeta = getsgdBeta(train_x, train_y);
        // 2) predict y for test data using beta calculated from train data
        Matrix predictedClosedY = modifiedX(test_x).times(closedBeta);
        Matrix predictedbgdY = test_x.times(bgdBeta);
        Matrix predictedsgdY = test_x.times(sgdBeta);
        mse(predictedClosedY, train_y, "ClosedFormLinearRegression : ");
        mse(predictedbgdY, train_y, "BGD : ");
        mse(predictedsgdY, train_y, "SGD : ");
        // Output
        printClosedOutput(predictedClosedY);
        printbgdOutput(predictedbgdY);
        printsgdOutput(predictedsgdY);
        System.out.println("Done");
    }
    
    /**  @params: X and Y matrix of training data
     * returns value of beta calculated using the formula beta = (X^T*X)^ -1)*(X^T*Y)
     */
    
 // start functions to calculate Closed LR
    private static Matrix getClosedBeta(Matrix trainX, Matrix trainY) {
    	Matrix xT = modifiedX(trainX).transpose();
    	Matrix result = xT.times(modifiedX(trainX)).inverse().times(xT.times(trainY));
    	return result;
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
 // end functions to calculate Closed LR
    // start functions to calculate SGD 
    private static Matrix getsgdBeta(Matrix trainX, Matrix trainY) {
    	int nCols = trainX.getColumnDimension();
    	int nRows = trainX.getRowDimension();
    	double eta = 0.0001;
    	Matrix oldBeta = new Matrix(1, nCols);
    	Matrix newBeta = new Matrix(1,nCols);
    		for(int c=0; c<nCols; c++){
    			oldBeta.set(0, c, 0);
    			newBeta.set(0, c, 1);
    		}
    	 int n = 0;
    	 while(n < 100) {
    	    	Matrix xi = new Matrix(1, nCols);
    	    	Matrix bi = new Matrix(1, nCols);
    	    	Matrix yi = new Matrix(1, 1);
    	    	Matrix calMat = new Matrix(1,1);
    	    	for (int r=0;r<nRows;r++){
    	    		for(int c=0;c<nCols;c++){
    	    			xi.set(0, c, trainX.get(r, c));  			
    	    			bi.set(0, c, oldBeta.get(0, c));
    	    		}
    	    		yi.set(0,0,trainY.get(r, 0));
    	    		calMat.set(0, 0, (bi.times(xi.transpose()).get(0, 0)));
    	    		yi.minusEquals(calMat);
    	    		yi.timesEquals(2*eta);
    	    		newBeta = bi.plus(xi.times(yi.get(0, 0)));
    	    		if(compareBetaEqual(oldBeta,newBeta)) {
    	    			return newBeta.transpose();
    			    	//break;
    		    	}
    		    	else {
    		    		
    		    		oldBeta = newBeta;
    		    	}
    	    	}
    	    	n++;
    	 }
    	 return newBeta.transpose();	
    }

    private static boolean compareBetaEqual(Matrix oldBeta, Matrix newBeta){
    	boolean test = true;
    	int nCols = oldBeta.getColumnDimension();
    	for(int c=0; c<nCols; c++){
    		double oldB =oldBeta.get(0, c);
    		double newB = newBeta.get(0, c);
    		if(oldB == newB) {
    			test = test & true;
    		}
    		else {
    			test = test & false;
    		}
    	}
    	return test;
    }
    // end functions to calculate SGD
    
    // start functions to calculate SGD
    private static Matrix getbgdBeta(Matrix trainX, Matrix trainY) {
    	/****************Please Fill Missing Lines Here*****************/
    	//get batch gradient descent beta
    	int nCols = trainX.getColumnDimension();
    	double eta = 0.000001;
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
    		if (n == 2000) {
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
    		yi = trainY.get(r, 0);
    		Matrix calMat = bi.times(xi.transpose());
    		double calVal = calMat.get(0, 0);
    		calVal -= yi;
    		sumXi.plusEquals(xi.times(2 * calVal));
    	}
    	return sumXi;
    } 
    
 // start functions to calculate SGD
    
    private static void mse(Matrix predictedY, Matrix trainY, String type){
    	double msqerr = 0;
    	int nRows = trainY.getRowDimension();
    	int nCols = trainY.getColumnDimension();
    	for (int r=0; r< nRows; r++){
    		for(int c=0; c< nCols; c++){
    			double trainy = trainY.get(r, c);
    			double predictedy = predictedY.get(r, c);
    			msqerr += (Math.pow((trainy - predictedy), 2))/nRows;
    		}
    	}
    	System.out.print(type);
    	System.out.println(msqerr);
    } 
    
    private static void normalization(Matrix trainX, Matrix testX,Matrix train_y){
    	int nRows = trainX.getRowDimension();
    	int nCols = trainX.getColumnDimension();
    	Matrix zscore_testX = new Matrix(nRows,nCols);
    	Matrix zscore_trainX = new Matrix(nRows,nCols);
    	double nu = 0;
    	double sigma = 0;
    	for(int r=0;r<nRows;r++)
    	{
	    	double nu_sum = 0;
	    	double sig_train_sum = 0;
	    	double sig_test_sum = 0;
	    	double sig_train = 0;
	    	double sig_test = 0;
	    	double sig_train_mean = 0;
	    	double sig_test_mean = 0;
	    	//double mean = 0;
	    	Double train[] = new Double[nRows];
	    	Double test[] = new Double[nRows];
	    	
	    	for(int c=0;c<nCols;c++){
	    		nu_sum += trainX.get(r, c);
	    		}
	    	nu = nu_sum/nRows;
	    
	    	for(int c=0;c<nCols;c++)
	    	{
	    		train[c] = Math.pow(trainX.get(r, c),2) - Math.pow(nu, 2);
	    		//test[c] = Math.pow(testX.get(r, c),2) - Math.pow(nu, 2);
	    		sig_train_sum += train[c];
	    		//sig_test_sum += test[c];
	    	}
	    	sig_train_mean = sig_train_sum/nRows;
	    	//sig_test_mean = sig_test_sum/nRows;
	    	sig_train = Math.sqrt(sig_train_mean);
	    	//sig_test = Math.sqrt(sig_test_mean);
	    	for(int c=0;c<nCols;c++)
	    	{
	    		zscore_trainX.set(r, c, (trainX.get(r, c) - nu) / sig_train);
	    		zscore_testX.set(r, c, (testX.get(r, c) - nu) / sig_test);
	    	}
    	}
    	Matrix closedBeta = getClosedBeta(zscore_trainX, train_y);
        Matrix bgdBeta = getbgdBeta(zscore_trainX, train_y);
        Matrix sgdBeta = getsgdBeta(zscore_trainX, train_y);
        Matrix predictedClosedY = modifiedX(zscore_trainX).times(closedBeta);
        Matrix predictedbgdY = zscore_trainX.times(bgdBeta);
        Matrix predictedsgdY = zscore_trainX.times(sgdBeta);
        mse(predictedClosedY, train_y, "ZscoreClosedFormLinearRegression : ");
        mse(predictedbgdY, train_y, "ZscoreBGD : ");
        mse(predictedsgdY, train_y, "ZscoreSGD : ");
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
