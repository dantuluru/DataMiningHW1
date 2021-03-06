package LinearRegression;

import Jama.Matrix;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

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
        Matrix test_y = testData.getMatrix(0, testData.getRowDimension()-1, testData.getColumnDimension()-1, testData.getColumnDimension()-1);
        /* Linear Regression */
        /* 2 step process */
        // 1) find beta
        
        System.out.println("\tLinear Regression");
        System.out.println("1) Closed-Form Linear Regression");
        System.out.println("2) Batch Gradient Descent");
        System.out.println("3) Stochastic Gradient Descent");
        System.out.println("4) Exit\n");
        System.out.println("Enter the number corresponding to the algorithm you want to run \n *(it may take some time to run):");
        Scanner in = new Scanner(System.in);
        int choice = in.nextInt();
        switch(choice){
            case 1: Matrix closedBeta = getClosedBeta(modifiedX(train_x), train_y);
            		Matrix predictedClosedY = modifiedX(test_x).times(closedBeta);
            		mse(predictedClosedY, test_y, "Closed-FormLinearRegression Mean Square Error: ");
            		Matrix zClosedBeta = getClosedBeta(normalization(modifiedX(train_x),modifiedX(test_x),train_y,test_y, false), train_y);
            		Matrix zPredictedClosedY = normalization(modifiedX(train_x),modifiedX(test_x),train_y,test_y, true).times(zClosedBeta);
            		mse(zPredictedClosedY, test_y, "Z score Closed-FormLinearRegression Mean Square Error: ");
            		printOutput(predictedClosedY);
                    break;
            case 2: Matrix bgdBeta = getbgdBeta(modifiedX(train_x), train_y);
            		Matrix predictedbgdY = modifiedX(test_x).times(bgdBeta);
            		mse(predictedbgdY, test_y, "Batch Gradient Descent Mean Square Error : ");
            		Matrix zBgdBeta = getbgdBeta(normalization(modifiedX(train_x),modifiedX(test_x),train_y,test_y, false), train_y);
            		Matrix zPredictedbgdY = normalization(modifiedX(train_x),modifiedX(test_x),train_y,test_y, true).times(zBgdBeta);
            		mse(zPredictedbgdY, test_y, "Z score Batch Gradient Descent Mean Square Error : ");
            		printOutput(predictedbgdY);
                    break;
            case 3: Matrix sgdBeta = getsgdBeta(modifiedX(train_x), train_y);
            		Matrix predictedsgdY = modifiedX(test_x).times(sgdBeta);
            		mse(predictedsgdY, test_y, "Stochastic Gradient Descent Mean Square Error : ");
            		Matrix zSgdBeta = getsgdBeta(normalization(modifiedX(train_x),modifiedX(test_x),train_y,test_y, false), train_y);
            		Matrix zPredictedsgdY = normalization(modifiedX(train_x),modifiedX(test_x),train_y,test_y,true).times(zSgdBeta);
            		mse(zPredictedsgdY, test_y, "Z score Stochastic Gradient Descent Mean Square Error : ");
            		printOutput(predictedsgdY);
            		break;
            case 4: System.exit(0);
        }
        
        System.out.println("Done"); 
    }
    
    /**  @params: X and Y matrix of training data
     * returns value of beta calculated using the formula beta = (X^T*X)^ -1)*(X^T*Y)
     */
    
 // start functions to calculate Closed LR
    private static Matrix getClosedBeta(Matrix trainX, Matrix trainY) {
    	Matrix xT = trainX.transpose();
    	Matrix result = xT.times(trainX).inverse().times(xT.times(trainY));
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
//    // end functions to calculate SGD
    
    // start functions to calculate BGD
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
    		yi = trainY.get(r, 0);
    		Matrix calMat = bi.times(xi.transpose());
    		double calVal = calMat.get(0, 0);
    		calVal -= yi;
    		sumXi.plusEquals(xi.times(2 * calVal));
    	}
    	return sumXi;
    } 
    
 // end functions to calculate BGD
    
    private static void mse(Matrix predictedY, Matrix trainY, String type){
    	double msqerr = 0;
    	int nRows = trainY.getRowDimension();
    	int nCols = trainY.getColumnDimension();
    	for (int r=0; r< nRows; r++){
    		for(int c=0; c< nCols; c++){
    			double trainy = trainY.get(r, c);
    			double predictedy = predictedY.get(r, c);
    			msqerr += (Math.pow((trainy - predictedy), 2));
    		}	
    	}
    	msqerr = msqerr/nRows;
    	System.out.print(type);
    	System.out.println(msqerr);
    } 
    
    private static Matrix normalization(Matrix trainX, Matrix testX, Matrix train_y, Matrix test_y, boolean test){
    	int nCols = trainX.getColumnDimension();
    	int nRows = trainX.getRowDimension();
    	Matrix zscore_trainX = new Matrix(nRows,nCols,1);
    	Matrix zscore_testX = new Matrix(nRows,nCols,1);
  	   	for(int c=1; c< nCols; c++){
  		   double[] temp = new double[nRows];
  		   int length = temp.length;
  		   double mean = 0;
  		   for(int r=0;r<nRows;r++){
  			   temp[r] = trainX.get(r, c);
  			   mean += temp[r];
  		   }
  		   mean = mean/length;
  		   double sigma = 0;
  		   for (int i = 0; i < length; i++){
  		       sigma += Math.pow((temp[i] - mean),2);
  		   }
  		   sigma = sigma/length;
  		   sigma = Math.sqrt(sigma);
  		   for(int r=0;r<nRows;r++){
	  			zscore_trainX.set(r,c,(trainX.get(r,c)-mean)/sigma);
	  			zscore_testX.set(r,c,(testX.get(r,c)-mean)/sigma);
  		  }   
  	   	}
  	   	if(test){
  		   return zscore_testX;
  	   	}else{
  		   return zscore_trainX;
			   } 
    }
    /**
     * @params: predicted Y matrix
     * outputs the predicted y values to the text file named "linear-regression-output"
     */
    public static void printOutput(Matrix predictedY) throws IOException {
        FileWriter fStream = new FileWriter("/Users/mounika/Documents/workspace/DataMiningHW1/output/linear_regression/linear-regression-output.txt");     // Output File
        BufferedWriter out = new BufferedWriter(fStream);
        for (int row =0; row<predictedY.getRowDimension(); row++) {
            out.write(String.valueOf(predictedY.get(row, 0)));
            out.newLine();
        }
        out.close();
    }
}
