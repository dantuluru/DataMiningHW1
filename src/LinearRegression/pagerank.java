package LinearRegression;

import Jama.Matrix;

public class pagerank {
	 public  static void main(String args[]) throws Exception {
		  double beta = 0.8;
		  double N = 0.16666667;
	      double N2 = (1 - beta)*N;
	      
		  Matrix R = new Matrix(6,1);
		  Matrix PR = new Matrix(6,1);
	      Matrix Y = new Matrix(6,1);
	      Matrix X = new Matrix(6,1);
	      Matrix M = new Matrix(6,6);
	      
	      M.set(0, 3, 0.5);M.set(1, 2, 0.25);M.set(1, 5, 1);
	      M.set(2, 0, 0.5);M.set(2, 1, 1);M.set(3,0, 0.5);
	      M.set(3, 2, 0.25);M.set(3, 3, 0.5);M.set(4, 2, 0.25);
	      M.set(5, 2, 0.25);M.set(5, 4, 1);
	      
		 for(int r=0;r<6;r++){
			 for(int c=0;c<1;c++){
				 R.set(r, c, N);
				 PR.set(r,c,N);
				 X.set(r, c, N2);
			 }
		 } 
		
		 //Calculate Page Rank
		 System.out.println("Page rank score");
		 R = recallPageRank(M,R,beta,X);
		 R.print(2, 3);
		 
		//Calculate Personalized Page Rank
		 System.out.println("Personalized Page rank score");
		 Y.set(0, 0, 1.0);
		 Y.timesEquals(1 - beta);
		
		 PR = recallPageRank(M,PR,beta,Y);
		 PR.print(2, 3);
	 }
	 
	 public static Matrix recallPageRank(Matrix M,Matrix R,double beta,Matrix X){
		 Matrix Rnew = new Matrix(6,1);
		 while(checkequal(Rnew,R)){
			 R = Rnew;
			 Rnew = ((M.times(R)).times(beta)).plus(X);
		 }
		 
		return R=Rnew;		 
	 }
	 
	 public static boolean checkequal(Matrix Rnew, Matrix R){
		 boolean test = false;
		 for(int r=0; r<6;r++){
			 for(int c=0; c<1;c++){
				 if(R.get(r, c) != Rnew.get(r, c)){
					 test = true;
				 }
			 }
		 }
		 return test;
	 } 
	 
}
