package DecisionTree;

import java.io.*;
import java.util.Enumeration;
import java.util.Scanner;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.Utils;
import Utility.Utility;

/*Class for constructing an unpruned decision tree based on
the ID3 algorithm. Can only deal with nominal attributes.
No missing values allowed. Empty leaves may result in unclassified instances.
*/
public class DecisionTree  {
    //The node's successors.
    private DecisionTree[] m_Successors;
    //Attribute used for splitting.
    private Attribute m_Attribute;
    //Class value if node is leaf.
    private double m_ClassValue;
    //Class distribution if node is leaf.
    private double[] m_Distribution;
    // Class attribute of data set.
    private Attribute m_ClassAttribute;
    int input = 1;
    public DecisionTree() {
    }
    //Builds decision tree classifier.
    public void buildClassifier(Instances data, int input) throws Exception {
    	
        data = new Instances(data);
        this.makeTree(data, input);
    }

    private void makeTree(Instances data,int input) throws Exception {
        if(data.numInstances() == 0) {
            this.m_Attribute = null;
            this.m_ClassValue = Instance.missingValue();
            this.m_Distribution = new double[data.numClasses()];
        } else {
            double[] infoGains = new double[data.numAttributes()];

            Attribute splitData;
            if(input == 1){
	            for(Enumeration attEnum = data.enumerateAttributes(); attEnum.hasMoreElements(); infoGains[splitData.index()] = this.computeInfoGain(data, splitData)) {
	                splitData = (Attribute)attEnum.nextElement();
	            }
            }else{
	            for(Enumeration attEnum = data.enumerateAttributes(); attEnum.hasMoreElements(); infoGains[splitData.index()] = this.computeGainRatio(data, splitData)) {
	                splitData = (Attribute)attEnum.nextElement();
	            }
            }
            this.m_Attribute = data.attribute(Utils.maxIndex(infoGains));
            if(Utils.eq(infoGains[this.m_Attribute.index()], 0.0D)) {
                this.m_Attribute = null;
                this.m_Distribution = new double[data.numClasses()];

                Instance j;
                for(Enumeration var6 = data.enumerateInstances(); var6.hasMoreElements(); ++this.m_Distribution[(int)j.classValue()]) {
                    j = (Instance)var6.nextElement();
                }

                Utils.normalize(this.m_Distribution);
                this.m_ClassValue = (double)Utils.maxIndex(this.m_Distribution);
                this.m_ClassAttribute = data.classAttribute();
            } else {
                Instances[] var7 = this.splitData(data, this.m_Attribute);
                this.m_Successors = new DecisionTree[this.m_Attribute.numValues()];

                for(int var8 = 0; var8 < this.m_Attribute.numValues(); ++var8) {
                    this.m_Successors[var8] = new DecisionTree();
                    this.m_Successors[var8].makeTree(var7[var8], input);
                }
            }
        }
    }
    //Classifies a given test instance using the decision tree.
    public double classifyInstance(Instance instance) throws NoSupportForMissingValuesException {
        if(instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("DecisionTree: no missing values, please.");
        } else {
            return this.m_Attribute == null?this.m_ClassValue:this.m_Successors[(int)instance.value(this.m_Attribute)].classifyInstance(instance);
         }
    }

    public String toString() {
        return this.m_Distribution == null && this.m_Successors == null?"DecisionTree: No model built yet.":"DecisionTree\n\n" + this.toString(0);
    }
    //Computes information gain for an attribute.
    private double computeInfoGain(Instances data, Attribute att) throws Exception {
        double infoGain = this.computeEntropy(data);
        Instances[] splitData = this.splitData(data, att);

        /****************Please Fill Missing Lines Here*****************/
        double attrInfo = 0;
        for(int i=0; i<splitData.length; i++){
        	double attrEntropy = this.computeEntropy(splitData[i]);
        	double calVal = (splitData[i].numInstances())*1.0/(data.numInstances())*1.0;
        	attrInfo = 1.0*(attrEntropy*calVal) + attrInfo;
        }
        infoGain -= attrInfo;
        return infoGain;
    }
    
    private double computeGainRatio(Instances data, Attribute att) throws Exception {
        double gainInfo=computeInfoGain(data,att);
    	double splitInfo=computeSplitInfo(data,att);
    	double gainRatio = gainInfo/splitInfo;
        return gainRatio;
    }

    private double computeSplitInfo(Instances data, Attribute att){
    	Instances[] splitData = this.splitData(data, att);
        double nInstances=0.0D;
        double gainInfoAtt=0.0D;
        nInstances=data.numInstances();
        for(int i=0;i<splitData.length;i++){       
            
        	Instances iSplitData = splitData[i];
        	double nSplitInstances = iSplitData.numInstances();
        	double instanceRatio=(nSplitInstances/nInstances)*1.0;
        	gainInfoAtt+=-1*instanceRatio*(Math.log(instanceRatio)/Math.log(2));
        }
        return gainInfoAtt;
    }
    
    //Computes the entropy of a dataset.
    private double computeEntropy(Instances data) throws Exception {
        double[] classCounts = new double[data.numClasses()];

        Instance entropy;
        for(Enumeration instEnum = data.enumerateInstances(); instEnum.hasMoreElements(); ++classCounts[(int)entropy.classValue()]) {
            entropy = (Instance)instEnum.nextElement();
        }

        double totalEntropy = 0.0D;
        int classNum = data.numClasses();
        double [] classProbVec = new double[classNum];
        
        for(int j = 0; j < classNum; ++j) {
            if(classCounts[j] > 0.0D) {
                classProbVec[j]= classCounts[j]/data.numInstances();
            }
            else
            	classProbVec[j]=0;
        }

        /****************Please Fill Missing Lines Here*****************/
        int length = (int)classProbVec.length;
        for(int i=0; i<length; i++){
        	if(classProbVec[i] != 0){
        		double calVal = (classProbVec[i]*Math.log(classProbVec[i])/Math.log(2))*1.0;
        		totalEntropy = (totalEntropy - calVal);
        	}
        }
        return (totalEntropy)*1.0;
    }
    //Splits a dataset according to the values of a nominal attribute.
    private Instances[] splitData(Instances data, Attribute att) {
        Instances[] splitData = new Instances[att.numValues()];

        for(int instEnum = 0; instEnum < att.numValues(); ++instEnum) {
            splitData[instEnum] = new Instances(data, data.numInstances());
        }

        Enumeration var6 = data.enumerateInstances();

        while(var6.hasMoreElements()) {
            Instance i = (Instance)var6.nextElement();
            splitData[(int)i.value(att)].add(i);
        }

        for(int var7 = 0; var7 < splitData.length; ++var7) {
            splitData[var7].compactify();
        }
        return splitData;
    }

    private String toString(int level) {
        StringBuffer text = new StringBuffer();
        if(this.m_Attribute == null) {
            if(Instance.isMissingValue(this.m_ClassValue)) {
                text.append(": null");
            } else {
                text.append(": " + this.m_ClassAttribute.value((int)this.m_ClassValue));
            }
        } else {
            for(int j = 0; j < this.m_Attribute.numValues(); ++j) {
                text.append("\n");

                for(int i = 0; i < level; ++i) {
                    text.append("|  ");
                }

                text.append(this.m_Attribute.name() + " = " + this.m_Attribute.value(j));
                text.append(this.m_Successors[j].toString(level + 1));
            }
        }
        return text.toString();
    }

    public void decisionTree() throws Exception {
        BufferedReader file = Utility.readFile("/Users/mounika/Documents/workspace/DataMiningHW1/data/decision_tree/congress.arff");
        Instances data = new Instances(file);
        int cIdx= 0; //data.numAttributes()-1;
        data.setClassIndex(cIdx);
        double count = 0.0;
        double accuracy = 0;
        double res = 0;
        System.out.println("\tLinear Regression");
        System.out.println("1) Information Gain");
        System.out.println("2) Gain Ratio");
        System.out.println("3) Exit\n");
        System.out.println("Enter the number corresponding to the algorithm you want to run \n *(it may take some time to run):");
        Scanner in = new Scanner(System.in);
        int choice = in.nextInt();
        switch(choice){
            case 1: input = 1;
            		break;
            case 2: input = 2;
            		break;
            case 3: System.exit(0);
            	}
        for (int n = 0; n < 5; n++) {
        	   Instances train = data.trainCV(5, n);
        	   Instances test = data.testCV(5, n);
        	   buildClassifier(train, input);
        	   for(int i=0; i<test.numInstances(); i++){
        		   count = printOutput(test);
        	   } 
        	   double cal = count/test.numInstances();
               accuracy = accuracy + cal;
                res = (accuracy * 100) / 5;   
        	 }
        System.out.println(res);
    }
    
    private double printOutput(Instances data) throws IOException, NoSupportForMissingValuesException {
        FileWriter fStream = new FileWriter("/Users/mounika/Documents/workspace/DataMiningHW1/output/decision_tree/decision-tree-output.txt");     // Output File
        BufferedWriter out = new BufferedWriter(fStream);
        double count = 0.0;
        //double accuracy = 0.0;
        for(int index =0; index<data.numInstances();index++) {
            Instance testRowInstance = data.instance(index);
            double prediction = classifyInstance(testRowInstance);
            double actualTarget = testRowInstance.classValue();
            if(actualTarget==prediction){	
            	count++;
            }
            out.write(data.classAttribute().value((int)prediction));
            out.newLine();
        }
        out.close();
        return count;
    }

//    private void printOutput(Instances data) throws IOException, NoSupportForMissingValuesException {
//        FileWriter fStream = new FileWriter("/Users/mounika/Documents/workspace/DataMiningHW1/output/decision_tree/decision-tree-output.txt");     // Output File
//        BufferedWriter out = new BufferedWriter(fStream);
//        for(int index =0; index<data.numInstances();index++) {
//            Instance testRowInstance = data.instance(index);
//            double prediction = classifyInstance(testRowInstance);
//            out.write(data.classAttribute().value((int) prediction));
//            out.newLine();
//        }
//        out.close();
//    }
}
