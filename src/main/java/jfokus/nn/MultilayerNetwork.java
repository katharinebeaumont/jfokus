package jfokus.nn;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.IOException;
import java.util.List;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 
 * Lifted from: dl4j-examples: 
 * /dl4j-examples/src/main/java/org/deeplearning4j/examples/feedforward/mnist/MLPMnistTwoLayerExample.java
 * See this ^^^ code for useful comments
*/ 
public class MultilayerNetwork {

	private static Logger log = LoggerFactory.getLogger(MultilayerNetwork.class);

	private final int numRows = 28; //MNIST images are 28x28
	private final int numColumns = 28;
	private int outputNum = 10; // Number of output classes: here 0,1,2,3,4,5,6,7,8,9
	private int batchSize = 64; // Batch size for each epoch
	private int rngSeed = 123; // Random number seed for reproducibility
	private int numEpochs = 1; // Number of epochs to perform
	private double rate = 0.0015; // The learning rate
	private MultiLayerNetwork model;
	
    public void train() throws IOException {
         //Get the DataSetIterators:
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);

        //Set the configuration
        MultiLayerConfiguration conf = buildConfiguration();
        
        //Initialise the model
        model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(5));  //print the score with every iteration

        //Train the model
        log.info("***Training model: Feed Forward***");
        for( int i=0; i<numEpochs; i++ ){
        	log.info("Epoch " + i);
            model.fit(mnistTrain);
        }

        //See how it does on the test set
        log.info("***Evaluating model***");
        evaluate(mnistTest);
        log.info("****************Finished!********************");
    }

	private void evaluate(DataSetIterator mnistTest) {
		
        Evaluation eval = new Evaluation(outputNum);
        while(mnistTest.hasNext()){
            DataSet next = mnistTest.next();
            
            //What does the model predict?
            INDArray output = model.output(next.getFeatureMatrix()); 
            //How does it compare against the class?
            eval.eval(next.getLabels(), output); 
        }

        log.info(eval.stats());
    }
	
	public int evaluateNewExample(INDArray array) {
		int[] prediction = model.predict(array);
    		return prediction[0];
	}

	private MultiLayerConfiguration buildConfiguration() {
		log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(rngSeed) 
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) 
            .iterations(1) //Optimisation iterations
            .activation(Activation.RELU) //Our friend the rectified linear neuron
            .weightInit(WeightInit.XAVIER) //A weight initialisation scheme
            .learningRate(rate) 
            .updater(new Nesterovs(0.98)) //How to update the gradients (e.g. rmsprop)
            .regularization(true).l2(rate * 0.005) // Regularisation
            .list()
            .layer(0, new DenseLayer.Builder() //Input layer is 784 in, 500 out
                    .nIn(numRows * numColumns)
                    .nOut(500)
                    .build())
            .layer(1, new DenseLayer.Builder() //create the second input layer
                    .nIn(500) //Second layer is 500 in (has to match output of previous layer)
                    .nOut(100) //100 out
                    .build())
            .layer(2, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
                    .activation(Activation.SOFTMAX) //For the final layer, use Softmax to get a probability distribution for output
                    .nIn(100) //100 in, 10 out (number of classes)
                    .nOut(outputNum)
                    .build())
            .pretrain(false).backprop(true) //use backpropagation to adjust weights
            .build();
		return conf;
	}

}

