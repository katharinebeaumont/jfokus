package jfokus.nn;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Code from:
 * /dl4j-examples/src/main/java/org/deeplearning4j/examples/convolution/LenetMnistExample.java
 * See this ^^^ code for useful comments
 */
public class ConvolutionalNetwork {

	private static final Logger log = LoggerFactory.getLogger(ConvolutionalNetwork.class);

    int nChannels = 1; // Number of input channels
    int outputNum = 10; // Output classes, which is 10 (0,1,2,3,4,5,6,7,8,9)
    int batchSize = 64; // Test batch size
    int nEpochs = 1; // Number of training epochs
    int iterations = 1; // Number of training iterations
    int seed = 123; //
    MultiLayerNetwork model;
    
    public void train() throws IOException {
    		log.info("***Training model: Convolutional NN***");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize,true,12345);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize,false,12345);

        MultiLayerConfiguration conf = buildConfiguration();

        model = new MultiLayerNetwork(conf);
        model.init();

        model.setListeners(new ScoreIterationListener(1));
        for( int i=0; i<nEpochs; i++ ) {
            model.fit(mnistTrain);
            log.info("*** Completed epoch {} ***", i);

            log.info("***Evaluating model***");
            Evaluation eval = model.evaluate(mnistTest);
            log.info(eval.stats());
            mnistTest.reset();
        }
        log.info("****************Example finished********************");
    }
    
	public int evaluateNewExample(INDArray array) {
		log.info("Evaluating with unseen data");
        log.info("Shape: " + array.shapeInfoToString());
        
    		int[] prediction = model.predict(array);
    		return prediction[0];
	}

	private MultiLayerConfiguration buildConfiguration() {
        // learning rate schedule in the form of <Iteration #, Learning Rate>
        Map<Integer, Double> lrSchedule = new HashMap<>();
        lrSchedule.put(0, 0.01);
        lrSchedule.put(1000, 0.005);
        lrSchedule.put(3000, 0.001);
        
        return new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations) // Training iterations as above
                .regularization(true).l2(0.0005)
                /*
                    Uncomment the following for learning decay and bias
                 */
                .learningRate(.01)//.biasLearningRate(0.02)
                /*
                    Alternatively, you can use a learning rate schedule.

                    NOTE: this LR schedule defined here overrides the rate set in .learningRate(). Also,
                    if you're using the Transfer Learning API, this same override will carry over to
                    your new model configuration.
                */
                .learningRateDecayPolicy(LearningRatePolicy.Schedule)
                .learningRateSchedule(lrSchedule)
                /*
                    Below is an example of using inverse policy rate decay for learning rate
                */
                //.learningRateDecayPolicy(LearningRatePolicy.Inverse)
                //.lrPolicyDecayRate(0.001)
                //.lrPolicyPower(0.75)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS) //To configure: .updater(new Nesterovs(0.9))
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        //Note that nIn need not be specified in later layers
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(28,28,1)) //See note below
                .backprop(true).pretrain(false).build();
	}
}
