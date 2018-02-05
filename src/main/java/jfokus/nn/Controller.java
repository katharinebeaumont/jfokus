package jfokus.nn;

import java.io.File;
import java.io.IOException;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.transform.doubletransform.MinMaxNormalizer;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.ResizeImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.datasets.mnist.MnistImageFile;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.NDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Controller {

	private static final Logger logger = LoggerFactory.getLogger(Controller.class);
	private MultilayerNetwork ml;
	private ConvolutionalNetwork cn;
	
	public Controller(boolean trainCNN) {
		ml = new MultilayerNetwork();
		if (trainCNN)
			cn = new ConvolutionalNetwork();
		
		try {
			ml.train();
			logger.info("Finished training MultilayerNetwork");
			
			if (trainCNN)
				cn.train();
			logger.info("Finished training ConvolutionalNetwork");
		} catch (IOException io) {
			logger.info("Failed to train model");
		}
	}
	
	public String predictFromImage(File imageFile) throws IOException, InterruptedException {
		//First, we want to load the image in a new shape (28x28, in greyscale)
		NativeImageLoader nativeLoader = new NativeImageLoader(28, 28, 1);
		
		//As the network is trained on batches of 64, it's going to expect a batch of 64
		// examples to predict. But we're only giving it one. So create some 'empty' data
		// and labels.
		int[] shape = {1,784};
		int[] stride = {784,1};
		
		//Then add the loaded image
		// Which has been scaled between 28x28, but now we also need to 
		// normalise to between 0 and 1.
		INDArray inputWithImage = nativeLoader.asRowVector(imageFile);
		float[] img_array = new float[inputWithImage.length()];
		for (int i=0; i<img_array.length; i++) {
			img_array[i] = 1.0f - (inputWithImage.getFloat(i) / 255.0f);
		}

		NDArray newExample = new NDArray(img_array, shape, stride);
		
		String retval = "";
		if (cn != null) {
			int predFromCN = cn.evaluateNewExample(newExample);
			retval+= "Convolutional Network predicts " + predFromCN + ".";
		}
		int predFromML = ml.evaluateNewExample(newExample);
		retval += ". Multilayer Network predicts " + predFromML;
		return retval;
	}
}
