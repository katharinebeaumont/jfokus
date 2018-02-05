package jfokus.nn;

import static spark.Spark.get;
import static spark.Spark.port;
import static spark.Spark.post;
import static spark.Spark.staticFiles;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Base64;
import java.util.HashMap;
import java.util.Map;

import org.deeplearning4j.datasets.mnist.MnistManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import spark.ModelAndView;
import spark.template.mustache.MustacheTemplateEngine;

/**
 * Hello world!
 *
 */
public class App 
{
	private static final Logger logger = LoggerFactory.getLogger(App.class);
	
	private static final int PORT = 4321;
	
    public static void main( String[] args )
    {
    		port(PORT);
    		staticFiles.location("/static/"); 
    		Controller c = new Controller(false);
    		
        get("/hello", (request, response) -> {
            Map<String, Object> model = new HashMap<>();
            model.put("message", "Hello !");
            return new ModelAndView(model, "hello.mustache"); 
        }, new MustacheTemplateEngine());
        
        post("/uploadImg", (request, response) -> {
        		String base64image = parseBody(request.body());
        		byte[] img = Base64.getMimeDecoder().decode(base64image.getBytes(StandardCharsets.UTF_8));
        		
        		File parentDir = new File("temp-mnist/label-unknown/");
        		parentDir.mkdirs();
        		File tempFile = new File(parentDir, "temp" + System.currentTimeMillis());
        		tempFile.createNewFile();
        		try (OutputStream stream = new FileOutputStream(tempFile)) {
        		    stream.write(img);
        		}
        		
        		String prediction = c.predictFromImage(tempFile);
        		logger.info("Predicting: " + prediction);
        		Map<String, Object> model = new HashMap<>();
            model.put("prediction", prediction);
            return new ModelAndView(model, "hello.mustache"); 
        }, new MustacheTemplateEngine());
        
    }
    
    private static String parseBody(String body) {
    		String startingIndex = "data:image/png;base64,";
    		int start = body.indexOf(startingIndex) + startingIndex.length();
    		int end = body.indexOf("------Web", start);
    		return body.substring(start,  end);
    }
}
