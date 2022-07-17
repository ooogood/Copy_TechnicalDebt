package main;

import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import process.DataReader;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.functions.RnnSequenceClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.stemmers.SnowballStemmer;
import weka.core.stopwords.WordsFromFile;
import weka.dl4j.NeuralNetConfiguration;
import weka.dl4j.activations.ActivationTanH;
import weka.dl4j.enums.GradientNormalization;
import weka.dl4j.iterators.instance.sequence.text.rnn.RnnTextEmbeddingInstanceIterator;
import weka.dl4j.layers.LSTM;
import weka.dl4j.layers.RnnOutputLayer;
import weka.dl4j.updater.Adam;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.StringToWordVector;
import domain.Document;
import model.EnsembleLearner;

public class Main {
	private static String rscdir = System.getProperty("user.dir") +"\\TechnicalDebt\\src\\main\\resources\\";

	public static void main(String args[]) throws Exception {

		List<String> projects = new ArrayList<String>();

		projects.add("argouml");
		projects.add("columba-1.4-src");
		projects.add("hibernate-distribution-3.3.2.GA");
		projects.add("jEdit-4.2");
		projects.add("jfreechart-1.0.19");
		projects.add("apache-jmeter-2.10");
		projects.add("jruby-1.4.0");
		projects.add("sql12");
		
		
		double ratio = 0.1;

		for (int target = 0; target < projects.size(); target++) {

			EnsembleLearner eLearner = new EnsembleLearner();
			List<Document> comments = DataReader.readComments(rscdir + "data\\");
			Set<String> projectForTesting = new HashSet<String>();
			projectForTesting.add(projects.get(target));

			// testDoc: all comments from the target project
			List<Document> testDoc = DataReader.selectProject(comments, projectForTesting);

			for (int source = 0; source < projects.size(); source++) {
				// skip target project
				if (source == target)
					continue;

				Set<String> projectForTraining = new HashSet<String>();
				projectForTraining.add(projects.get(source));

				// trainDoc: all comments from one project
				List<Document> trainDoc = DataReader.selectProject(comments, projectForTraining);

				// System.out.println("building dataset for training");
				String trainingDataPath = rscdir + "tmp\\trainingData.arff";
				DataReader.outputArffData(trainDoc, trainingDataPath);

				// System.out.println("building dataset for testing");
				String testingDataPath = rscdir + "tmp\\testingData.arff";
				DataReader.outputArffData(testDoc, testingDataPath);

				if (eLearner.getTestData() == null) {
					Instances tmp = DataSource.read(testingDataPath);
					tmp.setClassIndex(1);
					eLearner = new EnsembleLearner(tmp);
				}

//				// string to word vector (both for training and testing data)
//				StringToWordVector stw = new StringToWordVector(100000);
//				stw.setOutputWordCounts(true);
//				stw.setIDFTransform(true);
//				stw.setTFTransform(true);
//				SnowballStemmer stemmer = new SnowballStemmer();
//				stw.setStemmer(stemmer);
//				WordsFromFile stopwords = new WordsFromFile();
//				stopwords.setStopwords(new File(rscdir + "dic\\stopwords.txt"));
//				stw.setStopwordsHandler(stopwords);
//				Instances trainSet = DataSource.read(trainingDataPath);
//				Instances testSet = DataSource.read(testingDataPath);
//				stw.setInputFormat(trainSet);
//				trainSet = Filter.useFilter(trainSet, stw);
//				trainSet.setClassIndex(0);
//				testSet = Filter.useFilter(testSet, stw);
//				testSet.setClassIndex(0);
//
//				// attribute selection for training data
//				AttributeSelection attSelection = new AttributeSelection();
//				Ranker ranker = new Ranker();
//				ranker.setNumToSelect((int) (trainSet.numAttributes() * ratio));
//				InfoGainAttributeEval ifg = new InfoGainAttributeEval();
//				attSelection.setEvaluator(ifg);
//				attSelection.setSearch(ranker);
//				attSelection.setInputFormat(trainSet);
//				trainSet = Filter.useFilter(trainSet, attSelection);
//				testSet = Filter.useFilter(testSet, attSelection);
//
//				Classifier classifier = new MultilayerPerceptron();
				Instances trainSet = new Instances( new FileReader(trainingDataPath));
				trainSet.setClassIndex(1);
				Instances testSet = new Instances( new FileReader(testingDataPath));

				// Download e.g the SLIM Google News model from
// https://github.com/eyaler/word2vec-slim/raw/master/GoogleNews-vectors-negative300-SLIM.bin.gz
				final File modelSlim = new File(rscdir + "GoogleNews-vectors-negative300-SLIM.bin");

// Setup hyperparameters
				final int truncateLength = 80;
				final int batchSize = 64;
				final int seed = 1;
				final int numEpochs = 10;
				final int tbpttLength = 20;
				final double l2 = 1e-5;
				final double gradientThreshold = 1.0;
				final double learningRate = 0.02;

// Setup the iterator
				RnnTextEmbeddingInstanceIterator tii = new RnnTextEmbeddingInstanceIterator();
				tii.setWordVectorLocation(modelSlim);
				tii.setTruncateLength(truncateLength);
				tii.setTrainBatchSize(batchSize);

// Initialize the classifier
				RnnSequenceClassifier classifier = new RnnSequenceClassifier();
				classifier.setSeed(seed);
				classifier.setNumEpochs(numEpochs);
				classifier.setInstanceIterator(tii);
				classifier.settBPTTbackwardLength(tbpttLength);
				classifier.settBPTTforwardLength(tbpttLength);

// Define the layers
				LSTM lstm = new LSTM();
				lstm.setNOut(64);
				lstm.setActivationFunction(new ActivationTanH());

				RnnOutputLayer rnnOut = new RnnOutputLayer();

// Network config
				NeuralNetConfiguration nnc = new NeuralNetConfiguration();
				Adam adam = new Adam();
				adam.setLearningRate(learningRate);
				nnc.setUpdater( adam );
				nnc.setL2(l2);
				nnc.setGradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue);
				nnc.setGradientNormalizationThreshold(gradientThreshold);

// Config classifier
				classifier.setLayers(lstm, rnnOut);
				classifier.setNeuralNetConfiguration(nnc);

				classifier.buildClassifier(trainSet);

				SerializationHelper.write(rscdir+"classifier\\"+projects.get(source)+".rnn", classifier);

				for (int i = 0; i < testSet.numInstances(); i++) {
					Instance instance = testSet.instance(i);
					double score = 0;
					if (classifier.classifyInstance(instance) == 1.0) {
						score = 1;
					} else
						score = -1;
					eLearner.vote(i, score);

				}

			}

			System.out.print("target: " + projects.get(target) + " -- ");

			eLearner.evaluate();

		}

	}

}
