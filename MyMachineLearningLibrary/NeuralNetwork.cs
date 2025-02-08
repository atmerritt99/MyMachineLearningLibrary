using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Text.Json;

namespace MyMachineLearningLibrary
{
	public class NeuralNetwork
	{
		public List<ILayer> Layers { get; set; }
		public double LearningRate { get; set; }
		public int NumberOfInputs { get; set; }
		public ILossFunction LossFunction { get; set; }

		private NeuralNetwork()
		{
			Layers = new List<ILayer>();
			LossFunction = new NotDefinedLossFunction();
		}

		public NeuralNetwork(int NumberOfInputs, double LearningRate, ILossFunction LossFunction) 
		{
			Layers = [new InputLayer(NumberOfInputs)];
			this.LearningRate = LearningRate;
			this.NumberOfInputs = NumberOfInputs;
			this.LossFunction = LossFunction;
		}

		public void AddLayer(ILayer layer)
		{
			if(layer.ActivationFunction.GetType() == typeof(SoftmaxActivationFunction) && LossFunction.GetType() != typeof(CategoricalCrossEntropyLossFunction))
			{
				throw new Exception("Softmax Activation can only be used with Categorical Cross Entropy in this Library");
			}

			var previousLayer = Layers.Last();
			layer.InitializeLayer(previousLayer.NumberOfPerceptrons, previousLayer.Gradients);
			Layers.Add(layer);
		}

		/// <summary>
		/// Feeds the input array through the neural network
		/// </summary>
		/// <param name="inputsArray"></param>
		/// <returns>The output of the neural network</returns>
		public double[] Predict(double[] inputsArray)
		{
			var result = new NeuralNetMatrix(inputsArray);

			foreach(var layer in Layers)
			{
				result = layer.FeedForward(result);
			}

			return result.Flatten();
		}

		/// <summary>
		/// Given an input to the neural network, return the predicted class
		/// </summary>
		/// <param name="inputsArray"></param>
		/// <returns>If the output layer has only 1 perceptron, return an int array size 1 with either a 1 or a 0. Otherwise, return an int array with a 1 marking the predicted class, and a 0 marking the other classes</returns>
		public int[] Classify(double[] inputsArray)
		{
			var predictions = Predict(inputsArray);
			var result = new int[predictions.Length];

			if (predictions.Length == 1)
			{
				var prediction = predictions[0];

				result[0] = prediction >= .5 ? 1 : 0;

				return result;
			}

			int indexOfMaxValue = predictions.ToList().IndexOf(predictions.Max());
			result[indexOfMaxValue] = 1;

			return result;
		}

		public void Train(int numberOfEpochs, double[][] trainInputsArray, double[][] trainTargetsArray, int batchSize = 1 /*Stochastic Gradient Descent by default*/, bool verbose = false)
		{
			for (int currentEpoch = 1; currentEpoch <= numberOfEpochs; currentEpoch++)
			{
				int batchCount = 0;

				var shuffledIndexes = Shuffle(trainInputsArray.Length); //Shuffle the indexes so that the inputs are given in a random order

				double cost = 0; // The cost is the average of all the loss

				for (int i = 0; i < trainInputsArray.Length; i++)
				{
					int k = shuffledIndexes[i];
					var inputsArray = trainInputsArray[k];
					var targetsArray = trainTargetsArray[k];

					Predict(inputsArray);
					var outputsMatrix = Layers.Last().LayerOutputs;

					// Calculate the errors
					var targetsMatrix = new NeuralNetMatrix(targetsArray);
					var currentErrors = LossFunction.CalculateDerivativeOfLoss(targetsMatrix, outputsMatrix);
					cost += LossFunction.CalculateLoss(targetsMatrix, outputsMatrix);

					// Backpropagate the errors
					for (int j = Layers.Count - 1; j > 0; j--)
					{
						currentErrors = Layers[j].Backpropagate(currentErrors, LearningRate, Layers[j - 1].LayerOutputs);
						//var layer = Layers[j];
						//var previousLayerOutputs = Layers[j - 1].LayerOutputs;

						////Calculate the output gradients
						//layer.CalculateGradients(currentErrors, LearningRate);

						////Apply Gradients
						//if ((i + 1) % batchSize == 0 || (i + 1) == trainInputsArray.Length)
						//{
						//	if ((i + 1) == trainInputsArray.Length)
						//	{
						//		int lengthFinalBatch = trainInputsArray.Length - (batchSize * batchCount);
						//		layer.ApplyGradients(previousLayerOutputs, lengthFinalBatch);
						//	}
						//	else
						//	{
						//		batchCount++;
						//		layer.ApplyGradients(previousLayerOutputs, batchSize);
						//	}
						//}

						//// Calculate this layer's Errors
						//var weightsTransposed = layer.TransposeWeights();
						//currentErrors = NeuralNetMatrix.DotProduct(weightsTransposed, currentErrors);
					}
				}

				cost /= trainInputsArray.Length;

				if (verbose)
					Console.WriteLine($"EPOCH: {currentEpoch}\tCOST: {cost}");
			}
		}

		/// <summary>
		/// Tests the models ability to classify the data passed into this function
		/// </summary>
		/// <param name="testInputsArray"></param>
		/// <param name="testTargetsArray"></param>
		/// <returns>The accuracy of the model</returns>
		public double Test(double[][] testInputsArray, double[][] testTargetsArray)
		{
			var classifications = new int[testInputsArray.Length][];
			for(int i = 0; i < testInputsArray.Length; i++)
			{
				classifications[i] = Classify(testInputsArray[i]);
			}

			double accuracy = 1;
			for (int i = 0; i < classifications.Length; i++)
			{
				// In the case that there is only 1 perceptron in the output layer, compare whether or not the predicted class is a 1 or not (-1)
				int predictedClass = classifications[i].ToList().IndexOf(1);
				int actualClass = testTargetsArray[i].ToList().IndexOf(1);

				if (predictedClass != actualClass)
				{
					accuracy -= (1.0 / testInputsArray.Length);
				}
			}

			return accuracy;
		}

		/// <summary>
		/// Serializes and saves the model to a JSON file
		/// </summary>
		/// <param name="filePath"></param>
		public void Save(string filePath)
		{
			var json = JsonSerializer.Serialize(this);
			File.WriteAllText(filePath, json);
		}

		/// <summary>
		/// Deserializes and loads the model from a JSON file
		/// </summary>
		/// <param name="filePath"></param>
		/// <returns>The deserialized model</returns>
		public static NeuralNetwork Load(string filePath) 
		{
			var json = File.ReadAllText(filePath);
			return JsonSerializer.Deserialize<NeuralNetwork>(json) ?? new NeuralNetwork();
		}

		/// <summary>
		/// This function fills an int array of size n with the numbers 0 - (n - 1) in a random order.
		/// This is used to feed the training inputs in a random order for better training outcomes
		/// </summary>
		/// <param name="max"></param>
		/// <returns></returns>
		private int[] Shuffle(int max)
		{
			int[] result = new int[max];

			for (int i = 0; i < max; i++)
			{
				result[i] = i;
			}

			int n = max;
			var rng = new Random();
			while (n > 1)
			{
				n--;
				int k = rng.Next(n + 1);
				int value = result[k];
				result[k] = result[n];
				result[n] = value;
			}

			return result;
		}
	}
}
