using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Text.Json;
using MyMachineLearningLibrary.Activation_Functions;
using MyMachineLearningLibrary.Optimizers;
using MyMachineLearningLibrary.Layers;
using MyMachineLearningLibrary.Loss_Functions;
using MyMachineLearningLibrary.Weight_Initialization;
using System.Diagnostics;

namespace MyMachineLearningLibrary
{
	public class NeuralNetwork
	{
		public IOptimizer Optimizer { get; set; }
		public List<ILayer> Layers { get; set; }
		public double LearningRate { get; set; }
		public int DecayRate { get; set; }
		public int NumberOfInputs { get; set; }
		public ILossFunction LossFunction { get; set; }
		public MatrixExtension Inputs { get; set; }
		private NeuralNetwork()
		{
			Layers = new List<ILayer>();
			LossFunction = new NotDefinedLossFunction();
			Optimizer = new GradientDescentOptimizer();
			Inputs = new MatrixExtension();
		}

		public NeuralNetwork(int NumberOfInputs, double LearningRate, int DecayRate, ILossFunction LossFunction) 
		{
			//Layers = [new InputLayer(NumberOfInputs)];
			Layers = [];
			this.LearningRate = LearningRate;
			this.NumberOfInputs = NumberOfInputs;
			this.LossFunction = LossFunction;
			this.DecayRate = DecayRate;
			Optimizer = new GradientDescentOptimizer();
			Inputs = new MatrixExtension(NumberOfInputs, 1);
		}

		public void Compile(IOptimizer optimizer, IWeightInitializtion? weightInitializtion = null)
		{
			weightInitializtion ??= new UniformXavierWeightInitialization();
			weightInitializtion.InitializeWeights(this);
			this.Optimizer = optimizer;
			optimizer.Compile(this);
		}

		public void AddLayer(ILayer layer)
		{
			if(layer.ActivationFunction.GetType() == typeof(SoftmaxActivationFunction) && LossFunction.GetType() != typeof(CategoricalCrossEntropyLossFunction))
			{
				throw new Exception("Softmax Activation can only be used with Categorical Cross Entropy in this Library");
			}

			if(Layers.Count > 0)
			{
				var previousLayer = Layers.Last();
				layer.InitializeLayer(previousLayer.NumberOfPerceptrons, Layers.Count);
			}
			else
			{
				layer.InitializeLayer(NumberOfInputs, Layers.Count);
			}
			
			Layers.Add(layer);
		}

		/// <summary>
		/// Feeds the input array through the neural network
		/// </summary>
		/// <param name="inputsArray"></param>
		/// <returns>The output of the neural network</returns>
		public double[] Predict(double[] inputsArray)
		{
			Inputs = new MatrixExtension(inputsArray);
			var result = new MatrixExtension(inputsArray);

			foreach(var layer in Layers)
			{
				result = layer.FeedForward(result);
			}

			return result.Flatten();
		}

		private int[] MakeClassification(double[] predictions)
		{
			var result = new int[predictions.Length];

			if (predictions.Length == 1)
			{
				var prediction = predictions[0];

				var t = Layers.Last().ActivationFunction.MinClass == 0 ? .5 : 0;

				result[0] = prediction >= t ? Layers.Last().ActivationFunction.MaxClass : Layers.Last().ActivationFunction.MinClass;

				return result;
			}

			int indexOfMaxValue = predictions.ToList().IndexOf(predictions.Max());
			result[indexOfMaxValue] = 1;

			return result;
		}

		/// <summary>
		/// Given an input to the neural network, return the predicted class
		/// </summary>
		/// <param name="inputsArray"></param>
		/// <returns>If the output layer has only 1 perceptron, return an int array size 1 with either a 1 or a 0. Otherwise, return an int array with a 1 marking the predicted class, and a 0 marking the other classes</returns>
		public int[] Classify(double[] inputsArray)
		{
			var predictions = Predict(inputsArray);
			return MakeClassification(predictions);
		}

		public void Train(int numberOfEpochs, double[][] trainInputsArray, double[][] trainTargetsArray, int batchSize = 1 /*Stochastic Gradient Descent by default*/, int reportingRate = 1)
		{
			for (int currentEpoch = 1; currentEpoch <= numberOfEpochs; currentEpoch++)
			{
				int batchCount = 0;

				var shuffledIndexes = Shuffle(trainInputsArray.Length); //Shuffle the indexes so that the inputs are given in a random order

				double cost = 0; // The cost is the average of all the loss

				double accuracy = 0;

				var currentErrors = new MatrixExtension(trainTargetsArray[0].Length, 1);
				for (int i = 0; i < trainInputsArray.Length; i++)
				{
					int k = shuffledIndexes[i];
					var inputsArray = trainInputsArray[k];
					var targetsArray = trainTargetsArray[k];

					var outputsArray = Predict(inputsArray);
					var outputsMatrix = Layers.Last().LayerOutputs;

					// Calculate the errors
					var targetsMatrix = new MatrixExtension(targetsArray);
					currentErrors = currentErrors.Add(LossFunction.CalculateDerivativeOfLoss(targetsMatrix, outputsMatrix));
					cost += LossFunction.CalculateLoss(targetsMatrix, outputsMatrix);

					// Calculate the accuracy
					var classifications = MakeClassification(outputsArray);
					accuracy = TestClassification(classifications, targetsArray) ? accuracy + 1 : accuracy;

					// Finish the batch before preceeding
					// Using Stochasitic Gradient Descent until I fix batch learning
					//if (!((i + 1) % batchSize == 0 || (i + 1) == trainInputsArray.Length))
					//	continue;

					// Backpropagate the errors
					for (int j = Layers.Count - 1; j >= 0; j--)
					{
						var currentBatchLength = (i + 1) == trainInputsArray.Length ? trainInputsArray.Length - (batchSize * batchCount) : batchSize;
						if(j > 0)
							currentErrors = Layers[j].Backpropagate(currentErrors, LearningRate, DecayRate, Layers[j - 1].LayerOutputs, currentBatchLength, currentEpoch, Optimizer);
						else
							currentErrors = Layers[j].Backpropagate(currentErrors, LearningRate, DecayRate, Inputs, currentBatchLength, currentEpoch, Optimizer);
					}
					batchCount++;
					currentErrors = new MatrixExtension(trainTargetsArray[0].Length, 1);
				}

				cost /= trainInputsArray.Length;
				accuracy /= trainInputsArray.Length;

				if (reportingRate > 0 && currentEpoch % reportingRate == 0)
					Console.WriteLine($"EPOCH: {currentEpoch}\tCOST: {cost}\tTRAIN ACCURACY: {accuracy}");
			}
		}

		private bool TestClassification(int[] classifications, double[] targets)
		{
			for (int i = 0; i < classifications.Length; i++)
			{
				if (classifications[i] != targets[i])
					return false;
			}
			return true;
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

			double accuracy = 0;
			for (int i = 0; i < classifications.Length; i++)
			{
				if(TestClassification(classifications[i], testTargetsArray[i]))
				{
					accuracy++;
				}
			}

			return accuracy / classifications.Length;
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
